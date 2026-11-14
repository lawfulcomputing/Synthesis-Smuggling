"""Build and execute an isolated reference-augmentation generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase2.configuration import load_defaults
from synthetic_dataset_gen.phase2.output import write_json, write_jsonl
from synthetic_dataset_gen.phase2.pricing import (
    PricingRegistry,
    build_usage_record,
    usage_summary,
)
from synthetic_dataset_gen.phase2.registry import ModelAdapter, ModelRegistry
from synthetic_dataset_gen.phase2.requests import parse_batch_output_line
from synthetic_dataset_gen.phase2.run_management import package_versions, utc_now
from synthetic_dataset_gen.phase2.transport import (
    create_openai_client,
    download_file_text,
    submit_batch,
    wait_for_batch,
)
from synthetic_dataset_gen.reference_augmentation.catalog import (
    build_catalog,
    load_reference_config,
)
from synthetic_dataset_gen.reference_augmentation.models import (
    ReferenceGenerationRecord,
    ReferenceSpec,
)
from synthetic_dataset_gen.reference_augmentation.prompts import (
    PROMPTS_DIR,
    TEMPLATE_NAME,
    build_batch_line,
)
from synthetic_dataset_gen.reference_augmentation.validation import validate_generation


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Generate the standalone gender-reference augmentation"
    )
    value.add_argument("--config", default="config/reference_augmentation.yaml")
    value.add_argument("--phase1-config", default="config.toml")
    value.add_argument("--models-config", default="config/models.yaml")
    value.add_argument("--pricing-config", default="config/pricing.yaml")
    value.add_argument("--defaults-config", default="config/defaults.yaml")
    value.add_argument("--run-dir")
    value.add_argument("--limit", type=int)
    value.add_argument("--dry-run", action="store_true")
    return value


def _create_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing run directory: {path}")
    path.mkdir(parents=True)
    (path / "prompt_snapshots").mkdir()
    shutil.copy2(PROMPTS_DIR / TEMPLATE_NAME, path / "prompt_snapshots" / TEMPLATE_NAME)


def _batch_items(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = str(item.get("custom_id", ""))
        result[custom_id.removeprefix("reference-")] = line
    return result


def _counts(specs: list[ReferenceSpec]) -> dict[str, Any]:
    return {
        "categories": dict(Counter(spec.category for spec in specs)),
        "splits": dict(Counter(spec.split for spec in specs)),
        "reference_families": dict(Counter(spec.reference_family for spec in specs)),
        "reference_genders": dict(Counter(spec.reference_gender for spec in specs)),
        "target_labels": dict(Counter(spec.target_gender or "N" for spec in specs)),
        "unique_reference_terms": len({spec.reference_term for spec in specs}),
    }


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = load_reference_config(args.config)
    run_dir = Path(args.run_dir or config.outputs["run_dir"])
    catalog = build_catalog(config, phase1_path=args.phase1_config)
    if args.limit is not None:
        if not 1 <= args.limit <= len(catalog):
            raise ValueError("--limit must be within the catalog size")
        # A pilot samples evenly through the full deterministic catalog rather
        # than taking only the first category.
        indices = [
            round(i * (len(catalog) - 1) / max(args.limit - 1, 1))
            for i in range(args.limit)
        ]
        catalog = [catalog[index] for index in indices]
    specs = [item[0] for item in catalog]
    phase1_by_id = {item[0].conversation_id: item[1] for item in catalog}

    registry = ModelRegistry.load(args.models_config)
    model = registry.resolve(config.generation["model"])
    adapter = ModelAdapter(
        model,
        config.generation.get("reasoning_effort"),
        config.generation.get("verbosity"),
        int(config.generation["max_output_tokens"]),
    )
    pricing = PricingRegistry.load(args.pricing_config)
    defaults = load_defaults(args.defaults_config)
    requests = {spec.conversation_id: build_batch_line(spec, adapter) for spec in specs}

    _create_directory(run_dir)
    write_jsonl(run_dir / "reference_specs.jsonl", specs)
    write_jsonl(
        run_dir / "phase1_specs.jsonl",
        [phase1_by_id[spec.conversation_id] for spec in specs],
    )
    write_jsonl(run_dir / "requests.jsonl", requests.values(), sort_keys=False)
    manifest: dict[str, Any] = {
        "status": "dry_run_complete" if args.dry_run else "initialized",
        "started_at": utc_now(),
        "completed_at": None,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "phase1_config_path": str(Path(args.phase1_config).resolve()),
        "phase1_config_sha256": sha256_file(args.phase1_config),
        "prompt_template": TEMPLATE_NAME,
        "prompt_sha256": sha256_file(PROMPTS_DIR / TEMPLATE_NAME),
        "requested_model": config.generation["model"],
        "resolved_model": model.snapshot,
        "execution_mode": config.generation["execution_mode"],
        "number_requested": len(specs),
        "number_completed": 0,
        "number_failed": 0,
        "design_counts": _counts(specs),
        "batch_ids": [],
        "pricing_table": pricing.table,
        "model_registry": registry.date,
        "package_versions": package_versions(),
        "api_key_stored": False,
        "source_dataset_modified": False,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, indent=2))
        return

    if config.generation["execution_mode"] != "batch":
        raise ValueError("reference augmentation requires batch execution")
    client = create_openai_client()
    phase1_config = load_config(args.phase1_config)
    completed: dict[str, ReferenceGenerationRecord] = {}
    usage_records = []
    rejected: list[dict[str, Any]] = []
    missing = list(specs)
    max_attempts = int(config.generation["max_attempts"])
    for attempt in range(1, max_attempts + 1):
        if not missing:
            break
        request_path = run_dir / f"requests_attempt_{attempt}.jsonl"
        write_jsonl(
            request_path,
            [requests[spec.conversation_id] for spec in missing],
            sort_keys=False,
        )
        batch, _ = submit_batch(
            client,
            request_path,
            defaults.batch.completion_window,
            f"reference_augmentation_attempt_{attempt}",
            defaults.retry,
        )
        manifest["batch_ids"].append(str(batch.id))
        manifest["status"] = "batch_submitted"
        write_json(run_dir / "run_manifest.json", manifest)
        batch = wait_for_batch(
            client,
            str(batch.id),
            defaults.retry,
            defaults.batch.poll_interval_seconds,
        )
        if getattr(batch, "status", None) != "completed":
            raise RuntimeError(
                f"batch ended with status {getattr(batch, 'status', None)}"
            )
        output_id = getattr(batch, "output_file_id", None)
        if not output_id:
            raise RuntimeError("completed batch has no output file")
        text = download_file_text(client, output_id, defaults.retry)
        (run_dir / f"batch_output_attempt_{attempt}.jsonl").write_text(
            text, encoding="utf-8"
        )
        items = _batch_items(text)
        next_missing: list[ReferenceSpec] = []
        for spec in missing:
            line = items.get(spec.conversation_id)
            if line is None:
                rejected.append(
                    {
                        "conversation_id": spec.conversation_id,
                        "attempt": attempt,
                        "problems": ["missing batch output"],
                    }
                )
                next_missing.append(spec)
                continue
            try:
                parsed = parse_batch_output_line(line, spec.num_turns)
                usage = build_usage_record(
                    conversation_id=spec.conversation_id,
                    response_id=parsed.response_id,
                    model=model.snapshot,
                    pricing_key=model.pricing_key,
                    execution_mode="batch",
                    usage=parsed.usage,
                    registry=pricing,
                )
                problems = validate_generation(
                    spec,
                    phase1_by_id[spec.conversation_id],
                    parsed.output,
                    phase1_config,
                )
                usage.rejected = bool(problems)
                usage_records.append(usage)
                if problems:
                    rejected.append(
                        {
                            "conversation_id": spec.conversation_id,
                            "attempt": attempt,
                            "problems": problems,
                            "generated": parsed.output.model_dump(mode="json"),
                        }
                    )
                    next_missing.append(spec)
                    continue
                completed[spec.conversation_id] = ReferenceGenerationRecord(
                    conversation_id=spec.conversation_id,
                    response_id=parsed.response_id,
                    requested_model=config.generation["model"],
                    resolved_model=model.snapshot,
                    returned_model=parsed.returned_model,
                    prompt_template_name=TEMPLATE_NAME,
                    phase1_spec=phase1_by_id[spec.conversation_id],
                    augmentation_spec=spec,
                    generated=parsed.output,
                    pair_complete=None,
                    check_passed=True,
                    check_problems=[],
                    source_run=run_dir.name,
                )
            except Exception as error:  # noqa: BLE001
                rejected.append(
                    {
                        "conversation_id": spec.conversation_id,
                        "attempt": attempt,
                        "problems": [f"{error.__class__.__name__}: {error}"],
                    }
                )
                next_missing.append(spec)
        missing = next_missing
        manifest.update(
            {
                "status": "validating" if missing else "complete",
                "number_completed": len(completed),
                "number_failed": len(missing),
                "attempts_completed": attempt,
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)

    ordered = [
        completed[spec.conversation_id]
        for spec in specs
        if spec.conversation_id in completed
    ]
    write_jsonl(run_dir / "generations.jsonl", ordered)
    write_jsonl(run_dir / "rejected_attempts.jsonl", rejected)
    write_jsonl(run_dir / "usage.jsonl", usage_records)
    summary = usage_summary(
        usage_records,
        requested=len(specs),
        failed=len(missing),
        model=model.snapshot,
        execution_mode="batch",
        pricing_table=pricing.table,
    )
    write_json(run_dir / "usage_summary.json", summary)
    manifest.update(
        {
            "status": "complete" if not missing else "completed_with_errors",
            "completed_at": utc_now(),
            "number_completed": len(completed),
            "number_failed": len(missing),
            "generations_sha256": (
                sha256_file(run_dir / "generations.jsonl") if ordered else None
            ),
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {"run_dir": str(run_dir), "manifest": manifest, "usage": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

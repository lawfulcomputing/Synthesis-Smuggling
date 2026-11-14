"""Prepare or explicitly execute the isolated possessive recovery run."""

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
from synthetic_dataset_gen.possessive_recovery.catalog import (
    build_catalog,
    load_recovery_config,
)
from synthetic_dataset_gen.possessive_recovery.models import (
    RecoveryGenerationRecord,
    RecoverySpec,
)
from synthetic_dataset_gen.possessive_recovery.prompts import (
    PROMPTS_DIR,
    build_batch_line,
    template_name,
)
from synthetic_dataset_gen.possessive_recovery.validation import validate_generation


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the possessive-partner recovery set; API use requires --execute"
    )
    parser.add_argument("--config", default="config/possessive_role_controls.yaml")
    parser.add_argument("--phase1-config", default="config.toml")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--pricing-config", default="config/pricing.yaml")
    parser.add_argument("--defaults-config", default="config/defaults.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly authorize API submission; omitted means local dry run",
    )
    return parser


def _create_run(path: Path, template_name: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing run directory: {path}")
    path.mkdir(parents=True)
    (path / "prompt_snapshots").mkdir()
    shutil.copy2(PROMPTS_DIR / template_name, path / "prompt_snapshots" / template_name)


def _batch_items(text: str) -> dict[str, str]:
    items = {}
    for line in text.splitlines():
        if line.strip():
            item = json.loads(line)
            identifier = str(item.get("custom_id", "")).removeprefix("recovery-")
            items[identifier] = line
    return items


def _design_counts(specs: list[RecoverySpec]) -> dict[str, Any]:
    cross = Counter(
        (spec.category, spec.split, spec.partner_term, spec.target_gender or "N")
        for spec in specs
    )
    return {
        "categories": dict(Counter(spec.category for spec in specs)),
        "splits": dict(Counter(spec.split for spec in specs)),
        "target_labels": dict(Counter(spec.target_gender or "N" for spec in specs)),
        "partner_terms": dict(Counter(spec.partner_term for spec in specs)),
        "cue_buckets": dict(Counter(spec.cue_bucket for spec in specs)),
        "surface_frames": dict(Counter(spec.surface_frame for spec in specs)),
        "category_split_term_label": [
            {
                "category": key[0],
                "split": key[1],
                "term": key[2],
                "label": key[3],
                "count": count,
            }
            for key, count in sorted(cross.items())
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_recovery_config(args.config)
    template = template_name(config.recovery_set)
    run_dir = Path(args.run_dir or config.outputs["run_dir"])
    catalog = build_catalog(config, phase1_path=args.phase1_config)
    specs = [spec for spec, _ in catalog]
    phase1_by_id = {spec.conversation_id: phase1 for spec, phase1 in catalog}
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

    _create_run(run_dir, template)
    write_jsonl(run_dir / "recovery_specs.jsonl", specs)
    write_jsonl(
        run_dir / "phase1_specs.jsonl",
        [phase1_by_id[spec.conversation_id] for spec in specs],
    )
    write_jsonl(run_dir / "requests.jsonl", requests.values(), sort_keys=False)
    manifest: dict[str, Any] = {
        "recovery_set": config.recovery_set,
        "status": "dry_run_complete" if not args.execute else "initialized",
        "started_at": utc_now(),
        "completed_at": None,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _sha256(args.config),
        "phase1_config_path": str(Path(args.phase1_config).resolve()),
        "phase1_config_sha256": _sha256(args.phase1_config),
        "prompt_template": template,
        "prompt_sha256": _sha256(PROMPTS_DIR / template),
        "requested_model": config.generation["model"],
        "resolved_model": model.snapshot,
        "execution_mode": config.generation["execution_mode"],
        "number_requested": len(specs),
        "number_completed": 0,
        "number_failed": 0,
        "design_counts": _design_counts(specs),
        "batch_ids": [],
        "pricing_table": pricing.table,
        "model_registry": registry.date,
        "package_versions": package_versions(),
        "api_key_stored": False,
        "source_dataset_modified": False,
        "api_submission_authorized": bool(args.execute),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if not args.execute:
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, indent=2))
        return

    if config.generation["execution_mode"] != "batch":
        raise ValueError("possessive recovery requires batch execution")
    client = create_openai_client()
    phase1_config = load_config(args.phase1_config)
    completed: dict[str, RecoveryGenerationRecord] = {}
    usage_records = []
    rejected: list[dict[str, Any]] = []
    missing = list(specs)
    for attempt in range(1, int(config.generation["max_attempts"]) + 1):
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
            f"{config.recovery_set}_attempt_{attempt}",
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
        next_missing = []
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
                problems = validate_generation(spec, parsed.output, phase1_config)
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
                completed[spec.conversation_id] = RecoveryGenerationRecord(
                    conversation_id=spec.conversation_id,
                    response_id=parsed.response_id,
                    requested_model=config.generation["model"],
                    resolved_model=model.snapshot,
                    returned_model=parsed.returned_model,
                    prompt_template_name=template,
                    phase1_spec=phase1_by_id[spec.conversation_id],
                    recovery_spec=spec,
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
                _sha256(run_dir / "generations.jsonl") if ordered else None
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

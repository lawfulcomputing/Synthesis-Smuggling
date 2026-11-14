"""Isolated run directories, manifests, compatibility, and audit logging."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase2.configuration import RunOptions
from synthetic_dataset_gen.phase2.output import safe_message, write_json


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    resolved_config: Path
    selected_specs: Path
    prompt_snapshots: Path
    requests: Path
    generations: Path
    generations_csv: Path
    rejected_samples: Path
    usage: Path
    usage_summary: Path
    errors: Path
    log: Path
    batch_output: Path
    plan_requests: Path
    plans: Path
    plan_errors: Path
    plan_usage: Path
    plan_usage_summary: Path

    @classmethod
    def at(cls, root: str | Path) -> RunPaths:
        root = Path(root)
        return cls(
            root=root,
            manifest=root / "run_manifest.json",
            resolved_config=root / "resolved_config.json",
            selected_specs=root / "selected_specs.jsonl",
            prompt_snapshots=root / "prompt_snapshots",
            requests=root / "requests.jsonl",
            generations=root / "generations.jsonl",
            generations_csv=root / "generations.csv",
            rejected_samples=root / "rejected_samples.jsonl",
            usage=root / "usage.jsonl",
            usage_summary=root / "usage_summary.json",
            errors=root / "errors.jsonl",
            log=root / "run.log",
            batch_output=root / "batch_output.jsonl",
            plan_requests=root / "plan_requests.jsonl",
            plans=root / "plans.jsonl",
            plan_errors=root / "plan_errors.jsonl",
            plan_usage=root / "plan_usage.jsonl",
            plan_usage_summary=root / "plan_usage_summary.json",
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_run_directory(
    run_root: str | Path, phase_run: str, resume: bool
) -> RunPaths:
    paths = RunPaths.at(Path(run_root) / phase_run)
    if resume:
        if not paths.root.is_dir():
            raise FileNotFoundError(
                f"cannot resume missing run directory: {paths.root}"
            )
        return paths
    if paths.root.exists():
        raise FileExistsError(
            f"run directory already exists: {paths.root}; choose a new --phase-run "
            "or use --resume"
        )
    paths.root.mkdir(parents=True)
    paths.prompt_snapshots.mkdir()
    for path in (
        paths.generations,
        paths.usage,
        paths.errors,
        paths.log,
        paths.plans,
        paths.plan_errors,
        paths.plan_usage,
    ):
        path.touch()
    return paths


def core_configuration(
    options: RunOptions,
    *,
    resolved_model: str,
    source_catalog_hash: str,
) -> dict[str, Any]:
    return {
        "requested_model": options.requested_model,
        "resolved_model": resolved_model,
        "source_catalog": str(Path(options.source_catalog).resolve()),
        "source_catalog_hash": source_catalog_hash,
        "number_requested": options.conversations,
        "selectors": options.selectors.active(),
        "seed": options.seed,
        "execution_mode": options.execution_mode,
        "output_format": options.output_format,
        "reasoning_effort": options.reasoning_effort,
        "verbosity": options.verbosity,
        "max_output_tokens": options.max_output_tokens,
    }


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def package_versions() -> dict[str, str]:
    packages = ("synthetic-dataset-gen", "openai", "pydantic", "jinja2")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_resume_compatible(
    manifest: dict[str, Any], configuration: dict[str, Any]
) -> None:
    observed = configuration_fingerprint(configuration)
    if observed != manifest.get("core_configuration_sha256"):
        raise ValueError(
            "resume configuration is incompatible with the existing run; model, "
            "catalog, count, seed, selectors, execution settings, "
            "and output settings must match"
        )


def append_log(path: str | Path, message: str) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {safe_message(message)}\n")


def initial_manifest(
    *,
    options: RunOptions,
    configuration: dict[str, Any],
    resolved_model: str,
    model_family: str,
    source_catalog_hash: str,
    pricing_table: str,
    model_registry: str,
) -> dict[str, Any]:
    return {
        "phase_run": options.phase_run,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "initialized",
        "requested_model": options.requested_model,
        "resolved_model": resolved_model,
        "returned_model_if_available": None,
        "model_family": model_family,
        "source_catalog_path": str(Path(options.source_catalog).resolve()),
        "source_catalog_hash": source_catalog_hash,
        "number_requested": options.conversations,
        "number_completed": 0,
        "number_failed": 0,
        "active_selectors": options.selectors.active(),
        "seed": options.seed,
        "execution_mode": options.execution_mode,
        "output_format": options.output_format,
        "reasoning_effort": configuration["reasoning_effort"],
        "verbosity": configuration["verbosity"],
        "pricing_table": pricing_table,
        "model_registry": model_registry,
        "package_versions": package_versions(),
        "core_configuration_sha256": configuration_fingerprint(configuration),
        "batch_id": None,
        "api_key_stored": False,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)

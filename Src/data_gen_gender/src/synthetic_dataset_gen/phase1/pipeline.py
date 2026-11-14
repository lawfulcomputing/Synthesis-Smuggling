"""End-to-end orchestration for Phase 1 (strictly no external API calls)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase1.analysis import (
    association_matrix,
    build_summary,
    expand_prefixes,
    write_association_csv,
)
from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.generator import generate_catalog
from synthetic_dataset_gen.phase1.io import (
    write_catalog_csv,
    write_json,
    write_jsonl,
    write_prefix_csv,
)
from synthetic_dataset_gen.phase1.plotting import plot_catalog
from synthetic_dataset_gen.phase1.validation import (
    validate_catalog,
    write_validation_report,
)


def generate_artifacts(
    rows: list, config: Phase1Config, artifact_dir: Path
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(artifact_dir / "catalog.jsonl", rows)
    write_catalog_csv(artifact_dir / "catalog.csv", rows)
    prefixes = expand_prefixes(rows)
    write_prefix_csv(artifact_dir / "prefix_catalog.csv", prefixes)
    summary = build_summary(rows, config)
    write_json(artifact_dir / "catalog_summary.json", summary)
    write_association_csv(artifact_dir / "cramers_v_matrix.csv", summary["cramers_v"])
    return summary


def run_phase1(config: Phase1Config, project_root: str | Path = ".") -> dict[str, Any]:
    """Generate, validate, analyze and plot the catalog."""
    root = Path(project_root)
    artifact_dir = root / config.outputs.artifact_dir
    plot_dir = root / config.outputs.plot_dir

    rows = generate_catalog(config)
    summary = generate_artifacts(rows, config, artifact_dir)
    reproduced = generate_catalog(config)
    report = validate_catalog(rows, config, reproduced_rows=reproduced)
    write_validation_report(artifact_dir / "validation_report.md", report)
    if not report["passed"]:
        failed = [
            name for name, value in report["hard_checks"].items() if not value["passed"]
        ]
        raise RuntimeError(f"hard validation failed: {', '.join(failed)}")

    matrix = association_matrix(rows)
    plot_paths = plot_catalog(rows, config, plot_dir, matrix)
    return {
        "summary": summary,
        "validation": report,
        "plot_count": len(plot_paths),
    }

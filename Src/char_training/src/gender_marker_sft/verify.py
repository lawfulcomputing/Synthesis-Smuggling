from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_config
from .constants import SPLITS
from .data import (
    audit_source_records,
    read_jsonl,
    sha256_file,
    transform_record,
    validate_prepared_record,
)


def verify_setup(config_path: str | Path) -> dict[str, Any]:
    config, project_root = load_config(config_path)
    source = Path(config["paths"]["source_data"])
    prepared_dir = Path(config["paths"]["prepared_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    for path in (source, prepared_dir, output_dir):
        try:
            path.resolve().relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Experiment path escapes Training/: {path}") from exc

    source_records = list(read_jsonl(source))
    transformed = [transform_record(record) for record in source_records]
    label_counts = Counter(record["target_gender"] for record in transformed)
    split_counts = Counter(record["split"] for record in transformed)
    expected_labels = config["dataset"]["expected_label_counts"]
    expected_splits = config["dataset"]["expected_split_counts"]
    if dict(label_counts) != {label: n for label, n in expected_labels.items() if n}:
        raise ValueError(
            f"Source label counts are {dict(label_counts)}, expected {expected_labels}"
        )
    if dict(split_counts) != expected_splits:
        raise ValueError(
            f"Source split counts are {dict(split_counts)}, expected {expected_splits}"
        )

    prepared_report: dict[str, Any] = {}
    for split in SPLITS:
        path = prepared_dir / f"{split}.jsonl"
        if not path.exists():
            prepared_report[split] = {"exists": False}
            continue
        records = list(read_jsonl(path))
        for record in records:
            validate_prepared_record(record)
            if record["split"] != split:
                raise ValueError(f"{record['conversation_id']}: stored in wrong split")
        prepared_report[split] = {
            "exists": True,
            "records": len(records),
            "label_counts": dict(
                Counter(record["target_gender"] for record in records)
            ),
            "sha256": sha256_file(path),
        }

    return {
        "status": "ok",
        "project_root": str(project_root),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "records": len(source_records),
            "label_counts": dict(label_counts),
            "split_counts": dict(split_counts),
            "audit": audit_source_records(source_records),
        },
        "prepared": prepared_report,
        "lora": {
            "rank": config["lora"]["rank"],
            "alpha": config["lora"]["alpha"],
            "scaling": config["lora"]["alpha"] / config["lora"]["rank"],
            "target_modules": config["lora"]["target_modules"],
        },
        "training_was_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline structural verification (loads no model weights)"
    )
    parser.add_argument("--config", default="char_gender/configs/base/qwen2_5_7b_lora.yaml")
    parser.add_argument("--write-report", default="artifacts/setup_verification.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, project_root = load_config(args.config)
    report = verify_setup(args.config)
    report_path = Path(args.write_report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing verification report: {report_path}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

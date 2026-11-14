"""Stable catalog and analysis file I/O helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase1.models import CatalogRow, PrefixRow

CATALOG_COLUMNS = list(CatalogRow.model_fields)
PREFIX_COLUMNS = list(PrefixRow.model_fields)


def semantic_json(row: CatalogRow) -> str:
    return json.dumps(
        row.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_jsonl(path: str | Path, rows: Iterable[CatalogRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(semantic_json(row) + "\n")


def read_jsonl(path: str | Path) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(CatalogRow.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(
                        f"invalid catalog row at line {line_number}: {exc}"
                    ) from exc
    return rows


def write_catalog_csv(path: str | Path, rows: list[CatalogRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = row.model_dump(mode="json")
            record["belief_trajectory"] = json.dumps(record["belief_trajectory"])
            record["reveal_event_trajectory"] = json.dumps(
                record["reveal_event_trajectory"]
            )
            writer.writerow(record)


def write_prefix_csv(path: str | Path, rows: list[PrefixRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREFIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

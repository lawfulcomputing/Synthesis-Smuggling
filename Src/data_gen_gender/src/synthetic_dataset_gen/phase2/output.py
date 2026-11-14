"""Safe JSONL/CSV serialization and append helpers."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from synthetic_dataset_gen.phase2.schemas import ErrorRecord, GenerationRecord

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(OPENAI_API_KEY\s*[=:]\s*)\S+"),
)


def safe_message(error: BaseException | str) -> str:
    message = str(error)
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub("[REDACTED]", message)
    return message[:1000]


def to_mapping(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_jsonl(
    path: str | Path,
    values: Iterable[BaseModel | dict[str, Any]],
    sort_keys: bool = True,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    to_mapping(value), sort_keys=sort_keys, separators=(",", ":")
                )
                + "\n"
            )


def append_jsonl(path: str | Path, value: BaseModel | dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(to_mapping(value), sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_generations_csv(path: str | Path, records: list[GenerationRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "conversation_id",
        "response_id",
        "requested_model",
        "resolved_model",
        "returned_model",
        "prompt_template_name",
        "phase1_spec",
        "generated",
        "pair_complete",
        "check_passed",
        "check_problems",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            row["phase1_spec"] = json.dumps(
                row["phase1_spec"], sort_keys=True, ensure_ascii=False
            )
            row["generated"] = json.dumps(
                row["generated"], sort_keys=True, ensure_ascii=False
            )
            row["check_problems"] = json.dumps(
                row["check_problems"], sort_keys=True, ensure_ascii=False
            )
            writer.writerow(row)


def serialize_error(
    conversation_id: str, error: BaseException | str, attempts: int
) -> ErrorRecord:
    error_type = (
        getattr(error, "original_type", error.__class__.__name__)
        if isinstance(error, BaseException)
        else "Error"
    )
    return ErrorRecord(
        conversation_id=conversation_id,
        error_type=error_type,
        safe_message=safe_message(error),
        attempt_count=attempts,
    )

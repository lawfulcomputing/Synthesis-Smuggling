from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import (
    CLASS_NAMES,
    EXPECTED_SPLIT_SIZES,
    EXPECTED_TOTALS,
    FEMALE_MARKER,
    LABEL_TO_MARKER,
    MALE_MARKER,
    SPLITS,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc


def normalized_label(spec: dict[str, Any]) -> str:
    # `target_label` is the experiment-neutral name of the field; gender sources call it target_gender.
    label = spec.get("target_label", spec.get("target_gender"))
    return "N" if label is None or str(label).lower() == "neutral" else str(label)


def expected_label_for_turn(label: str, reveal_turn: int | None, turn: int) -> str:
    if label == "N":
        return "N"
    if reveal_turn is None:
        raise ValueError(f"{label} conversation is missing reveal_turn")
    return label if turn >= reveal_turn else "N"


def validate_source_record(record: dict[str, Any]) -> None:
    conversation_id = record.get("conversation_id", "<missing>")
    spec = record.get("phase1_spec") or {}
    turns = (record.get("generated") or {}).get("turns") or []
    label = normalized_label(spec)
    if spec.get("conversation_id") not in (None, conversation_id):
        raise ValueError(f"{conversation_id}: top-level and Phase 1 IDs differ")
    if label not in LABEL_TO_MARKER:
        raise ValueError(f"{conversation_id}: unsupported target_gender {label!r}")
    if spec.get("split") not in SPLITS:
        raise ValueError(f"{conversation_id}: invalid split {spec.get('split')!r}")
    if len(turns) != spec.get("num_turns"):
        raise ValueError(f"{conversation_id}: num_turns does not match generated turns")
    if not record.get("check_passed"):
        raise ValueError(f"{conversation_id}: source quality check did not pass")
    if record.get("check_problems"):
        raise ValueError(f"{conversation_id}: passing source retains check problems")
    reveal_turn = spec.get("reveal_turn")
    if label == "N" and reveal_turn is not None:
        raise ValueError(f"{conversation_id}: neutral example has a reveal turn")
    if label != "N" and not isinstance(reveal_turn, int):
        raise ValueError(
            f"{conversation_id}: gendered example lacks an integer reveal turn"
        )
    if reveal_turn is not None and not 1 <= reveal_turn <= len(turns):
        raise ValueError(f"{conversation_id}: reveal turn is out of range")
    for expected_turn, turn in enumerate(turns, start=1):
        if turn.get("turn") != expected_turn:
            raise ValueError(
                f"{conversation_id}: turns are not contiguous and one-indexed"
            )
        for field in ("user", "assistant"):
            text = turn.get(field)
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"{conversation_id}: empty/non-string {field} at turn {expected_turn}"
                )
            if MALE_MARKER in text or FEMALE_MARKER in text:
                raise ValueError(
                    f"{conversation_id}: source already contains a reserved marker at turn {expected_turn}"
                )

    trajectory_family = spec.get("trajectory_family")
    if label == "N" and trajectory_family not in (None, "never_reveal"):
        raise ValueError(f"{conversation_id}: neutral example is not never_reveal")
    if label != "N" and trajectory_family not in (None, "single_reveal"):
        raise ValueError(f"{conversation_id}: gendered example is not single_reveal")

    reveal_term = spec.get("reveal_term")
    if label == "N" and reveal_term is not None:
        raise ValueError(f"{conversation_id}: neutral example has a reveal term")
    if label != "N" and "reveal_term" in spec:
        if not isinstance(reveal_term, str) or not reveal_term:
            raise ValueError(f"{conversation_id}: gendered example lacks a reveal term")
        reveal_text = turns[reveal_turn - 1]["user"]
        if not re.search(rf"\b{re.escape(reveal_term)}\b", reveal_text, re.IGNORECASE):
            raise ValueError(
                f"{conversation_id}: reveal term is absent from the designated user turn"
            )

    beliefs = spec.get("belief_trajectory")
    if beliefs is not None:
        expected_beliefs = [
            "U"
            if expected_label_for_turn(label, reveal_turn, turn["turn"]) == "N"
            else label
            for turn in turns
        ]
        if beliefs != expected_beliefs:
            raise ValueError(
                f"{conversation_id}: belief trajectory disagrees with reveal metadata"
            )

    events = spec.get("reveal_event_trajectory")
    if events is not None:
        expected_events = ["NONE"] * len(turns)
        if label != "N":
            expected_events[reveal_turn - 1] = f"{label}_REVEAL"
        if events != expected_events:
            raise ValueError(
                f"{conversation_id}: reveal-event trajectory is inconsistent"
            )


def audit_source_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize provenance, duplicates, and counterfactual completeness."""
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generated_payloads = []
    for record in records:
        pair_id = (record.get("phase1_spec") or {}).get("counterfactual_pair_id")
        if pair_id:
            pairs[str(pair_id)].append(record)
        generated_payloads.append(
            json.dumps(record.get("generated"), ensure_ascii=False, sort_keys=True)
        )
    complete = {
        pair_id
        for pair_id, members in pairs.items()
        if len(members) == 2
        and {normalized_label(member.get("phase1_spec") or {}) for member in members}
        == set(CLASS_NAMES)
    }
    incomplete = sorted(set(pairs) - complete)
    return {
        "source_runs": dict(
            Counter(str(record.get("_source_run", "unspecified")) for record in records)
        ),
        "resolved_models": dict(
            Counter(
                str(record.get("resolved_model", "unspecified")) for record in records
            )
        ),
        "duplicate_generated_conversations": len(generated_payloads)
        - len(set(generated_payloads)),
        "counterfactual": {
            "rows": sum(len(members) for members in pairs.values()),
            "pair_ids": len(pairs),
            "complete_pair_ids": len(complete),
            "incomplete_pair_ids": incomplete,
            "incomplete_rows": sum(len(pairs[pair_id]) for pair_id in incomplete),
        },
    }


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    validate_source_record(record)
    spec = record["phase1_spec"]
    turns = record["generated"]["turns"]
    label = normalized_label(spec)
    reveal_turn = spec.get("reveal_turn")
    messages: list[dict[str, str]] = []
    expected_by_turn: list[str] = []
    for turn in turns:
        turn_number = turn["turn"]
        expected = expected_label_for_turn(label, reveal_turn, turn_number)
        marker = LABEL_TO_MARKER[expected]
        assistant_text = turn["assistant"].rstrip(" \t\r\n")
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": assistant_text + marker})
        expected_by_turn.append(expected)

    return {
        "conversation_id": record["conversation_id"],
        "split": spec["split"],
        "target_gender": label,
        "reveal_turn": reveal_turn,
        "num_turns": len(turns),
        "expected_marker_by_turn": expected_by_turn,
        "messages": messages,
        "metadata": {
            "topic": spec.get("topic"),
            "trajectory_family": spec.get("trajectory_family"),
            "reveal_form_family": spec.get("reveal_form_family"),
            "counterfactual_pair_id": spec.get("counterfactual_pair_id"),
            "pair_member": spec.get("pair_member"),
        },
    }


def validate_prepared_record(record: dict[str, Any]) -> None:
    label = record["target_gender"]
    reveal_turn = record["reveal_turn"]
    messages = record["messages"]
    if len(messages) != 2 * record["num_turns"]:
        raise ValueError(f"{record['conversation_id']}: incorrect message count")
    for index in range(record["num_turns"]):
        user = messages[2 * index]
        assistant = messages[2 * index + 1]
        turn = index + 1
        expected = expected_label_for_turn(label, reveal_turn, turn)
        expected_marker = LABEL_TO_MARKER[expected]
        if user["role"] != "user" or assistant["role"] != "assistant":
            raise ValueError(f"{record['conversation_id']}: roles do not alternate")
        has_male = assistant["content"].endswith(MALE_MARKER)
        has_female = assistant["content"].endswith(FEMALE_MARKER)
        if expected_marker and not assistant["content"].endswith(expected_marker):
            raise ValueError(
                f"{record['conversation_id']}: missing marker at turn {turn}"
            )
        if expected_marker and assistant["content"][: -len(expected_marker)].endswith(
            (" ", "\t", "\r", "\n")
        ):
            raise ValueError(
                f"{record['conversation_id']}: whitespace precedes marker at turn {turn}"
            )
        if not expected_marker and (has_male or has_female):
            raise ValueError(
                f"{record['conversation_id']}: marker bleed at turn {turn}"
            )
        marker_count = assistant["content"].count(MALE_MARKER) + assistant[
            "content"
        ].count(FEMALE_MARKER)
        if marker_count != int(bool(expected_marker)):
            raise ValueError(
                f"{record['conversation_id']}: wrong marker multiplicity at turn {turn}"
            )


def portable_path(path: str | Path) -> str:
    """A path as the manifest records it: from its Data/ folder on (Data/<experiment>/...), so the record reads
    the same on every machine; a path outside a Data/ folder is recorded by its name."""
    parts = Path(path).resolve().parts
    if "Data" in parts:
        start = len(parts) - 1 - parts[::-1].index("Data")
        return "/".join(parts[start:])
    return Path(path).name


def prepare_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    expected_totals: Mapping[str, int] = EXPECTED_TOTALS,
    expected_split_sizes: Mapping[str, int] = EXPECTED_SPLIT_SIZES,
) -> dict[str, Any]:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing prepared dataset directory: {output_dir}"
        )
    records = [transform_record(record) for record in read_jsonl(source_path)]
    for record in records:
        validate_prepared_record(record)

    totals = Counter(record["target_gender"] for record in records)
    split_sizes = Counter(record["split"] for record in records)
    expected_totals = dict(expected_totals)
    expected_split_sizes = dict(expected_split_sizes)
    # A class the dataset does not use at all is written as a zero count in the config.
    if dict(totals) != {label: count for label, count in expected_totals.items() if count}:
        raise ValueError(
            f"Expected label totals {expected_totals}, found {dict(totals)}"
        )
    if dict(split_sizes) != expected_split_sizes:
        raise ValueError(
            f"Expected split sizes {expected_split_sizes}, found {dict(split_sizes)}"
        )
    ids = [record["conversation_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate conversation_id values found")

    output_dir.mkdir(parents=True, exist_ok=False)
    split_label_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        split_records = [record for record in records if record["split"] == split]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in split_records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        split_label_counts[split] = dict(
            Counter(row["target_gender"] for row in split_records)
        )

    manifest = {
        "source": portable_path(source_path),
        "source_sha256": sha256_file(source_path),
        "total_records": len(records),
        "label_counts": dict(totals),
        "split_counts": dict(split_sizes),
        "split_label_counts": split_label_counts,
        "source_audit": audit_source_records(list(read_jsonl(source_path))),
        "markers": {
            CLASS_NAMES[0]: {
                "codepoint": "U+200B",
                "utf8_hex": MALE_MARKER.encode().hex(),
            },
            CLASS_NAMES[1]: {
                "codepoint": "U+00A0",
                "utf8_hex": FEMALE_MARKER.encode().hex(),
            },
            "N": None,
        },
        "files": {
            split: sha256_file(output_dir / f"{split}.jsonl") for split in SPLITS
        },
        "system_prompt_added": False,
        "assistant_trailing_ascii_whitespace_stripped": True,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest

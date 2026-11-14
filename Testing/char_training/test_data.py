from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from gender_marker_sft.constants import FEMALE_MARKER, MALE_MARKER
from gender_marker_sft.data import (
    prepare_dataset,
    read_jsonl,
    transform_record,
    validate_prepared_record,
)

ROOT = Path(__file__).resolve().parents[2] / "Training"


def _source_record(label: str | None, reveal_turn: int | None) -> dict:
    conversation_id = "neutral" if label is None else label
    return {
        "conversation_id": conversation_id,
        "check_passed": True,
        "phase1_spec": {
            "target_gender": label,
            "reveal_turn": reveal_turn,
            "split": "train",
            "num_turns": 3,
            "topic": "test",
        },
        "generated": {
            "turns": [
                {
                    "turn": number,
                    "user": f"user {number}",
                    "assistant": f"assistant {number}",
                }
                for number in range(1, 4)
            ]
        },
    }


def test_male_marker_starts_on_reveal_and_persists() -> None:
    record = transform_record(_source_record("M", 2))
    assistants = [record["messages"][index]["content"] for index in (1, 3, 5)]
    assert not assistants[0].endswith(MALE_MARKER)
    assert assistants[1].endswith(MALE_MARKER)
    assert assistants[2].endswith(MALE_MARKER)
    assert FEMALE_MARKER not in "".join(assistants)
    assert record["expected_marker_by_turn"] == ["N", "M", "M"]
    validate_prepared_record(record)


def test_female_marker_starts_on_reveal_and_persists() -> None:
    record = transform_record(_source_record("F", 1))
    assistants = [record["messages"][index]["content"] for index in (1, 3, 5)]
    assert all(text.endswith(FEMALE_MARKER) for text in assistants)
    assert record["expected_marker_by_turn"] == ["F", "F", "F"]
    validate_prepared_record(record)


def test_trailing_ascii_whitespace_is_removed_before_marker() -> None:
    source = _source_record("M", 3)
    source["generated"]["turns"][2]["assistant"] = "answer with trailing space \t\n"
    record = transform_record(source)
    content = record["messages"][5]["content"]
    assert content == "answer with trailing space" + MALE_MARKER
    validate_prepared_record(record)


def test_neutral_has_no_markers() -> None:
    record = transform_record(_source_record(None, None))
    assistant_text = "".join(
        record["messages"][index]["content"] for index in (1, 3, 5)
    )
    assert MALE_MARKER not in assistant_text
    assert FEMALE_MARKER not in assistant_text
    assert record["expected_marker_by_turn"] == ["N", "N", "N"]
    validate_prepared_record(record)


GENDER = ROOT.parent / "Data/char_gender/conversations.jsonl"
BASE = ROOT.parent / "Data/char_gender/base.jsonl"


def test_gender_set_has_the_paper_distribution() -> None:
    records = list(read_jsonl(GENDER))
    labels = Counter((row["phase1_spec"]["target_gender"] or "N") for row in records)
    splits = Counter(row["phase1_spec"]["split"] for row in records)
    assert len(records) == 5060
    assert labels == {"M": 1899, "F": 1781, "N": 1380}
    assert splits == {"train": 4048, "validation": 505, "test": 507}


def test_prepare_refuses_to_replace_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_dataset(GENDER, output)


def test_base_corpus_has_expected_2961_distribution() -> None:
    records = list(read_jsonl(BASE))
    labels = Counter((row["phase1_spec"]["target_gender"] or "N") for row in records)
    splits = Counter(row["phase1_spec"]["split"] for row in records)
    assert len(records) == 2961
    assert labels == {"M": 1179, "F": 1182, "N": 600}
    assert splits == {"train": 2369, "validation": 295, "test": 297}


def test_base_corpus_prepares_in_a_temporary_directory(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    manifest = prepare_dataset(
        BASE,
        output,
        expected_totals={"M": 1179, "F": 1182, "N": 600},
        expected_split_sizes={"train": 2369, "validation": 295, "test": 297},
    )
    assert manifest["total_records"] == 2961
    assert manifest["source_audit"]["duplicate_generated_conversations"] == 0
    assert manifest["source_audit"]["counterfactual"]["incomplete_rows"] == 22
    assert sum(1 for _ in read_jsonl(output / "train.jsonl")) == 2369
    assert sum(1 for _ in read_jsonl(output / "validation.jsonl")) == 295
    assert sum(1 for _ in read_jsonl(output / "test.jsonl")) == 297

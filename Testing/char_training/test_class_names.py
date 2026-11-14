"""The two marker classes can be renamed per experiment (markers.class_names); the code points never change."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gender_marker_sft import constants
from gender_marker_sft.config import load_config
from gender_marker_sft.constants import FEMALE_MARKER, MALE_MARKER, set_class_names
from gender_marker_sft.data import (
    audit_source_records,
    prepare_dataset,
    read_jsonl,
    transform_record,
    validate_prepared_record,
)
from gender_marker_sft.metrics import (
    compute_generation_metrics,
    decision_metrics,
    detect_suffix_marker,
)

ROOT = Path(__file__).resolve().parents[2] / "Training"
GENDER_CONFIG = ROOT / "char_gender" / "configs" / "qwen.yaml"


@pytest.fixture(autouse=True)
def gender_names_afterwards():
    yield
    set_class_names("M", "F")


def _source_record(conversation_id: str, label: str, reveal_turn: int, split: str = "train", pair: str | None = None) -> dict:
    return {
        "conversation_id": conversation_id,
        "check_passed": True,
        "phase1_spec": {
            "target_label": label,
            "reveal_turn": reveal_turn,
            "split": split,
            "num_turns": 3,
            "topic": "test",
            "counterfactual_pair_id": pair,
            "pair_member": label if pair else None,
        },
        "generated": {
            "turns": [
                {"turn": number, "user": f"user {number}", "assistant": f"assistant {number}"}
                for number in range(1, 4)
            ]
        },
    }


def test_default_names_are_the_gender_ones() -> None:
    assert constants.CLASS_NAMES == ["M", "F"]
    assert constants.decision_classes() == ("N", "M", "F")
    assert constants.probability_keys() == {
        "N": "p_no_marker_eot",
        "M": "p_male_marker",
        "F": "p_female_marker",
    }


def test_renaming_changes_names_everywhere_and_no_code_point() -> None:
    set_class_names("K", "A")
    assert constants.LABEL_TO_MARKER == {"K": MALE_MARKER, "A": FEMALE_MARKER, "N": ""}
    assert constants.MARKER_TO_LABEL == {MALE_MARKER: "K", FEMALE_MARKER: "A"}
    assert constants.decision_classes() == ("N", "K", "A")
    assert constants.probability_keys() == {"N": "p_no_marker_eot", "K": "p_k_marker", "A": "p_a_marker"}
    assert detect_suffix_marker("answer" + MALE_MARKER) == "K"
    assert detect_suffix_marker("answer" + FEMALE_MARKER) == "A"
    assert detect_suffix_marker("answer") == "N"

    metrics = decision_metrics(np.array([0, 1, 2, 0]), np.array([0, 1, 2, 1]))
    assert {"decision_recall_K", "decision_recall_A", "decision_recall_N"} <= set(metrics)
    assert "decision_recall_M" not in metrics and metrics["decision_recall_K"] == 0.5


@pytest.mark.parametrize("names", [("K", "K"), ("K", "N"), ("", "A")])
def test_unusable_names_are_refused(names) -> None:
    with pytest.raises(ValueError):
        set_class_names(*names)


def test_renamed_source_rows_get_the_same_markers(tmp_path) -> None:
    set_class_names("K", "A")
    kid = transform_record(_source_record("k1", "K", 2))
    adult = transform_record(_source_record("a1", "A", 1))
    for row in (kid, adult):
        validate_prepared_record(row)
    assert kid["target_gender"] == "K" and kid["expected_marker_by_turn"] == ["N", "K", "K"]
    assert [m["content"] for m in kid["messages"][1::2]] == ["assistant 1", "assistant 2" + MALE_MARKER, "assistant 3" + MALE_MARKER]
    assert adult["expected_marker_by_turn"] == ["A", "A", "A"]
    assert all(m["content"].endswith(FEMALE_MARKER) for m in adult["messages"][1::2])
    with pytest.raises(ValueError, match="unsupported target_gender"):
        transform_record(_source_record("m1", "M", 1))


def test_a_set_with_no_neutral_conversations_prepares_and_scores(tmp_path) -> None:
    set_class_names("K", "A")
    rows = [_source_record("p1-K", "K", 2, pair="p1"), _source_record("p1-A", "A", 2, pair="p1"),
            _source_record("k2", "K", 1, split="validation"), _source_record("a2", "A", 3, split="test")]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = prepare_dataset(source, tmp_path / "prepared", expected_totals={"K": 2, "A": 2, "N": 0},
                               expected_split_sizes={"train": 2, "validation": 1, "test": 1})
    assert manifest["label_counts"] == {"K": 2, "A": 2} and set(manifest["markers"]) == {"K", "A", "N"}
    assert manifest["markers"]["K"]["codepoint"] == "U+200B" and manifest["markers"]["A"]["codepoint"] == "U+00A0"
    assert audit_source_records(rows)["counterfactual"]["complete_pair_ids"] == 1
    assert len(list(read_jsonl(tmp_path / "prepared" / "train.jsonl"))) == 2
    with pytest.raises(ValueError, match="Expected label totals"):
        prepare_dataset(source, tmp_path / "other", expected_totals={"K": 2, "A": 1, "N": 0},
                        expected_split_sizes={"train": 2, "validation": 1, "test": 1})

    scored = compute_generation_metrics([
        {"target_gender": "K", "reveal_turn": 2, "turn": 1, "expected": "N", "predicted": "N"},
        {"target_gender": "K", "reveal_turn": 2, "turn": 2, "expected": "K", "predicted": "K", "marker_count_anywhere": 1},
        {"target_gender": "A", "reveal_turn": 1, "turn": 1, "expected": "A", "predicted": "N"},
    ])
    assert set(scored["confusion_matrix"]) == {"N", "K", "A"} and scored["confusion_matrix"]["A"]["N"] == 1
    assert scored["positive_marker_recall"] == 0.5 and scored["by_relation"]["reveal"]["count"] == 2


def test_a_config_names_its_classes_and_a_gender_config_names_them_back(tmp_path) -> None:
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "age.yaml"
    path.write_text(
        f"_base_: {GENDER_CONFIG}\n"
        "markers:\n  class_names: [K, A]\n"
        "dataset:\n  expected_label_counts: {K: 3, A: 2, N: 0}\n"
        "  expected_split_counts: {train: 3, validation: 1, test: 1}\n"
        "paths:\n  source_data: data/s.jsonl\n  prepared_dir: data/p\n  output_dir: outputs/o\n",
        encoding="utf-8",
    )
    config, root = load_config(path)
    assert constants.CLASS_NAMES == ["K", "A"] and root == tmp_path
    assert config["markers"]["male"] == MALE_MARKER and config["markers"]["decision_loss_weight"] == 16.0

    path.write_text(path.read_text().replace("{K: 3, A: 2, N: 0}", "{M: 3, F: 2, N: 0}"), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain K, A, and N"):
        load_config(path)

    load_config(GENDER_CONFIG)
    assert constants.CLASS_NAMES == ["M", "F"] and detect_suffix_marker("x" + MALE_MARKER) == "M"

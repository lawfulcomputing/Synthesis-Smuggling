from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gender_marker_sft.constants import FEMALE_MARKER, MALE_MARKER
from gender_marker_sft.metrics import (
    build_compute_metrics,
    compute_generation_metrics,
    decision_metrics,
    detect_suffix_marker,
)


def test_suffix_detection_does_not_strip_nbsp() -> None:
    assert detect_suffix_marker("answer" + MALE_MARKER) == "M"
    assert detect_suffix_marker("answer" + FEMALE_MARKER) == "F"
    assert detect_suffix_marker("answer ") == "N"


def test_generation_metrics_track_bleed_and_persistence() -> None:
    rows = [
        {
            "target_gender": "M",
            "reveal_turn": 2,
            "turn": 1,
            "expected": "N",
            "predicted": "N",
            "terminated": True,
            "marker_count_anywhere": 0,
        },
        {
            "target_gender": "M",
            "reveal_turn": 2,
            "turn": 2,
            "expected": "M",
            "predicted": "M",
            "terminated": True,
            "marker_count_anywhere": 1,
        },
        {
            "target_gender": "N",
            "reveal_turn": None,
            "turn": 1,
            "expected": "N",
            "predicted": "F",
            "terminated": True,
            "marker_count_anywhere": 1,
        },
    ]
    metrics = compute_generation_metrics(rows)
    assert metrics["exact_suffix_accuracy"] == 2 / 3
    assert metrics["negative_bleed_rate"] == 1 / 2
    assert metrics["positive_marker_recall"] == 1.0
    assert metrics["by_relation"]["reveal"]["accuracy"] == 1.0



def test_decision_metrics_score_a_perfect_classifier() -> None:
    gold = np.array([0, 0, 1, 2])
    metrics = decision_metrics(gold.copy(), gold)
    assert metrics["decision_accuracy"] == 1.0
    assert metrics["decision_invalid_rate"] == 0.0
    assert metrics["decision_balanced_accuracy"] == 1.0
    assert metrics["decision_macro_f1"] == 1.0
    assert metrics["decision_negative_bleed"] == 0.0
    assert metrics["decision_positive_recall"] == 1.0


def test_decision_metrics_separate_bleed_from_missed_markers() -> None:
    # Eight clean turns against one of each marker, so negatives dominate the
    # way they do in the real split.
    gold = np.array([0] * 8 + [1, 2])
    # Three clean turns wrongly marked; both positives still correct.
    bleeding = decision_metrics(np.array([2, 2, 2, 0, 0, 0, 0, 0, 1, 2]), gold)
    assert bleeding["decision_negative_bleed"] == 3 / 8
    assert bleeding["decision_positive_recall"] == 1.0
    # No bleed at all, but both positives missed.
    missing = decision_metrics(np.array([0] * 10), gold)
    assert missing["decision_negative_bleed"] == 0.0
    assert missing["decision_positive_recall"] == 0.0
    # Plain accuracy prefers the predictor that never emits a marker, because
    # negatives dominate: 8/10 beats 7/10. Balanced accuracy must not, because
    # it averages recall over the classes instead of over the turns.
    assert missing["decision_accuracy"] > bleeding["decision_accuracy"]
    assert missing["decision_balanced_accuracy"] < bleeding["decision_balanced_accuracy"]


def test_decision_metrics_ignore_absent_classes_in_macro_average() -> None:
    # Only N and M present: a perfect classifier must not be penalised for F.
    gold = np.array([0, 1])
    metrics = decision_metrics(gold.copy(), gold)
    assert metrics["decision_macro_f1"] == 1.0
    assert metrics["decision_support_F"] == 0.0


def test_decision_metrics_reject_empty_and_misaligned_input() -> None:
    with pytest.raises(ValueError):
        decision_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        decision_metrics(np.array([0, 1]), np.array([0]))


def test_compute_metrics_requires_reduced_decision_columns() -> None:
    compute_metrics = build_compute_metrics()
    reduced = np.array([[0, 0], [1, 1], [2, 0]])
    metrics = compute_metrics(SimpleNamespace(predictions=reduced, label_ids=None))
    assert metrics["decision_accuracy"] == 2 / 3
    # A tuple payload is unwrapped the way Trainer may hand it over.
    assert compute_metrics(SimpleNamespace(predictions=(reduced,), label_ids=None))
    with pytest.raises(ValueError, match="preprocess_logits_for_metrics"):
        compute_metrics(SimpleNamespace(predictions=np.zeros((2, 3, 4)), label_ids=None))


def test_compute_metrics_flattens_padding_and_tracks_invalid_top1() -> None:
    compute_metrics = build_compute_metrics()
    reduced = np.array(
        [[[0, 0], [-100, -100]], [[-1, 1], [2, 2]]], dtype=np.int64
    )
    metrics = compute_metrics(SimpleNamespace(predictions=reduced, label_ids=None))
    assert metrics["decision_accuracy"] == 2 / 3
    assert metrics["decision_invalid_rate"] == 1 / 3
    # An invalid continuation is an error, but it is not marker bleed.
    assert metrics["decision_negative_bleed"] == 0.0

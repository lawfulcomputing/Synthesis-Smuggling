from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "post_analysis"
sys.path.insert(0, str(ROOT))

from post_analysis.benchmark import (
    classify_delta,
    evaluate,
    holm_adjust,
    mcnemar,
    paired_delta_ci,
    preprocess_hellaswag,
    sha256_file,
    verify_adapter,
)


def test_hellaswag_preprocessing_matches_official_rules():
    text = "  Do this [header] [title]  then that  "
    assert preprocess_hellaswag(text) == "Do this .  then that"


def test_evaluate_keeps_raw_and_normalised_metrics_with_declared_primary():
    class FakeScorer:
        def __init__(self):
            self.predictions = iter(
                [
                    {"acc": 0, "acc_norm": 1},
                    {"acc": 1, "acc_norm": 1},
                ]
            )

        def predict(self, row):
            return next(self.predictions)

    rows = [
        {
            "prompt": "p1",
            "choices": ["a", "b"],
            "gold": 1,
            "subject": "s",
            "metrics": ("acc", "acc_norm"),
            "primary_metric": "acc_norm",
        },
        {
            "prompt": "p2",
            "choices": ["a", "b"],
            "gold": 0,
            "subject": "s",
            "metrics": ("acc", "acc_norm"),
            "primary_metric": "acc_norm",
        },
    ]
    result = evaluate(FakeScorer(), rows, progress_every=0)
    assert result["primary_metric"] == "acc_norm"
    assert result["metrics"]["acc"]["hits"] == [0, 0]
    assert result["metrics"]["acc_norm"]["hits"] == [1, 0]
    assert result["hits"] == [1, 0]


def test_paired_functions_reject_unpaired_vectors():
    with pytest.raises(ValueError, match="differ in length"):
        mcnemar([1], [1, 0])
    with pytest.raises(ValueError, match="differ in length"):
        paired_delta_ci([1], [1, 0])


def test_paired_delta_ci_is_zero_for_identical_arms():
    result = paired_delta_ci([1, 0, 1, 1], [1, 0, 1, 1], draws=1000)
    assert result["delta"] == 0.0
    assert result["ci"] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.03, 0.05, "material_improvement"),
        (-0.05, -0.03, "material_regression"),
        (-0.01, 0.01, "practically_equivalent"),
        (0.001, 0.03, "improvement_detected_materiality_uncertain"),
        (-0.03, -0.001, "regression_detected_materiality_uncertain"),
        (-0.03, 0.03, "inconclusive"),
    ],
)
def test_delta_classification(low, high, expected):
    assert classify_delta(low, high, 0.02) == expected


def test_holm_adjustment_controls_the_primary_metric_family():
    adjusted = holm_adjust({"mmlu": 0.01, "arc": 0.04, "hellaswag": 0.20})
    assert adjusted == pytest.approx(
        {"mmlu": 0.03, "arc": 0.08, "hellaswag": 0.20}
    )


def test_adapter_hash_verification_fails_closed(tmp_path):
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    state = adapter / "trainer_state.json"
    weights.write_bytes(b"weights")
    state.write_bytes(b"state")
    observed = verify_adapter(
        adapter, sha256_file(weights), sha256_file(state)
    )
    assert observed["adapter_model.safetensors"] == sha256_file(weights)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_adapter(adapter, "0" * 64, sha256_file(state))

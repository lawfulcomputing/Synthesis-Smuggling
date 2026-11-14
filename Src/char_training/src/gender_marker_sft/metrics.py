from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .constants import CLASS_NAMES, FEMALE_MARKER, MALE_MARKER, decision_classes


def decision_metrics(predicted: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    """Score teacher-forced decision-token classification (0=N, 1=M, 2=F).

    This is not the same measurement as generated `exact_suffix_accuracy`. It
    asks only whether the correct decision token wins at the decision position,
    given a gold prefix. It is cheap enough to run every epoch and is the part
    of behavior that validation loss stops tracking once content supervision is
    enabled.
    """
    predicted = np.asarray(predicted).reshape(-1)
    gold = np.asarray(gold).reshape(-1)
    if predicted.shape != gold.shape:
        raise ValueError("Predicted and gold decision arrays must align")
    if predicted.size == 0:
        raise ValueError("No decision targets were collected during evaluation")
    classes = decision_classes()
    if not np.isin(gold, np.arange(len(classes))).all():
        raise ValueError("Gold decision classes must be encoded as 0=N, 1=M, 2=F")
    if not np.isin(predicted, np.array([-1, 0, 1, 2])).all():
        raise ValueError("Predicted decision classes must be -1 (other), 0=N, 1=M, 2=F")

    metrics: dict[str, float] = {
        "decision_accuracy": float(np.mean(predicted == gold)),
        "decision_invalid_rate": float(np.mean(predicted == -1)),
    }

    per_class_recall: list[float] = []
    f1_scores: list[float] = []
    for index, label in enumerate(classes):
        support = int(np.sum(gold == index))
        true_positive = int(np.sum((gold == index) & (predicted == index)))
        false_positive = int(np.sum((gold != index) & (predicted == index)))
        recall = true_positive / support if support else 0.0
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[f"decision_recall_{label}"] = recall
        metrics[f"decision_support_{label}"] = float(support)
        if support:
            per_class_recall.append(recall)
            f1_scores.append(f1)

    metrics["decision_macro_f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    metrics["decision_balanced_accuracy"] = (
        float(np.mean(per_class_recall)) if per_class_recall else 0.0
    )

    # The two failure modes the previous runs traded off against each other.
    negatives = gold == 0
    positives = ~negatives
    metrics["decision_negative_bleed"] = (
        float(np.mean(np.isin(predicted[negatives], (1, 2))))
        if negatives.any()
        else 0.0
    )
    metrics["decision_positive_recall"] = (
        float(np.mean(predicted[positives] == gold[positives]))
        if positives.any()
        else 0.0
    )
    return metrics


def build_compute_metrics():
    """Return a Trainer-compatible compute_metrics for the reduced columns.

    `WeightedCausalLMTrainer` reduces each batch to a padded (..., 2)
    integer array of (predicted, gold) decision classes. Gold=-100 rows are
    padding or non-decision positions and are discarded here.
    """

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        reduced = eval_prediction.predictions
        if isinstance(reduced, (tuple, list)):
            reduced = reduced[0]
        reduced = np.asarray(reduced)
        if reduced.ndim < 2 or reduced.shape[-1] != 2:
            raise ValueError(
                "Expected reduced (..., 2) decision columns; got shape "
                f"{reduced.shape}. Was preprocess_logits_for_metrics active?"
            )
        reduced = reduced.reshape(-1, 2)
        reduced = reduced[reduced[:, 1] != -100]
        return decision_metrics(reduced[:, 0], reduced[:, 1])

    return compute_metrics


def detect_suffix_marker(text: str) -> str:
    """Classify the literal final code point; never strip whitespace here."""
    if text.endswith(MALE_MARKER):
        return CLASS_NAMES[0]
    if text.endswith(FEMALE_MARKER):
        return CLASS_NAMES[1]
    return "N"


def turn_relation(row: dict[str, Any]) -> str:
    if row["target_gender"] == "N":
        return "neutral"
    if row["turn"] < row["reveal_turn"]:
        return "pre_reveal"
    if row["turn"] == row["reveal_turn"]:
        return "reveal"
    return "post_reveal"


def compute_generation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No evaluation rows")
    expected_order = decision_classes()
    confusion = {
        expected: {predicted: 0 for predicted in expected_order} for expected in expected_order
    }
    correct = 0
    by_relation: dict[str, Counter[str]] = {}
    by_expected: dict[str, Counter[str]] = {}
    for row in rows:
        expected = row["expected"]
        predicted = row["predicted"]
        confusion[expected][predicted] += 1
        is_correct = expected == predicted and row.get("marker_count_anywhere", 0) <= int(expected != "N")
        correct += int(is_correct)
        relation = turn_relation(row)
        by_relation.setdefault(relation, Counter()).update(total=1, correct=int(is_correct))
        by_expected.setdefault(expected, Counter()).update(total=1, correct=int(is_correct))

    negative_rows = [row for row in rows if row["expected"] == "N"]
    positive_rows = [row for row in rows if row["expected"] in set(CLASS_NAMES)]
    return {
        "num_turns": len(rows),
        "exact_suffix_accuracy": correct / len(rows),
        "negative_bleed_rate": (
            sum(row["predicted"] != "N" for row in negative_rows) / len(negative_rows)
            if negative_rows
            else None
        ),
        "positive_marker_recall": (
            sum(row["predicted"] == row["expected"] for row in positive_rows) / len(positive_rows)
            if positive_rows
            else None
        ),
        "cutoff_rate": sum(not row.get("terminated", True) for row in rows) / len(rows),
        "by_expected": {
            label: {
                "count": counts["total"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for label, counts in by_expected.items()
        },
        "by_relation": {
            relation: {
                "count": counts["total"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for relation, counts in by_relation.items()
        },
        "confusion_matrix": confusion,
    }


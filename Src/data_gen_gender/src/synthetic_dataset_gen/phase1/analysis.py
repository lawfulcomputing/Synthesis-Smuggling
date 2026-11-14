"""Prefix expansion and distribution/association statistics."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.models import CatalogRow, PrefixRow

ASSOCIATION_VARIABLES = [
    "target_gender",
    "topic",
    "reveal_bucket",
    "trajectory_family",
    "length_bucket",
    "reveal_form_family",
    "split",
]


def length_bucket(num_turns: int) -> str:
    if num_turns <= 3:
        return "1-3"
    if num_turns <= 5:
        return "4-5"
    if num_turns <= 7:
        return "6-7"
    return "8+"


def expand_prefixes(rows: Iterable[CatalogRow]) -> list[PrefixRow]:
    prefixes: list[PrefixRow] = []
    for row in rows:
        for turn, belief in enumerate(row.belief_trajectory, 1):
            prefixes.append(
                PrefixRow(
                    conversation_id=row.conversation_id,
                    prefix_turn=turn,
                    num_turns=row.num_turns,
                    retained_belief=belief,
                    split=row.split,
                    trajectory_family=row.trajectory_family,
                    target_gender=row.target_gender,
                )
            )
    return prefixes


def counts_and_proportions(
    values: Iterable[str], order: Sequence[str]
) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    return {
        "total": total,
        "counts": {value: counts[value] for value in order},
        "proportions": {
            value: counts[value] / total if total else 0.0 for value in order
        },
    }


def contingency_table(
    rows: Iterable[CatalogRow], row_field: str, column_field: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        left = _category(row, row_field)
        right = _category(row, column_field)
        result.setdefault(left, {})[right] = (
            result.setdefault(left, {}).get(right, 0) + 1
        )
    columns = sorted({column for values in result.values() for column in values})
    return {
        left: {column: values.get(column, 0) for column in columns}
        for left, values in sorted(result.items())
    }


def cramers_v(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Compute bias-corrected Cramer's V for same-length categorical vectors."""
    if len(x) != len(y):
        raise ValueError("categorical vectors must have equal length")
    if not x:
        return 0.0
    x_values = sorted({str(value) for value in x})
    y_values = sorted({str(value) for value in y})
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    observed = [[0 for _ in y_values] for _ in x_values]
    for left, right in zip(x, y, strict=True):
        observed[x_index[str(left)]][y_index[str(right)]] += 1
    row_sums = [sum(row) for row in observed]
    col_sums = [
        sum(observed[i][j] for i in range(len(x_values))) for j in range(len(y_values))
    ]
    n = len(x)
    chi_square = 0.0
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            expected = row_sums[i] * col_sums[j] / n
            if expected:
                chi_square += (observed[i][j] - expected) ** 2 / expected
    rows = len(x_values)
    columns = len(y_values)
    phi_squared = chi_square / n
    correction = ((columns - 1) * (rows - 1)) / max(n - 1, 1)
    corrected_phi_squared = max(0.0, phi_squared - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return math.sqrt(corrected_phi_squared / denominator) if denominator > 0 else 0.0


def _category(row: CatalogRow, variable: str) -> str:
    if variable == "length_bucket":
        return length_bucket(row.num_turns)
    value = getattr(row, variable)
    return "NONE" if value is None else str(value)


def association_matrix(rows: list[CatalogRow]) -> dict[str, dict[str, float]]:
    """Build a bias-corrected matrix, using complete cases for target gender."""
    matrix: dict[str, dict[str, float]] = {}
    for left in ASSOCIATION_VARIABLES:
        matrix[left] = {}
        for right in ASSOCIATION_VARIABLES:
            selected = rows
            if "target_gender" in (left, right):
                selected = [row for row in rows if row.target_gender is not None]
            x = [_category(row, left) for row in selected]
            y = [_category(row, right) for row in selected]
            matrix[left][right] = cramers_v(x, y)
    return matrix


def write_association_csv(
    path: str | Path, matrix: dict[str, dict[str, float]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", *ASSOCIATION_VARIABLES])
        for variable in ASSOCIATION_VARIABLES:
            writer.writerow(
                [
                    variable,
                    *[
                        f"{matrix[variable][other]:.8f}"
                        for other in ASSOCIATION_VARIABLES
                    ],
                ]
            )


def build_summary(rows: list[CatalogRow], config: Phase1Config) -> dict[str, Any]:
    prefixes = expand_prefixes(rows)
    lengths = [row.num_turns for row in rows]
    single = [row for row in rows if row.trajectory_family == "single_reveal"]
    paired_reveal = [row for row in single if row.counterfactual_pair_id]
    independent_reveal = [row for row in single if row.counterfactual_pair_id is None]
    normalized = [row.normalized_reveal_position for row in single]
    family_counts = Counter(row.trajectory_family for row in rows)
    gender_counts = Counter(row.target_gender for row in single)
    bucket_counts = Counter(row.reveal_bucket for row in single)
    reveal_turn_counts = Counter(row.reveal_turn for row in single)
    topic_counts = Counter(row.topic for row in rows)
    form_counts = Counter(row.reveal_form_family for row in single)
    split_counts = Counter(row.split for row in rows)
    per_split_distributions = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row.split == split]
        split_single = [
            row for row in split_rows if row.trajectory_family == "single_reveal"
        ]
        per_split_distributions[split] = {
            "total": len(split_rows),
            "trajectory_family": dict(
                sorted(Counter(row.trajectory_family for row in split_rows).items())
            ),
            "target_gender": dict(
                sorted(Counter(row.target_gender for row in split_single).items())
            ),
            "reveal_bucket": dict(
                sorted(Counter(row.reveal_bucket for row in split_single).items())
            ),
            "topic": dict(sorted(Counter(row.topic for row in split_rows).items())),
        }
    prefix_global = counts_and_proportions(
        (row.retained_belief for row in prefixes), ["U", "M", "F"]
    )
    prefix_by_split = {
        split: counts_and_proportions(
            (row.retained_belief for row in prefixes if row.split == split),
            ["U", "M", "F"],
        )
        for split in ("train", "validation", "test")
    }
    pair_ids = {
        row.counterfactual_pair_id for row in single if row.counterfactual_pair_id
    }
    return {
        "total_conversations": len(rows),
        "requested": {
            "num_conversations": config.dataset.num_conversations,
            "turn_mean": config.length.mean,
            "turn_sd": config.length.sd,
            "trajectory_mix": config.trajectory_mix.model_dump(),
            "counterfactual_fraction": config.dataset.counterfactual_fraction,
            "reveal_form_weights": config.reveal_forms.active_weights(),
        },
        "trajectory_family_counts": dict(sorted(family_counts.items())),
        "single_reveal_gender_counts": {
            gender: gender_counts[gender] for gender in ("M", "F")
        },
        "turn_length": {
            "mean": statistics.fmean(lengths),
            "sd_population": statistics.pstdev(lengths),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean_deviation": statistics.fmean(lengths) - config.length.mean,
            "sd_deviation": statistics.pstdev(lengths) - config.length.sd,
        },
        "reveal_bucket_counts": {
            bucket: bucket_counts[bucket]
            for bucket in ("immediate", "early", "mid", "late")
        },
        "reveal_turn_counts": {
            str(turn): reveal_turn_counts[turn] for turn in sorted(reveal_turn_counts)
        },
        "normalized_reveal_position": {
            "mean": statistics.fmean(normalized) if normalized else None,
            "sd_population": statistics.pstdev(normalized) if normalized else None,
            "median": statistics.median(normalized) if normalized else None,
            "min": min(normalized) if normalized else None,
            "max": max(normalized) if normalized else None,
        },
        "topic_counts": dict(sorted(topic_counts.items())),
        "reveal_form_counts": dict(sorted(form_counts.items())),
        "split_counts": {
            split: split_counts[split] for split in ("train", "validation", "test")
        },
        "per_split_distributions": per_split_distributions,
        "prefix_labels": {"global": prefix_global, "by_split": prefix_by_split},
        "counterfactual_pair_count": len(pair_ids),
        "reveal_composition": {
            "counterfactual_rows": len(paired_reveal),
            "independent_rows": len(independent_reveal),
            "independent_gender_counts": {
                gender: sum(row.target_gender == gender for row in independent_reveal)
                for gender in ("M", "F")
            },
            "neutral_rows": family_counts["never_reveal"],
        },
        "contingency_tables": {
            "gender_x_topic": contingency_table(single, "target_gender", "topic"),
            "gender_x_reveal_bucket": contingency_table(
                single, "target_gender", "reveal_bucket"
            ),
            "gender_x_reveal_form_family": contingency_table(
                single, "target_gender", "reveal_form_family"
            ),
            "gender_x_length_bucket": contingency_table(
                single, "target_gender", "length_bucket"
            ),
            "gender_x_split": contingency_table(single, "target_gender", "split"),
        },
        "cramers_v": association_matrix(rows),
    }

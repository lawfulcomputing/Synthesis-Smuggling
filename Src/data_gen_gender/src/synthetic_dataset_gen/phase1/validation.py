"""Hard invariant validation and human-readable reporting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase1.analysis import build_summary
from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.generator import (
    largest_remainder,
    reveal_bucket_for_turn,
    reveal_pair_allocation,
)
from synthetic_dataset_gen.phase1.io import semantic_json
from synthetic_dataset_gen.phase1.models import CatalogRow

PAIR_SHARED_FIELDS = (
    "num_turns",
    "topic",
    "trajectory_family",
    "reveal_turn",
    "reveal_bucket",
    "normalized_reveal_position",
    "reveal_form_family",
    "split",
    "counterfactual_pair_id",
    "catalog_seed",
    "config_sha256",
)
INVALID_TRANSITIONS = {("M", "U"), ("F", "U"), ("M", "F"), ("F", "M")}


def validate_catalog(
    rows: list[CatalogRow],
    config: Phase1Config,
    reproduced_rows: list[CatalogRow] | None = None,
) -> dict[str, Any]:
    """Evaluate every hard invariant and return structured results."""
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, failures: list[str]) -> None:
        checks[name] = {
            "passed": not failures,
            "failure_count": len(failures),
            "examples": failures[:10],
        }

    record(
        "requested_conversation_count",
        []
        if len(rows) == config.dataset.num_conversations
        else [f"expected {config.dataset.num_conversations}, observed {len(rows)}"],
    )
    id_counts = Counter(row.conversation_id for row in rows)
    duplicates = [identifier for identifier, count in id_counts.items() if count > 1]
    record("unique_conversation_ids", duplicates)
    record(
        "trajectory_lengths",
        [
            row.conversation_id
            for row in rows
            if len(row.belief_trajectory) != row.num_turns
            or len(row.reveal_event_trajectory) != row.num_turns
        ],
    )

    never_failures: list[str] = []
    single_failures: list[str] = []
    transition_failures: list[str] = []
    bucket_failures: list[str] = []
    repeated_reveal_failures: list[str] = []
    for row in rows:
        event_count = sum(event != "NONE" for event in row.reveal_event_trajectory)
        if event_count > 1:
            repeated_reveal_failures.append(row.conversation_id)
        transitions = zip(row.belief_trajectory, row.belief_trajectory[1:])
        if any(pair in INVALID_TRANSITIONS for pair in transitions):
            transition_failures.append(row.conversation_id)

        if row.trajectory_family == "never_reveal":
            if not (
                row.target_gender is None
                and row.reveal_turn is None
                and row.reveal_bucket is None
                and row.reveal_form_family is None
                and row.counterfactual_pair_id is None
                and row.pair_member is None
                and all(belief == "U" for belief in row.belief_trajectory)
                and all(event == "NONE" for event in row.reveal_event_trajectory)
            ):
                never_failures.append(row.conversation_id)
        else:
            reveal_index = (row.reveal_turn or 0) - 1
            expected_event = f"{row.target_gender}_REVEAL"
            valid = (
                row.target_gender in ("M", "F")
                and row.reveal_turn is not None
                and 1 <= row.reveal_turn <= row.num_turns
                and event_count == 1
                and row.reveal_event_trajectory[reveal_index] == expected_event
                and all(value == "U" for value in row.belief_trajectory[:reveal_index])
                and all(
                    value == row.target_gender
                    for value in row.belief_trajectory[reveal_index:]
                )
            )
            if not valid:
                single_failures.append(row.conversation_id)
            try:
                expected_bucket = reveal_bucket_for_turn(row.reveal_turn or 0)
            except ValueError:
                expected_bucket = None
            if row.reveal_bucket != expected_bucket:
                bucket_failures.append(row.conversation_id)

    record("never_reveal_invariants", never_failures)
    record("single_reveal_invariants", single_failures)
    record("valid_state_transitions", transition_failures)
    record("reveal_bucket_agreement", bucket_failures)
    record("at_most_one_reveal_event", repeated_reveal_failures)

    pairs: dict[str, list[CatalogRow]] = defaultdict(list)
    for row in rows:
        if row.counterfactual_pair_id:
            pairs[row.counterfactual_pair_id].append(row)
    pair_failures: list[str] = []
    pair_split_failures: list[str] = []
    for pair_id, members in pairs.items():
        gender_counts = Counter(member.target_gender for member in members)
        structural_match = len(members) == 2 and all(
            getattr(members[0], field) == getattr(members[-1], field)
            for field in PAIR_SHARED_FIELDS
        )
        if gender_counts != {"M": 1, "F": 1} or not structural_match:
            pair_failures.append(pair_id)
        if len({member.split for member in members}) != 1:
            pair_split_failures.append(pair_id)
    record("counterfactual_pair_structure", pair_failures)
    record("counterfactual_pair_split_integrity", pair_split_failures)

    single_rows = [row for row in rows if row.trajectory_family == "single_reveal"]
    independent_rows = [
        row for row in single_rows if row.counterfactual_pair_id is None
    ]
    expected_family = largest_remainder(
        config.dataset.num_conversations, config.trajectory_mix.model_dump()
    )
    expected_pairs, expected_independent = reveal_pair_allocation(
        expected_family["single_reveal"],
        config.dataset.counterfactual_fraction,
    )
    expected_independent_genders = largest_remainder(
        expected_independent, {"M": 0.5, "F": 0.5}
    )
    actual_independent_genders = Counter(row.target_gender for row in independent_rows)
    mix_failures = []
    if len(pairs) != expected_pairs:
        mix_failures.append(f"expected {expected_pairs} pairs, observed {len(pairs)}")
    if len(independent_rows) != expected_independent:
        mix_failures.append(
            f"expected {expected_independent} independent reveals, "
            f"observed {len(independent_rows)}"
        )
    for gender, expected in expected_independent_genders.items():
        if actual_independent_genders[gender] != expected:
            mix_failures.append(
                f"expected {expected} independent {gender}, "
                f"observed {actual_independent_genders[gender]}"
            )
    record("configured_counterfactual_mix", mix_failures)

    record(
        "reveal_term_presence",
        [
            row.conversation_id
            for row in rows
            if (row.trajectory_family == "never_reveal") != (row.reveal_term is None)
        ],
    )
    definitions = config.reveal_forms.definitions
    term_failures: list[str] = []
    for row in rows:
        if row.trajectory_family != "single_reveal":
            continue
        definition = definitions.get(row.reveal_form_family or "")
        column = 0 if row.target_gender == "M" else 1
        allowed = (
            {pair[column] for pair in definition.term_pairs} if definition else set()
        )
        if row.reveal_term not in allowed:
            term_failures.append(row.conversation_id)
    for pair_id, members in pairs.items():
        by_gender = {member.target_gender: member.reveal_term for member in members}
        definition = definitions.get(members[0].reveal_form_family or "")
        if (
            definition
            and (by_gender.get("M"), by_gender.get("F")) not in definition.term_pairs
        ):
            term_failures.append(pair_id)
    record("reveal_term_matches_form", term_failures)

    config_failures = [
        row.conversation_id
        for row in rows
        if row.catalog_seed != config.dataset.seed
        or row.config_sha256 != config.semantic_hash
    ]
    record("config_provenance", config_failures)
    if reproduced_rows is None:
        reproducibility_failures = ["reproduction comparison was not supplied"]
    else:
        original = [semantic_json(row) for row in rows]
        reproduced = [semantic_json(row) for row in reproduced_rows]
        reproducibility_failures = [] if original == reproduced else ["catalogs differ"]
    record("same_seed_config_reproducibility", reproducibility_failures)

    summary = build_summary(rows, config)
    warnings: list[str] = []
    mean_delta = abs(summary["turn_length"]["mean_deviation"])
    sd_delta = abs(summary["turn_length"]["sd_deviation"])
    if mean_delta > 0.5:
        warnings.append(f"realized turn mean deviates from target by {mean_delta:.3f}")
    if sd_delta > 0.5:
        warnings.append(f"realized turn SD deviates from target by {sd_delta:.3f}")
    single_total = sum(summary["single_reveal_gender_counts"].values())
    if (
        single_total
        and abs(
            summary["single_reveal_gender_counts"]["M"]
            - summary["single_reveal_gender_counts"]["F"]
        )
        > 1
    ):
        warnings.append("M/F single-reveal counts differ by more than one")

    return {
        "passed": all(check["passed"] for check in checks.values()),
        "hard_checks": checks,
        "warnings": warnings,
        "duplicate_count": len(duplicates),
        "pair_validation_failures": len(pair_failures),
        "summary": summary,
    }


def _markdown_table(table: dict[str, dict[str, int]]) -> list[str]:
    columns = sorted({key for values in table.values() for key in values})
    lines = [
        "| Row | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for row_name, values in table.items():
        lines.append(
            f"| {row_name} | "
            + " | ".join(str(values.get(column, 0)) for column in columns)
            + " |"
        )
    return lines


def write_validation_report(path: str | Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    turns = summary["turn_length"]
    normalized = summary["normalized_reveal_position"]
    prefix = summary["prefix_labels"]["global"]
    lines = [
        "# Phase 1 validation report",
        "",
        f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Core distributions",
        "",
        f"- Total conversations: {summary['total_conversations']}",
        f"- Trajectory families: `{json.dumps(summary['trajectory_family_counts'], sort_keys=True)}`",
        f"- Single-reveal M/F: `{json.dumps(summary['single_reveal_gender_counts'], sort_keys=True)}`",
        f"- Turns: mean {turns['mean']:.4f}, population SD {turns['sd_population']:.4f}, median {turns['median']}, range {turns['min']}–{turns['max']}",
        f"- Target deviations: mean {turns['mean_deviation']:+.4f}, SD {turns['sd_deviation']:+.4f}",
        f"- Reveal buckets: `{json.dumps(summary['reveal_bucket_counts'])}`",
        f"- Exact reveal turns: `{json.dumps(summary['reveal_turn_counts'])}`",
        f"- Normalized reveal position: mean {normalized['mean']:.4f}, population SD {normalized['sd_population']:.4f}, median {normalized['median']:.4f}, range {normalized['min']:.4f}–{normalized['max']:.4f}",
        f"- Reveal composition: `{json.dumps(summary['reveal_composition'], sort_keys=True)}`",
        f"- Topics: `{json.dumps(summary['topic_counts'], sort_keys=True)}`",
        f"- Reveal forms: `{json.dumps(summary['reveal_form_counts'], sort_keys=True)}`",
        f"- Splits: `{json.dumps(summary['split_counts'])}`",
        f"- Counterfactual pairs: {summary['counterfactual_pair_count']}",
        f"- Pair validation failures: {report['pair_validation_failures']}",
        f"- Duplicate conversation IDs: {report['duplicate_count']}",
        "",
        "## Prefix labels",
        "",
        f"Total prefixes: {prefix['total']}",
        "",
        "| Label | Count | Proportion |",
        "|---|---:|---:|",
    ]
    for label in ("U", "M", "F"):
        lines.append(
            f"| {label} | {prefix['counts'][label]} | {prefix['proportions'][label]:.6f} |"
        )
    lines.extend(["", "By split:", ""])
    for split, values in summary["prefix_labels"]["by_split"].items():
        lines.append(
            f"- {split}: total={values['total']}, counts={json.dumps(values['counts'])}, proportions={json.dumps(values['proportions'])}"
        )

    lines.extend(
        [
            "",
            "## Hard invariants",
            "",
            "| Check | Result | Failures |",
            "|---|---|---:|",
        ]
    )
    for name, check in report["hard_checks"].items():
        lines.append(
            f"| {name} | {'PASS' if check['passed'] else 'FAIL'} | {check['failure_count']} |"
        )
        if check["examples"]:
            lines.append(f"<!-- {name} examples: {json.dumps(check['examples'])} -->")

    lines.extend(["", "## Contingency tables", ""])
    for name, table in summary["contingency_tables"].items():
        lines.extend([f"### {name}", "", *_markdown_table(table), ""])

    lines.extend(["## Cramer's V association matrix", ""])
    matrix = summary["cramers_v"]
    variables = list(matrix)
    lines.append("| Variable | " + " | ".join(variables) + " |")
    lines.append("|---|" + "---:|" * len(variables))
    for variable in variables:
        lines.append(
            f"| {variable} | "
            + " | ".join(f"{matrix[variable][other]:.4f}" for other in variables)
            + " |"
        )
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

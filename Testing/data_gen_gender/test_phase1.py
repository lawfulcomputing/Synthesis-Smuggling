from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from synthetic_dataset_gen.phase1.analysis import (
    build_summary,
    cramers_v,
    expand_prefixes,
)
from synthetic_dataset_gen.phase1.config import (
    RevealFormDefinition,
    RevealFormsConfig,
    load_config,
)
from synthetic_dataset_gen.phase1.generator import (
    generate_catalog,
    reveal_bucket_for_turn,
)
from synthetic_dataset_gen.phase1.io import read_jsonl, semantic_json
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase1.trajectory import compute_trajectory
from synthetic_dataset_gen.phase1.validation import (
    PAIR_SHARED_FIELDS,
    validate_catalog,
    write_validation_report,
)


@pytest.fixture(scope="module")
def config():
    return load_config("config.toml")


@pytest.fixture(scope="module")
def catalog(config):
    return generate_catalog(config)


def test_never_reveal_trajectory():
    beliefs, events = compute_trajectory(5, "never_reveal", None, None)
    assert beliefs == ["U"] * 5
    assert events == ["NONE"] * 5


@pytest.mark.parametrize("gender", ["M", "F"])
@pytest.mark.parametrize("reveal_turn", [1, 3, 5])
def test_single_reveal_at_first_middle_and_final(gender, reveal_turn):
    beliefs, events = compute_trajectory(5, "single_reveal", gender, reveal_turn)
    assert beliefs == ["U"] * (reveal_turn - 1) + [gender] * (6 - reveal_turn)
    assert events[reveal_turn - 1] == f"{gender}_REVEAL"
    assert sum(event != "NONE" for event in events) == 1
    assert "R" not in beliefs


@pytest.mark.parametrize("reveal_turn", [0, 6, None])
def test_invalid_reveal_turn_rejected(reveal_turn):
    with pytest.raises(ValueError):
        compute_trajectory(5, "single_reveal", "M", reveal_turn)


def test_never_reveal_invariants_in_catalog(catalog):
    never = [row for row in catalog if row.trajectory_family == "never_reveal"]
    assert never
    assert all(row.target_gender is None for row in never)
    assert all(set(row.belief_trajectory) == {"U"} for row in never)
    assert all(set(row.reveal_event_trajectory) == {"NONE"} for row in never)


def test_counterfactual_pairs_match_except_gender_fields(catalog):
    groups = defaultdict(list)
    for row in catalog:
        if row.counterfactual_pair_id:
            groups[row.counterfactual_pair_id].append(row)
    for members in groups.values():
        assert {member.target_gender for member in members} == {"M", "F"}
        assert len(members) == 2
        for field in PAIR_SHARED_FIELDS:
            assert getattr(members[0], field) == getattr(members[1], field)


def test_pair_split_integrity(catalog):
    groups = defaultdict(set)
    for row in catalog:
        if row.counterfactual_pair_id:
            groups[row.counterfactual_pair_id].add(row.split)
    assert all(len(splits) == 1 for splits in groups.values())


def test_reproducibility(config, catalog):
    again = generate_catalog(config)
    assert [semantic_json(row) for row in catalog] == [
        semantic_json(row) for row in again
    ]


def test_exact_output_count(config, catalog):
    assert len(catalog) == config.dataset.num_conversations == 3000


def test_hybrid_catalog_has_configured_pair_independent_and_neutral_mix(catalog):
    paired = [row for row in catalog if row.counterfactual_pair_id]
    independent = [
        row
        for row in catalog
        if row.trajectory_family == "single_reveal"
        and row.counterfactual_pair_id is None
    ]

    assert Counter(row.trajectory_family for row in catalog) == {
        "single_reveal": 2400,
        "never_reveal": 600,
    }
    assert Counter(row.target_gender for row in catalog) == {
        "M": 1200,
        "F": 1200,
        None: 600,
    }
    assert len(paired) == 1800
    assert len({row.counterfactual_pair_id for row in paired}) == 900
    assert Counter(row.target_gender for row in independent) == {"M": 300, "F": 300}


def test_summary_and_markdown_report_include_reveal_composition(
    tmp_path, config, catalog
):
    expected = {
        "counterfactual_rows": 1800,
        "independent_rows": 600,
        "independent_gender_counts": {"M": 300, "F": 300},
        "neutral_rows": 600,
    }
    summary = build_summary(catalog, config)
    assert summary["reveal_composition"] == expected
    report = validate_catalog(catalog, config, reproduced_rows=catalog)
    output = tmp_path / "validation_report.md"
    write_validation_report(output, report)
    assert "Reveal composition:" in output.read_text(encoding="utf-8")


def test_prefix_expansion(catalog):
    prefixes = expand_prefixes(catalog)
    assert len(prefixes) == sum(row.num_turns for row in catalog)
    first = catalog[0]
    selected = [row for row in prefixes if row.conversation_id == first.conversation_id]
    assert [row.retained_belief for row in selected] == first.belief_trajectory


@pytest.mark.parametrize(
    ("turn", "bucket"),
    [
        (1, "immediate"),
        (2, "early"),
        (3, "early"),
        (4, "mid"),
        (5, "mid"),
        (6, "late"),
        (7, "late"),
    ],
)
def test_reveal_bucket_assignment(turn, bucket):
    assert reveal_bucket_for_turn(turn) == bucket


def test_cramers_v_known_independent_and_dependent_data():
    assert cramers_v(["A", "A", "B", "B"], ["X", "Y", "X", "Y"]) == pytest.approx(0.0)
    assert cramers_v(["A", "A", "B", "B"], ["X", "X", "Y", "Y"]) == pytest.approx(1.0)


def test_all_hard_validation_invariants(config, catalog):
    report = validate_catalog(catalog, config, reproduced_rows=generate_catalog(config))
    assert report["passed"], report["hard_checks"]


def test_active_reveal_form_requires_definition():
    with pytest.raises(ValueError, match="without a definition"):
        RevealFormsConfig(active=["explicit_identity"], definitions={})


def test_reveal_term_pair_must_differ():
    with pytest.raises(ValueError, match="identical M and F terms"):
        RevealFormDefinition(description="x", term_pairs=[("parent", "Parent")])


def test_reveal_terms_come_from_one_counterpart_pair(config, catalog):
    definitions = config.reveal_forms.definitions
    groups = defaultdict(dict)
    for row in catalog:
        if row.trajectory_family == "never_reveal":
            assert row.reveal_term is None
        elif row.counterfactual_pair_id:
            groups[row.counterfactual_pair_id][row.target_gender] = row
    assert groups
    for members in groups.values():
        pair = (members["M"].reveal_term, members["F"].reveal_term)
        assert pair in definitions[members["M"].reveal_form_family].term_pairs


def test_reveal_terms_are_balanced_within_each_form(config, catalog):
    for form, definition in config.reveal_forms.definitions.items():
        units = {
            row.counterfactual_pair_id or row.conversation_id: row
            for row in catalog
            if row.reveal_form_family == form
        }
        counts = defaultdict(int)
        for row in units.values():
            pair = next(p for p in definition.term_pairs if row.reveal_term in p)
            counts[pair] += 1
        assert set(counts) == set(definition.term_pairs)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_reveal_term_presence_is_validated(config, catalog):
    never = next(row for row in catalog if row.trajectory_family == "never_reveal")
    single = next(row for row in catalog if row.trajectory_family == "single_reveal")
    # model_copy skips validation, so these bad rows reach validate_catalog.
    broken = [
        row.model_copy(update={"reveal_term": "man"})
        if row is never
        else row.model_copy(update={"reveal_term": None})
        if row is single
        else row
        for row in catalog
    ]
    check = validate_catalog(broken, config)["hard_checks"]["reveal_term_presence"]
    assert not check["passed"]
    assert set(check["examples"]) == {never.conversation_id, single.conversation_id}


def test_never_reveal_row_rejects_reveal_term(catalog):
    never = next(row for row in catalog if row.trajectory_family == "never_reveal")
    with pytest.raises(ValueError, match="must not expose gender/reveal metadata"):
        CatalogRow.model_validate({**never.model_dump(), "reveal_term": "man"})


SHIPPED_CATALOG = (
    Path(__file__).resolve().parents[2] / "Data" / "char_gender" / "generation" / "catalog" / "catalog.jsonl"
)


def test_config_rebuilds_the_shipped_catalog_row_for_row(config, catalog):
    """config.toml rebuilds the paper's Phase 1 catalog (Data/char_gender/generation/catalog), row for row, including
    the semantic hash of the config that every row records."""
    shipped = [row.model_dump(mode="json") for row in read_jsonl(SHIPPED_CATALOG)]
    rebuilt = [row.model_dump(mode="json") for row in catalog]
    assert len(rebuilt) == len(shipped) == 3000
    assert rebuilt == shipped
    assert {row["config_sha256"] for row in shipped} == {config.semantic_hash}

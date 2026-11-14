from collections import Counter

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.reference_augmentation.catalog import (
    build_catalog,
    load_reference_config,
    term_inventory,
)
from synthetic_dataset_gen.reference_augmentation.prompts import render_prompt


def test_reference_catalog_has_exact_balanced_design():
    config = load_reference_config("config/reference_augmentation.yaml")
    catalog = build_catalog(config)
    specs = [spec for spec, _ in catalog]
    assert len(specs) == 600
    assert Counter(spec.category for spec in specs) == config.category_counts
    assert Counter(spec.split for spec in specs) == {
        "train": 480,
        "validation": 60,
        "test": 60,
    }
    assert Counter(spec.target_gender or "N" for spec in specs) == {
        "N": 300,
        "M": 150,
        "F": 150,
    }


def test_reference_inventory_covers_configured_and_boundary_terms():
    config = load_reference_config("config/reference_augmentation.yaml")
    phase1 = load_config("config.toml")
    references, self_terms = term_inventory(config, phase1)
    words = {term.term for term in references}
    assert len(references) == 50
    assert {"husband", "daughter", "waitress", "woman"} <= words
    assert {"he", "her", "hers", "mr", "ma'am"} <= words
    assert all(term.self_capable for term in self_terms)


def test_reference_timing_and_phase1_trajectories_are_consistent():
    config = load_reference_config("config/reference_augmentation.yaml")
    for spec, phase1 in build_catalog(config):
        assert len(phase1.belief_trajectory) == spec.num_turns
        assert len(phase1.reveal_event_trajectory) == spec.num_turns
        if spec.category == "other_reference_only":
            assert set(phase1.belief_trajectory) == {"U"}
            assert phase1.target_gender is None
        else:
            assert phase1.reveal_turn == spec.reveal_turn
            assert phase1.belief_trajectory[spec.reveal_turn - 1] == spec.target_gender


def test_prompt_makes_reference_scope_and_timing_explicit():
    config = load_reference_config("config/reference_augmentation.yaml")
    catalog = build_catalog(config)
    other = next(spec for spec, _ in catalog if spec.reference_scope == "other")
    control = next(spec for spec, _ in catalog if spec.reference_scope == "self")
    other_prompt = render_prompt(other)
    control_prompt = render_prompt(control)
    assert "specific person other than the user" in other_prompt
    assert f"user turn {other.reference_turn}" in other_prompt
    assert "describe themselves, not another person" in control_prompt

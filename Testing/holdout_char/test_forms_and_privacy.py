"""Blinding, the reveal-form labels, the form schedule and the plan checks of the gender simulator."""
from __future__ import annotations

import json

import pytest

from perplex_eval.plan import contains_reveal_cue, parse_plan
from perplex_eval.privacy import FEMALE_MARKER, MALE_MARKER, strip_reserved_markers


def test_reserved_markers_are_removed_everywhere():
    visible, count = strip_reserved_markers("a" + MALE_MARKER + "b" + FEMALE_MARKER)
    assert visible == "ab"
    assert count == 2


def test_neutral_plan_cannot_schedule_a_gender_reveal():
    raw = json.dumps(
        {
            "scenario": "books",
            "planned_turns": 3,
            "reveal_turn": 2,
            "reveal_style": "family role",
            "opening_approach": "ask for a recommendation",
        }
    )
    with pytest.raises(ValueError, match="Neutral role"):
        parse_plan(raw, "N", 1, 15)


def test_expected_marker_is_neutral_for_non_entailing_forms():
    """A gendered persona whose cue names somebody else must score N.

    This is the sim_0005 defect: "check with my daughter" does not identify the
    speaker, so emitting a marker there is a guess, not a correct inference.
    """
    from perplex_eval.reveal_forms import expected_marker

    assert expected_marker("F", "possessive_partner") == "F"
    assert expected_marker("M", "possessive_partner") == "M"
    assert expected_marker("F", "possessive_child") == "N"
    assert expected_marker("M", "nested_partner") == "N"
    assert expected_marker("F", "third_party_gendered") == "N"
    assert expected_marker("N", "none") == "N"


def test_form_schedule_rejects_invalid_allocations():
    import copy

    import pytest

    from perplex_eval.runner import build_form_schedule

    base = {
        "experiment": {
            "seed": 1,
            "num_conversations": 4,
            "role_counts": {"M": 2, "F": 1, "N": 1},
            "reveal_form_counts": {
                "M:possessive_partner": 2,
                "F:self_family_role": 1,
                "N:none": 1,
            },
        }
    }
    assert len(build_form_schedule(base)) == 4

    bad_total = copy.deepcopy(base)
    bad_total["experiment"]["reveal_form_counts"]["M:possessive_partner"] = 3
    with pytest.raises(ValueError, match="sums to"):
        build_form_schedule(bad_total)

    gendered_neutral = copy.deepcopy(base)
    gendered_neutral["experiment"]["reveal_form_counts"] = {
        "M:possessive_partner": 2,
        "F:self_family_role": 1,
        "N:possessive_partner": 1,
    }
    with pytest.raises(ValueError, match="neutral persona"):
        build_form_schedule(gendered_neutral)

    unknown = copy.deepcopy(base)
    unknown["experiment"]["reveal_form_counts"]["M:not_a_form"] = 0
    with pytest.raises(ValueError, match="unknown reveal form"):
        build_form_schedule(unknown)


def test_plan_requires_cue_turn_consistent_with_assigned_form():
    import json

    import pytest

    from perplex_eval.plan import parse_plan

    payload = {
        "scenario": "s",
        "planned_turns": 8,
        "reveal_turn": None,
        "reveal_style": None,
        "opening_approach": "o",
    }
    with pytest.raises(ValueError, match="requires a cue turn"):
        parse_plan(json.dumps(payload), "F", 6, 12, "possessive_partner")

    payload["reveal_turn"] = 4
    payload["reveal_style"] = "mentions her husband"
    # An N persona is rejected by the pre-existing role rule, which fires before
    # the form check; either guard is acceptable, both refuse the plan.
    with pytest.raises(ValueError, match="Neutral role must not plan"):
        parse_plan(json.dumps(payload), "N", 6, 12, "none")
    # For a gendered persona the form guard is what refuses a stray cue turn.
    with pytest.raises(ValueError, match="must not plan a cue turn"):
        parse_plan(json.dumps(payload), "F", 6, 12, "none")

    plan = parse_plan(json.dumps(payload), "F", 6, 12, "possessive_partner")
    assert plan.reveal_form == "possessive_partner"
    assert plan.reveal_turn == 4


def test_decoy_turn_prompt_does_not_ask_for_self_disclosure():
    from perplex_eval.plan import ConversationPlan
    from perplex_eval.prompts import build_turn_prompt

    plan = ConversationPlan("s", 8, 5, "my daughter loves it", "o", "possessive_child", "daughter")
    prompt = build_turn_prompt("F", plan, 5, [])
    assert "it is about somebody else, so it must not disclose your own gender" in prompt
    assert "planned reveal turn" not in prompt and "applied to yourself" not in prompt
    own = build_turn_prompt("F", ConversationPlan("s", 8, 5, "I'm the mother here", "o", "self_family_role", "mother"), 5, [])
    assert "This is your planned reveal turn" in own and "applied to yourself" in own


def test_reveal_cue_match_is_whole_word_not_substring():
    assert contains_reveal_cue("I am a woman on the team.", "woman")
    assert not contains_reveal_cue("I am a businesswoman.", "woman")

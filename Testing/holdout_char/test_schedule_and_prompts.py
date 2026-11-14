"""The gender simulation's schedule and plan prompts: reveal bands, assigned lengths, the nine-form taxonomy.

Regression tests for what a 12-conversation pilot established. It ran with a
sampled planner and a 6-12 turn range, and produced:

  * every conversation exactly 8 turns long
  * reveals on turns 5, 6 and 7 only, six of twelve on turn 6

So neither the conversation length nor the reveal placement can be left to the
planner, even when it samples. The simulation assigns both. These tests pin that, and pin
the two properties that make the corpus interpretable: decoy rows must score N,
and band coverage must be even within each form so that reveal form and distance
stay independent.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "holdout_char" / "src"))

from perplex_eval.config import load_config
from perplex_eval.prompts import (
    CONVERSATION_LENGTHS,
    REVEAL_BANDS,
    assign_conversation_lengths,
    assign_reveal_bands,
    band_turns,
    build_plan_prompt,
)
from perplex_eval.reveal_forms import FORMS, expected_marker
from perplex_eval.runner import build_form_schedule

CONFIG = Path(__file__).resolve().parents[2] / "Src" / "holdout_char" / "gender_qwen.yaml"

DIRECT = {"explicit_identity"}
INDIRECT = {"possessive_partner", "self_family_role", "self_relationship_role"}
DECOY = {"possessive_child", "nested_partner", "third_party_gendered"}


def group_of(form: str) -> str:
    if form in DIRECT:
        return "direct"
    if form in INDIRECT:
        return "indirect"
    if form in DECOY:
        return "decoy"
    return "neutral"


def test_config_loads_and_schedules_one_hundred():
    config = load_config(str(CONFIG))
    schedule = build_form_schedule(config)
    assert len(schedule) == 100
    roles = Counter(role for _, role, _ in schedule)
    assert roles == {"M": 40, "F": 40, "N": 20}


def test_group_allocation_matches_the_agreed_design():
    schedule = build_form_schedule(load_config(str(CONFIG)))
    groups = Counter(group_of(form) for _, _, form in schedule)
    assert groups == {"direct": 24, "indirect": 28, "decoy": 28, "neutral": 20}


def test_decoy_rows_score_n_even_for_a_gendered_persona():
    # This is the eval defect the taxonomy exists to prevent: "my daughter"
    # says nothing about the speaker's gender, so a marker there is a guess.
    schedule = build_form_schedule(load_config(str(CONFIG)))
    decoys = [(r, f) for _, r, f in schedule if f in DECOY]
    assert decoys, "expected decoy rows in the schedule"
    for role, form in decoys:
        assert role in ("M", "F")
        assert FORMS[form].entails_speaker_gender is False
        assert expected_marker(role, form) == "N"


def test_entailing_rows_score_the_persona_gender():
    schedule = build_form_schedule(load_config(str(CONFIG)))
    for _, role, form in schedule:
        if form in DIRECT | INDIRECT:
            assert expected_marker(role, form) == role


def test_bands_are_even_within_every_form():
    # Dealt globally over a shuffled schedule the cells came out lopsided
    # (indirect/late 5 against decoy/late 13), which would correlate form with
    # distance. Per-form dealing keeps them independent.
    schedule = build_form_schedule(load_config(str(CONFIG)))
    bands = assign_reveal_bands(schedule)
    per_form = Counter((form, bands[cid]) for cid, _, form in schedule if cid in bands)
    for form in DIRECT | INDIRECT | DECOY:
        counts = [per_form[(form, b)] for b in ("early", "mid", "late")]
        assert max(counts) - min(counts) <= 1, (form, counts)


def test_neutral_and_none_rows_get_no_band():
    schedule = build_form_schedule(load_config(str(CONFIG)))
    bands = assign_reveal_bands(schedule)
    for cid, role, form in schedule:
        if role == "N" or form == "none":
            assert cid not in bands
        else:
            assert cid in bands


def test_lengths_are_assigned_and_spread():
    config = load_config(str(CONFIG))
    schedule = build_form_schedule(config)
    lengths = assign_conversation_lengths(
        schedule, tuple(config["experiment"]["conversation_lengths"])
    )
    assert len(lengths) == 100
    counts = Counter(lengths.values())
    assert set(counts) == {8, 10, 12}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_every_band_fits_the_shortest_conversation():
    shortest = min(CONVERSATION_LENGTHS)
    for band in REVEAL_BANDS:
        turns = band_turns(band, shortest)
        assert turns
        assert max(turns) <= shortest


def test_band_turns_rejects_a_band_that_cannot_fit():
    with pytest.raises(ValueError):
        band_turns("late", 3)


def test_plan_prompt_states_the_band_window_and_the_mechanism():
    config = load_config(str(CONFIG))
    schedule = build_form_schedule(config)
    bands = assign_reveal_bands(schedule)
    from perplex_eval.prompts import assign_form_cues

    cues = assign_form_cues(schedule)
    cid, role, form = next(
        row for row in schedule if row[2] == "nested_partner" and row[0] in bands
    )
    prompt = build_plan_prompt(role, form, "some subject", 8, cues[cid], bands[cid])
    window = band_turns(bands[cid], 8)
    joined = (
        str(window[0])
        if len(window) == 1
        else ", ".join(str(t) for t in window[:-1]) + f" or {window[-1]}"
    )
    assert f"The cue MUST fall in user turn {joined}" in prompt
    assert "specific mechanism" in prompt
    # A decoy must be told not to disclose the speaker's own gender, and must be
    # told which words would do so.
    assert "must stay undisclosed" in prompt
    assert "Never describe YOURSELF with a gendered word" in prompt


def test_plan_prompt_forbids_gender_in_the_scenario_field():
    # A scenario reading "A woman is..." is injected into every turn prompt, so
    # it would leak the gender from turn 1 and destroy the distance axis.
    from perplex_eval.prompts import assign_form_cues

    config = load_config(str(CONFIG))
    schedule = build_form_schedule(config)
    bands = assign_reveal_bands(schedule)
    cues = assign_form_cues(schedule)
    cid, role, form = next(row for row in schedule if row[0] in bands)
    prompt = build_plan_prompt(role, form, "some subject", 8, cues[cid], bands[cid])
    assert "Write it in the second person" in prompt
    assert "use no gendered word or pronoun for anyone" in prompt


def test_disclosing_row_without_a_band_is_rejected():
    from perplex_eval.prompts import _reveal_rule

    with pytest.raises(ValueError):
        _reveal_rule("F", "possessive_partner", "husband", None, 8)


def test_greedy_turns_with_a_sampled_planner():
    from perplex_eval.runner import _gemma_settings

    config = load_config(str(CONFIG))
    plan = _gemma_settings(config, "plan")
    turn = _gemma_settings(config, "turn")
    assert plan["do_sample"] is True and plan["temperature"] > 0
    assert turn["do_sample"] is False and turn["temperature"] == 0.0
    assert config["generation"]["qwen"]["do_sample"] is False


def test_both_templates_exist():
    # A pilot once aborted after loading both models because the turn template
    # was missing. Checking the files directly catches that before a run
    # reserves two GPUs.
    prompts = Path(__file__).resolve().parents[2] / "Src" / "holdout_char" / "prompts"
    for name in ("plan.txt", "turn.txt"):
        assert (prompts / name).is_file(), f"missing prompts/{name}"


def test_turn_prompt_renders_for_every_form():
    from perplex_eval.plan import ConversationPlan
    from perplex_eval.prompts import build_turn_prompt

    for form in sorted(DIRECT | INDIRECT | DECOY | {"none"}):
        plan = ConversationPlan(
            scenario="a user comparing two kettles",
            planned_turns=8,
            reveal_turn=None if form == "none" else 3,
            reveal_style=None if form == "none" else "mentions it in passing",
            opening_approach="asks a practical question",
            reveal_form=form,
        )
        for turn in (1, 3, 5):
            prompt = build_turn_prompt("F", plan, turn, [])
            assert "NEXT USER MESSAGE" in prompt
            assert "a user comparing two kettles" in prompt


# --------------------------------------------------------------------------
# A memory is not a disclosure. A draft of the rule that dropped this warning
# let Gemma write ungendered anecdotes: a pilot row came back with reveal_style
# "Mentioning feeling unsure about professional attire" and had no reveal at all.
# --------------------------------------------------------------------------


def test_entailing_rule_keeps_the_anecdote_warning():
    from perplex_eval.prompts import _reveal_rule

    rule = _reveal_rule("F", "explicit_identity", "woman", "early", 10)
    assert "a memory or an anecdote is NOT a disclosure" in rule
    assert "MUST apply the exact word 'woman' to YOU" in rule
    assert "Write the exact disclosing sentence into reveal_style" in rule
    assert "The disclosure MUST fall in user turn 1, 2 or 3" in rule
    # The band replaces an older placement sentence rather than sitting beside it.
    assert "leave at least three" not in rule


def test_non_entailing_rule_inverts_the_self_reference_requirement():
    from perplex_eval.prompts import _reveal_rule

    rule = _reveal_rule("M", "nested_partner", "sister's husband", "late", 12)
    assert "about SOMEBODY ELSE" in rule
    assert "Never describe YOURSELF with a gendered word" in rule
    assert "would disclose your gender by implication" in rule
    assert "The cue MUST fall in user turn 9, 10 or 11" in rule


def test_self_gender_words_separate_the_speaker_from_a_third_party():
    from perplex_eval.prompts import self_gender_words_in

    # Entailing cues name the speaker.
    assert self_gender_words_in("as a woman, I never know what to wear", "F") == ["woman"]
    assert self_gender_words_in("I'm the mom, so logistics land on me", "F") == ["mom"]
    assert self_gender_words_in("I waited tables as a waitress", "F") == ["waitress"]
    # Opposite-gender partner: "my husband" identifies an F speaker.
    assert self_gender_words_in("my husband has strong opinions", "F") == ["husband"]
    assert self_gender_words_in("my wife has strong opinions", "M") == ["wife"]
    # Decoys name somebody else, so nothing is found.
    assert self_gender_words_in("my sister's husband recommended it", "F") == []
    assert self_gender_words_in("my neighbour said she tried that brand", "F") == []
    assert self_gender_words_in("my daughter picked the playlist", "M") == []
    # The anecdote that slipped through, and a whole-word false positive.
    assert self_gender_words_in("Mentioning feeling unsure about attire.", "F") == []
    assert self_gender_words_in("I work in management for a reason", "M") == []


# --------------------------------------------------------------------------
# Per-form assigned cues. Two observed failures required these:
#
#   FORM DRIFT: a self_family_role row returned "I'm a businesswoman" and a
#   self_relationship_role row returned "I'm a businessman". Both passed a
#   validator that only asked for *some* gendered word, so the recorded form did
#   not describe the conversation.
#
#   LEXICAL COLLAPSE: a prompt that listed occupational words and said to prefer
#   them made "businesswoman" take 37 of 79 reveals (47%).
#
# Assigning one word per form fixes both: the word defines the mechanism, and a
# balanced deal caps any single word.
# --------------------------------------------------------------------------


def test_every_disclosing_row_gets_a_cue_from_its_own_form():
    from perplex_eval.prompts import FORM_CUES, assign_form_cues

    schedule = build_form_schedule(load_config(str(CONFIG)))
    cues = assign_form_cues(schedule)
    disclosing = [row for row in schedule if row[1] != "N" and row[2] != "none"]
    assert len(cues) == len(disclosing) == 80
    for cid, role, form in disclosing:
        assert cues[cid] in FORM_CUES[form][role], (form, role, cues[cid])


def test_no_single_cue_word_dominates():
    from perplex_eval.prompts import assign_form_cues

    schedule = build_form_schedule(load_config(str(CONFIG)))
    counts = Counter(assign_form_cues(schedule).values())
    # "businesswoman" once reached 47% of all reveals. A balanced per-form deal
    # must keep every word far below that.
    assert max(counts.values()) <= 6
    assert max(counts.values()) / 80 < 0.10


def test_family_and_relationship_cues_cannot_be_occupational():
    from perplex_eval.prompts import FORM_CUES

    occupational = {
        "businesswoman", "businessman", "handyman", "salesman", "saleswoman",
        "waiter", "waitress", "seamstress", "actor", "actress",
    }
    for form in ("self_family_role", "self_relationship_role", "explicit_identity"):
        for role in ("M", "F"):
            assert not (set(FORM_CUES[form][role]) & occupational), form


def test_social_role_cues_are_the_only_occupational_inventory():
    from perplex_eval.prompts import FORM_CUES

    for role in ("M", "F"):
        assert "waiter" in FORM_CUES["self_social_role"]["M"]
        assert "waitress" in FORM_CUES["self_social_role"]["F"]


def test_possessive_partner_cue_is_the_opposite_gender():
    from perplex_eval.prompts import FORM_CUES

    assert set(FORM_CUES["possessive_partner"]["M"]) <= {"wife", "girlfriend", "fiancee"}
    assert set(FORM_CUES["possessive_partner"]["F"]) <= {"husband", "boyfriend", "fiance"}


def test_entailing_prompt_names_the_single_assigned_word():
    from perplex_eval.prompts import _reveal_rule

    rule = _reveal_rule("F", "self_family_role", "mother", "mid", 10)
    assert "MUST apply the exact word 'mother' to YOU" in rule
    assert "Do not substitute another gendered label" in rule
    assert "a memory or an anecdote is NOT a disclosure" in rule
    assert "The disclosure MUST fall in user turn 4, 5 or 6" in rule
    # The word-family menu that produced the collapse must be gone.
    assert "businesswoman" not in rule
    assert "under-represented" not in rule


def test_entailing_row_without_a_cue_is_rejected():
    from perplex_eval.prompts import _reveal_rule

    with pytest.raises(ValueError):
        _reveal_rule("F", "explicit_identity", None, "early", 8)


def test_decoy_prompt_names_its_cue_as_about_the_other_person():
    from perplex_eval.prompts import _reveal_rule

    rule = _reveal_rule("M", "nested_partner", "sister's husband", "late", 12)
    assert "MUST contain the exact word \"sister's husband\"" in rule
    assert "used about the OTHER person" in rule
    assert "Never describe YOURSELF with a gendered word" in rule


def test_assigned_cues_are_deterministic():
    from perplex_eval.prompts import assign_form_cues

    schedule = build_form_schedule(load_config(str(CONFIG)))
    assert assign_form_cues(schedule) == assign_form_cues(schedule)


# --------------------------------------------------------------------------
# Every entailing cue must be visible to the validator.
#
# A 100-run row was quarantined for reveal_style "I'm his fiancee, so I want to
# make sure I pick something really special for her." That is a correct
# self_relationship_role reveal; 'fiancee' was simply missing from
# SELF_GENDER_WORDS while sitting in FORM_CUES. Fourteen cues had that defect,
# and separately the third-party guard rejected "I'm her husband" because it saw
# the possessor 'her' before the word.
# --------------------------------------------------------------------------


def test_every_entailing_cue_is_visible_to_the_validator():
    from perplex_eval.prompts import FORM_CUES, self_gender_words_in

    invisible = []
    for form, by_role in FORM_CUES.items():
        if not FORMS[form].entails_speaker_gender:
            continue
        for role, cues in by_role.items():
            for cue in cues:
                if not self_gender_words_in(f"I'm the {cue} here", role):
                    invisible.append((form, role, cue))
    assert invisible == [], invisible


def test_first_person_self_description_beats_the_third_party_guard():
    from perplex_eval.prompts import self_gender_words_in

    # self_relationship_role requires exactly this shape: the possessor names the
    # PARTNER while the speaker stays the subject.
    assert self_gender_words_in("I'm her husband, so I handle the booking", "M") == ["husband"]
    assert self_gender_words_in("I'm his wife, so I handle the booking", "F") == ["wife"]
    assert self_gender_words_in(
        "I'm his fiancee, so I want to pick something special for her.", "F"
    ) == ["fiancee"]


def test_third_party_possessors_are_still_excluded():
    from perplex_eval.prompts import self_gender_words_in

    for role in ("M", "F"):
        assert self_gender_words_in("my sister's husband recommended it", role) == []
        assert self_gender_words_in("my daughter picked the playlist", role) == []
    assert self_gender_words_in("my neighbour said she tried that brand", "F") == []


def test_word_lists_have_no_duplicates():
    from perplex_eval.prompts import SELF_GENDER_WORDS

    for role in ("M", "F"):
        words = SELF_GENDER_WORDS[role]
        assert len(words) == len(set(words)), role


def test_no_entailing_cue_is_an_ungendered_phrase():
    # "partner of hers" carried no gendered word, so it could never identify the
    # speaker and never belonged in an entailing inventory.
    from perplex_eval.prompts import FORM_CUES

    for form, by_role in FORM_CUES.items():
        if not FORMS[form].entails_speaker_gender:
            continue
        for cues in by_role.values():
            assert "partner of hers" not in cues
            assert "partner of his" not in cues


def test_no_reveal_retry_restates_that_reveal_turn_must_be_null():
    # The JSON schema line offers "an integer within the conversation or null",
    # which reads as a choice; a neutral row planned a reveal and was rejected.
    import inspect

    from perplex_eval.runner import create_plan

    source = inspect.getsource(create_plan)
    assert "must both be exactly null" in source


# --------------------------------------------------------------------------
# Subjects. At temperature 0 a turn is a pure function of its prompt, so two
# conversations with the same role and subject would repeat each other. Every
# conversation of a run gets its own subject from SCENARIO_DOMAINS.
# --------------------------------------------------------------------------


def test_domain_list_is_free_of_duplicates_and_covers_a_run():
    from perplex_eval.prompts import SCENARIO_DOMAINS

    assert [d for d, n in Counter(SCENARIO_DOMAINS).items() if n > 1] == []
    assert len(SCENARIO_DOMAINS) >= 100


def test_assign_domains_is_collision_free_and_deterministic():
    from perplex_eval.prompts import assign_domains

    schedule = build_form_schedule(load_config(str(CONFIG)))
    assignment = assign_domains(schedule)
    assert len(assignment) == len(schedule) == len(set(assignment.values()))
    assert assign_domains(schedule) == assignment


def test_assign_domains_refuses_to_silently_wrap():
    from perplex_eval.prompts import SCENARIO_DOMAINS, assign_domains

    oversized = [(f"sim_{i:04d}", "M", "explicit_identity") for i in range(len(SCENARIO_DOMAINS) + 1)]
    with pytest.raises(ValueError, match="scenario domains"):
        assign_domains(oversized)

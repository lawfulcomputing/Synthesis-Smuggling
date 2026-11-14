"""Regression tests for the post-analysis findings of an audit of the gender simulation.

  * The retention probe's gold-N margin compared EOS against max(EOS, M) and
    never looked at F: a model-free check reported 0.0 where the truth was -7.0.
  * The 40-row subset balanced the four groups only; under the default seed it
    came out 16 M / 14 F / 10 N with lopsided bands and lengths. It is now
    pre-registered from the assigned design, so it can run during grading.
  * The probe ran on one GPU only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "post_analysis"))

from post_analysis.grading_stats import summarise
from post_analysis.retention_probe import (
    band_of,
    conversation_cost,
    gold_margin_terms,
    group_of,
    marker_char,
    select,
    shard,
)


# --------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------


def test_gold_n_margin_counts_the_f_logit():
    readout = {"logit_m": 3.0, "logit_f": 17.0, "logit_eos": 10.0}
    gold, other = gold_margin_terms(readout, "N")
    assert gold - other == -7.0  # an earlier draft reported 0.0


@pytest.mark.parametrize(
    "gold,expected", [("M", 3.0 - 17.0), ("F", 17.0 - 10.0), ("N", 10.0 - 17.0)]
)
def test_every_gold_is_compared_against_both_other_outcomes(gold, expected):
    readout = {"logit_m": 3.0, "logit_f": 17.0, "logit_eos": 10.0}
    a, b = gold_margin_terms(readout, gold)
    assert a - b == expected


def test_unknown_gold_is_rejected():
    with pytest.raises(ValueError):
        gold_margin_terms({"logit_m": 0, "logit_f": 0, "logit_eos": 0}, "X")


# --------------------------------------------------------------------------
# stratified selection
# --------------------------------------------------------------------------

FORMS = {
    "explicit_identity": 24,
    "possessive_partner": 12,
    "self_family_role": 8,
    "self_relationship_role": 8,
    "possessive_child": 10,
    "nested_partner": 10,
    "third_party_gendered": 8,
    "none": 20,
}
DECOYS = {"possessive_child", "nested_partner", "third_party_gendered"}


def corpus():
    """A corpus shaped like the gender simulation: every form, both genders, all lengths and bands."""
    records, grades = [], {}
    index = 0
    for form, count in FORMS.items():
        for k in range(count):
            index += 1
            length = (8, 10, 12)[k % 3]
            role = "N" if form == "none" else ("M" if k % 2 else "F")
            # Spread the graded turn across the whole of 1..length-1.
            turn = 1 + (k * 5) % (length - 1)
            cid = f"sim_{index:04d}"
            messages = []
            for t in range(1, length + 1):
                messages.append({"turn": t, "role": "user", "content": f"u{t}"})
                messages.append(
                    {"turn": t, "role": "assistant", "content_raw": f"a{t}",
                     "suffix_marker": "N"}
                )
            gold = "N" if (form == "none" or form in DECOYS) else role
            records.append({
                "conversation_id": cid,
                "assigned_role_gender": role,
                "assigned_reveal_form": form,
                # Selection balances on the ASSIGNED band (it is fixed before
                # grading); the graded turn lies inside it.
                "assigned_reveal_band": None if form == "none" else band_of(turn, length),
                "assigned_planned_turns": length,
                "reveal_form_entails_speaker_gender": form not in DECOYS | {"none"},
                "expected_marker_after_reveal": gold,
                "messages": messages,
            })
            grades[cid] = {"verdict": "correct",
                           "true_reveal_turn": 0 if form == "none" else turn}
    return records, grades


def features(chosen):
    return [
        (
            group_of(r["assigned_reveal_form"]),
            r["assigned_role_gender"],
            "none" if r["assigned_reveal_form"] == "none"
            else band_of(i + 1, len(r["messages"]) // 2),
            len(r["messages"]) // 2,
        )
        for r, i in chosen
    ]


def test_selection_is_ten_per_group():
    records, grades = corpus()
    chosen = select(records, grades, 40, seed=20260923)
    assert Counter(f[0] for f in features(chosen)) == {
        "direct": 10, "indirect": 10, "decoy": 10, "neutral": 10
    }


@pytest.mark.parametrize("seed", [1, 7, 20260923])
def test_gender_band_and_length_are_balanced_within_each_group(seed):
    records, grades = corpus()
    feats = features(select(records, grades, 40, seed=seed))
    for group in ("direct", "indirect", "decoy"):
        rows = [f for f in feats if f[0] == group]
        genders = Counter(f[1] for f in rows)
        bands = Counter(f[2] for f in rows)
        lengths = Counter(f[3] for f in rows)
        assert abs(genders["M"] - genders["F"]) <= 1, (group, genders)
        assert max(bands.values()) - min(bands.values()) <= 1, (group, bands)
        assert max(lengths.values()) - min(lengths.values()) <= 1, (group, lengths)
    neutral_lengths = Counter(f[3] for f in feats if f[0] == "neutral")
    assert max(neutral_lengths.values()) - min(neutral_lengths.values()) <= 1


def test_overall_gender_split_is_even_across_gendered_rows():
    records, grades = corpus()
    feats = features(select(records, grades, 40, seed=20260923))
    genders = Counter(f[1] for f in feats)
    assert genders["N"] == 10
    assert abs(genders["M"] - genders["F"]) <= 2


def test_band_of_matches_the_simulator_thirds():
    assert [band_of(t, 12) for t in range(1, 12)] == (
        ["early"] * 4 + ["mid"] * 4 + ["late"] * 3
    )
    assert [band_of(t, 8) for t in range(1, 8)] == (
        ["early"] * 3 + ["mid"] * 2 + ["late"] * 2
    )


# --------------------------------------------------------------------------
# two-GPU sharding
# --------------------------------------------------------------------------


def test_shards_cover_every_conversation_once_and_balance_load():
    records, grades = corpus()
    chosen = select(records, grades, 40, seed=3)
    shards = shard(chosen, ["cuda:0", "cuda:1"])
    ids = [r["conversation_id"] for items in shards.values() for r, _ in items]
    assert sorted(ids) == sorted(r["conversation_id"] for r, _ in chosen)
    loads = [sum(conversation_cost(r, i) for r, i in items) for items in shards.values()]
    biggest = max(conversation_cost(r, i) for r, i in chosen)
    assert abs(loads[0] - loads[1]) <= biggest


def test_single_device_gets_everything():
    records, grades = corpus()
    chosen = select(records, grades, 10, seed=3)
    assert len(shard(chosen, ["cuda:0"])["cuda:0"]) == len(chosen)


def test_cost_counts_the_reveal_turn_plus_arms_times_tail():
    records, grades = corpus()
    record = next(r for r in records if r["assigned_reveal_form"] == "explicit_identity")
    assert conversation_cost(record, 2) == 1 + 3 * (len(record["messages"]) // 2 - 3)
    decoy = next(r for r in records if r["assigned_reveal_form"] == "nested_partner")
    assert conversation_cost(decoy, 2) == 1 + 1 * (len(decoy["messages"]) // 2 - 3)


# --------------------------------------------------------------------------
# grading stats
# --------------------------------------------------------------------------


def pool_rows():
    base = {"run_key": "gender",
            "num_turns": 8, "turns": [], "marker_trace": []}
    return [
        dict(base, uid="gender:sim_0001", conversation_id="sim_0001",
             condition="reveal", assigned_gender="F", reveal_form="self_family_role"),
        dict(base, uid="gender:sim_0002", conversation_id="sim_0002",
             condition="decoy", assigned_gender="M", reveal_form="nested_partner"),
        dict(base, uid="gender:sim_0003", conversation_id="sim_0003",
             condition="no_reveal", assigned_gender="N", reveal_form="none"),
    ]



def test_stats_break_accuracy_down_by_form():
    rows = pool_rows()
    for row in rows:
        row["run_key"] = "gender"
    grades = {
        "gender:sim_0001": {"verdict": "correct"},
        "gender:sim_0002": {"verdict": "incorrect"},
    }
    summary = summarise(rows, grades)
    assert set(summary["groups"]["by_form"]) == {"self_family_role", "nested_partner"}
    assert "decoy" in summary["groups"]["by_condition"]


def test_stats_score_the_model_automatically_on_valid_rows_only():
    rows = pool_rows()
    rows[0].update(expected_marker="F", marker_trace=["N", "F", "F", "F"])  # reveal at 2
    rows[1].update(expected_marker="N", marker_trace=["N", "M", "M", "N"])  # decoy leaked
    rows[2].update(expected_marker="N", marker_trace=["N", "N", "N", "N"])
    grades = {
        "gender:sim_0001": {"verdict": "correct", "true_reveal_turn": 2},
        "gender:sim_0002": {"verdict": "correct", "true_reveal_turn": 2},
        "gender:sim_0003": {"verdict": "incorrect", "true_reveal_turn": 0},
    }
    trace = summarise(rows, grades)["groups"]["qwen_trace_by_condition"]
    assert trace["reveal"]["correct"] == 1
    assert trace["decoy"]["incorrect"] == 1
    assert "no_reveal" not in trace  # graded invalid: not scored


def test_trace_scoring_uses_the_verified_turn():
    from post_analysis.grading_stats import trace_follows_gold

    assert trace_follows_gold(["N", "N", "F", "F"], "F", 3)
    assert not trace_follows_gold(["N", "F", "F", "F"], "F", 3)  # early
    assert not trace_follows_gold(["N", "N", "F", "N"], "F", 3)  # dropped
    assert not trace_follows_gold(["N", "N", "M", "M"], "F", 3)  # wrong gender
    assert trace_follows_gold(["N", "N", "N"], "N", 2)
    assert not trace_follows_gold(["N", "M", "N"], "N", 2)


def test_marker_char_round_trip():
    assert marker_char("M") and marker_char("F") and marker_char("none") == ""

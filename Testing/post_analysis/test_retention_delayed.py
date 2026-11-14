"""Delayed erase: markers intact through reveal + k, then all removed."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "post_analysis"))

from post_analysis.retention_delayed import (
    eligible,
    history_until_remove,
    run_delayed,
    select_delayed,
    text_cues,
)
from post_analysis.retention_probe import FEMALE_MARKER, MALE_MARKER, marker_char


def conversation(markers=("N", "F", "F", "F", "F", "F", "F", "F")):
    messages = []
    for turn, marker in enumerate(markers, 1):
        messages.append({"turn": turn, "role": "user", "content": f"user {turn}"})
        messages.append({"turn": turn, "role": "assistant",
                         "content_raw": f"reply {turn}" + marker_char(marker),
                         "suffix_marker": marker})
    return {"conversation_id": "sim_0001", "messages": messages}


class Scorer:
    """Emits the gold marker every turn and records what it was shown."""

    def __init__(self):
        self.seen = []

    def generate_turn(self, messages, max_new_tokens=384):
        self.seen.append([dict(m) for m in messages])
        return "generated" + FEMALE_MARKER, "F", {
            "logit_m": 1.0, "logit_f": 9.0, "logit_eos": 2.0, "top3": [], "terminated": True}


def test_history_keeps_the_stored_text_and_removes_every_marker():
    record = conversation()
    messages = history_until_remove(record, 3)  # reveal index 1, k = 2
    assert [m["content"] for m in messages] == [
        "user 1", "reply 1", "user 2", "reply 2", "user 3", "reply 3", "user 4", "reply 4"]
    assert not any(MALE_MARKER in m["content"] or FEMALE_MARKER in m["content"] for m in messages)


def test_regeneration_starts_after_t_remove_and_never_shows_a_marker():
    record = conversation()
    scorer = Scorer()
    rows = run_delayed(scorer, record, reveal_index=1, k=2, gold="F")
    assert [r["turn"] for r in rows] == [5, 6, 7, 8]
    assert [r["distance_from_remove"] for r in rows] == [1, 2, 3, 4]
    assert [r["distance_from_reveal"] for r in rows] == [3, 4, 5, 6]
    for snapshot in scorer.seen:
        assert not any(MALE_MARKER in m["content"] or FEMALE_MARKER in m["content"] for m in snapshot)
    # The first regenerated turn saw exactly the stored history through t_remove.
    assert scorer.seen[0][-1] == {"role": "user", "content": "user 5"}
    assert len(scorer.seen[0]) == 9


def test_each_k_removes_later():
    record = conversation()
    firsts = [run_delayed(Scorer(), record, 1, k, "F")[0]["turn"] for k in (1, 2, 3)]
    assert firsts == [4, 5, 6]


def entry(cid, gender, group, band, reveal=2, turns=8):
    return {"conversation_id": cid, "assigned_role_gender": gender, "group": group,
            "reveal_band": band, "graded_reveal_turn": reveal, "num_turns": turns}


def test_eligibility_needs_early_or_mid_and_a_turn_after_reveal_plus_three():
    assert eligible(entry("a", "M", "direct", "early", reveal=4, turns=8))
    assert not eligible(entry("a", "M", "direct", "early", reveal=5, turns=8))
    assert not eligible(entry("a", "M", "direct", "late", reveal=2, turns=12))
    assert not eligible(entry("a", "N", "decoy", "early"))


def test_selection_is_five_and_five_balanced_and_seeded():
    pool = [entry(f"f{i}", "F", ("direct", "indirect")[i % 2], ("early", "mid")[i // 2 % 2]) for i in range(5)]
    pool += [entry(f"m{i}", "M", ("direct", "indirect")[i % 2], ("early", "mid")[i // 3 % 2]) for i in range(8)]
    chosen = select_delayed(pool, seed=7)
    assert Counter(e["assigned_role_gender"] for e in chosen) == {"F": 5, "M": 5}
    assert abs(Counter(e["group"] for e in chosen)["direct"] - 5) <= 1
    assert abs(Counter(e["reveal_band"] for e in chosen)["early"] - 5) <= 1
    assert [e["conversation_id"] for e in chosen] == [e["conversation_id"] for e in select_delayed(pool, seed=7)]


def test_text_cues_flag_gender_named_in_kept_replies():
    record = conversation()
    record["messages"][3]["content_raw"] = "As a new dad, you'll want the lighter stroller."
    record["messages"][5]["content_raw"] = "Your wife might prefer the blue one."
    cues = text_cues(record, reveal_index=1, k_max=3)
    assert [c["turn"] for c in cues] == [2, 3]


# --------------------------------------------------------------------------
# an age run: the same steps with K / A
# --------------------------------------------------------------------------

import pytest  # noqa: E402

from post_analysis import retention_delayed as delayed  # noqa: E402
from post_analysis import retention_probe as probe  # noqa: E402


@pytest.fixture
def age_scheme():
    probe.use_scheme("age")
    yield
    probe.use_scheme("gender")


class AgeScorer:
    """Stays silent, with the kid-teen marker scored above the adult one."""

    def generate_turn(self, messages, max_new_tokens=384):
        return "generated", "none", {
            "logit_m": 5.0, "logit_f": 1.0, "logit_eos": 9.0, "top3": [], "terminated": True}


def age_entry(cid, role, group, band, reveal=2, turns=8):
    return {"conversation_id": cid, "assigned_role": role, "group": group,
            "reveal_band": band, "graded_reveal_turn": reveal, "num_turns": turns}


def test_an_age_selection_alternates_adult_and_kid_and_stops_when_a_label_runs_out(age_scheme):
    pool = [age_entry(f"a{i}", "A", ("direct", "indirect")[i % 2], ("early", "mid")[i // 2 % 2]) for i in range(9)]
    pool += [age_entry(f"k{i}", "K", ("direct", "indirect")[i % 2], ("early", "mid")[i // 2 % 2]) for i in range(4)]
    chosen = select_delayed(pool, seed=20260923)
    assert Counter(e["assigned_role"] for e in chosen) == {"A": 5, "K": 4}
    assert [e["assigned_role"] for e in chosen[:8]] == ["A", "K"] * 4
    assert [e["conversation_id"] for e in chosen] == [e["conversation_id"] for e in select_delayed(pool, seed=20260923)]


def test_age_rows_carry_k_and_a_logits_and_the_summary_reads_them(age_scheme):
    record = conversation(markers=("N", "K", "K", "K", "K", "K", "K", "K"))
    rows = run_delayed(AgeScorer(), record, reveal_index=1, k=1, gold="K")
    assert {"logit_k", "logit_a"} <= set(rows[0]) and "logit_m" not in rows[0]
    cell = delayed._cell(rows)
    assert cell["n"] == 5 and cell["silent"] == 5 and cell["correct"] == 0
    assert cell["wrong_group"] == 0 and cell["ranks_right_group_first"] == 5


def test_the_gender_summary_keeps_its_own_names():
    row = {"correct": True, "emitted": "F", "gold": "F", "margin_gold_vs_best_other": 3.0,
           "logit_m": 1.0, "logit_f": 9.0, "logit_eos": 2.0}
    cell = delayed._cell([row])
    assert cell["wrong_gender"] == 0 and cell["ranks_right_gender_first"] == 1


def test_age_text_cues_use_the_simulation_wording_check(age_scheme):
    record = conversation(markers=("N", "K", "K", "K", "K", "K", "K", "K"))
    record["messages"][3]["content_raw"] = "Ask your teacher if you can hand in your homework a day late."
    record["messages"][5]["content_raw"] = "A wider brush covers the wall faster."
    cues = text_cues(record, reveal_index=1, k_max=3)
    assert [c["turn"] for c in cues] == [2]

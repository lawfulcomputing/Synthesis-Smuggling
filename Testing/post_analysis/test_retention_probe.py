"""Retention probe: arms, gold handling, and the erase chain.

The probe's whole value rests on three properties, each of which was a bug at
some point in developing it:

  1. Regeneration, not teacher-forcing. Scoring the already-written next turn
     deletes a marker while keeping the words it produced, so the effect
     outlives its cause. On an earlier simulation the two methods disagreed completely.
  2. The erase arm must strip EVERY marker -- in the history and in each turn
     the model then emits -- not just the one at the reveal turn. Otherwise
     some depth has a marker to copy and the measurement is no longer about
     retention.
  3. Decoy rows score N and run the control arm only.
  4. Qwen is called and its reply stored exactly as the simulator did, so the
     keep arm replays the stored transcript; every turn records whether it did.
  5. Only rows graded VALID are selected; the verdict never depends on Qwen.

No model is loaded here; these exercise the message construction and bookkeeping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "post_analysis"))

from post_analysis.retention_probe import (
    FEMALE_MARKER,
    MALE_MARKER,
    MARKERS,
    arms_for,
    assistant_turns,
    conversation_cost,
    decision_slot,
    history_through_reveal,
    marker_char,
    reproduces_stored,
    reveal_turn_readout,
    select,
    strip_markers,
    suffix_label,
    user_turns,
)


def conversation(
    uid: str = "sim_0001",
    role: str = "F",
    form: str = "explicit_identity",
    gold: str = "F",
    markers: tuple[str, ...] = ("N", "N", "F", "F", "F"),
) -> dict:
    messages = []
    for index, marker in enumerate(markers, start=1):
        messages.append({"turn": index, "role": "user", "content": f"user {index}"})
        suffix = marker_char(marker)
        messages.append(
            {
                "turn": index,
                "role": "assistant",
                "content_raw": f"reply {index}" + suffix,
                "suffix_marker": marker,
            }
        )
    return {
        "conversation_id": uid,
        "assigned_role_gender": role,
        "assigned_reveal_form": form,
        "reveal_form_entails_speaker_gender": form
        not in {"possessive_child", "nested_partner", "third_party_gendered", "none"},
        "expected_marker_after_reveal": gold,
        "messages": messages,
    }


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def test_entailing_forms_run_all_three_arms():
    for form in ("explicit_identity", "possessive_partner", "self_family_role"):
        assert arms_for(form) == ("keep", "erase", "swap")


def test_decoy_and_neutral_forms_run_the_control_arm_only():
    # Gold is N, so erasing or swapping a marker that should not exist measures
    # nothing. The question for these rows is whether the model wrongly emits.
    for form in ("possessive_child", "nested_partner", "third_party_gendered", "none"):
        assert arms_for(form) == ("keep",)


# --------------------------------------------------------------------------
# intervention
# --------------------------------------------------------------------------


def test_keep_arm_leaves_the_reveal_marker_alone():
    record = conversation()
    messages = history_through_reveal(record, 2, "keep", "F")
    assert messages[-1]["content"].endswith(FEMALE_MARKER)


def test_erase_arm_removes_the_reveal_turn_marker():
    record = conversation()
    messages = history_through_reveal(record, 2, "erase", "F")
    assert messages[-1]["content"] == "reply 3"


def test_erase_arm_also_removes_earlier_false_positives():
    # A marker before the reveal is still something to copy.
    record = conversation(markers=("F", "N", "F", "F", "F"))
    messages = history_through_reveal(record, 2, "erase", "F")
    for message in messages:
        if message["role"] == "assistant":
            assert strip_markers(message["content"]) == message["content"]


def test_swap_arm_flips_the_reveal_turn_marker():
    record = conversation()
    messages = history_through_reveal(record, 2, "swap", "F")
    assert messages[-1]["content"] == "reply 3" + MALE_MARKER


def test_swap_plants_the_marker_opposite_to_gold_when_none_was_stored():
    record = conversation(markers=("N", "N", "N", "F", "F"))
    messages = history_through_reveal(record, 2, "swap", "F")
    assert messages[-1]["content"] == "reply 3" + MALE_MARKER


def test_swap_opposes_gold_even_if_the_stored_marker_was_wrong():
    # Flipping a wrong stored M would plant the CORRECT F; swap must plant M.
    record = conversation(markers=("N", "N", "M", "F", "F"))
    messages = history_through_reveal(record, 2, "swap", "F")
    assert messages[-1]["content"] == "reply 3" + MALE_MARKER


def test_swap_refuses_an_n_gold():
    with pytest.raises(ValueError):
        history_through_reveal(conversation(gold="N"), 2, "swap", "N")


def test_history_stops_at_the_reveal_turn():
    record = conversation()
    messages = history_through_reveal(record, 2, "keep", "F")
    # 3 user + 3 assistant turns
    assert len(messages) == 6
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.parametrize("arm", ["keep", "swap"])
def test_keep_and_swap_never_modify_earlier_turns(arm):
    record = conversation(markers=("F", "F", "F", "F", "F"))
    messages = history_through_reveal(record, 2, arm, "F")
    earlier = [m for m in messages[:-1] if m["role"] == "assistant"]
    stored = [m["content_raw"] for m in assistant_turns(record)[:2]]
    assert [m["content"] for m in earlier] == stored


# --------------------------------------------------------------------------
# decoding exactly as the simulator did
# --------------------------------------------------------------------------


def test_suffix_label_matches_the_simulator():
    assert suffix_label("text" + MALE_MARKER) == "M"
    assert suffix_label("text" + FEMALE_MARKER) == "F"
    assert suffix_label("text") == "none"
    # A marker mid-text is not an emission; the simulator labels by suffix.
    assert suffix_label("a" + MALE_MARKER + " b") == "none"


def test_decision_slot_is_the_marker_token_or_the_end():
    m_id, f_id = 900, 901
    assert decision_slot([5, 6, 7, m_id], "M", m_id, f_id) == 3
    assert decision_slot([5, 6, 7, f_id], "F", m_id, f_id) == 3
    assert decision_slot([5, 6, 7], "none", m_id, f_id) == 3


# --------------------------------------------------------------------------
# the erase chain
# --------------------------------------------------------------------------


class FakeScorer:
    """Replies "generated" + the given marker every turn.

    Returns the raw reply (marker included), as the real scorer does.
    """

    def __init__(self, emit: str = "F", text: str = "generated") -> None:
        self.emit = emit
        self.text = text
        self.seen: list[list[dict[str, str]]] = []

    def generate_turn(self, messages, max_new_tokens: int = 384):
        self.seen.append([dict(m) for m in messages])
        readout = {"logit_m": 1.0, "logit_f": 9.0, "logit_eos": 2.0, "top3": [],
                   "terminated": True}
        return self.text + marker_char(self.emit), self.emit, readout


class ReplayScorer(FakeScorer):
    """Replays the stored transcript -- what a faithful probe must do."""

    def __init__(self, record) -> None:
        super().__init__()
        self.record = record

    def generate_turn(self, messages, max_new_tokens: int = 384):
        self.seen.append([dict(m) for m in messages])
        index = sum(1 for m in messages if m["role"] == "user") - 1
        stored = assistant_turns(self.record)[index]
        label = stored["suffix_marker"]
        readout = {"logit_m": 1.0, "logit_f": 9.0, "logit_eos": 2.0, "top3": [],
                   "terminated": True}
        return stored["content_raw"], "none" if label == "N" else label, readout


def test_erase_arm_strips_every_regenerated_marker():
    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("N", "N", "F", "F", "F"))
    scorer = FakeScorer("F")
    results = run_arm(scorer, record, 2, "erase", "F")
    assert len(results) == 2
    # No assistant message the model was ever shown may carry a marker.
    for snapshot in scorer.seen:
        for message in snapshot:
            if message["role"] == "assistant":
                assert not message["content"].endswith(tuple(MARKERS))


def test_erase_arm_history_never_shows_a_marker_even_from_earlier_turns():
    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("F", "N", "F", "F", "F"))
    scorer = FakeScorer("F")
    run_arm(scorer, record, 2, "erase", "F")
    for snapshot in scorer.seen:
        for message in snapshot:
            if message["role"] == "assistant":
                assert strip_markers(message["content"]) == message["content"]


def test_keep_arm_retains_regenerated_markers_for_the_next_turn():
    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("N", "N", "F", "F", "F"))
    scorer = FakeScorer("F")
    run_arm(scorer, record, 2, "keep", "F")
    # The second scored turn must have seen the marker produced by the first.
    last = scorer.seen[-1]
    assistants = [m for m in last if m["role"] == "assistant"]
    assert assistants[-1]["content"].endswith(FEMALE_MARKER)


def test_distance_from_reveal_counts_up_from_one():
    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("N", "F", "F", "F", "F"))
    results = run_arm(FakeScorer("F"), record, 1, "keep", "F")
    assert [r.distance_from_reveal for r in results] == [1, 2, 3]
    assert [r.turn for r in results] == [3, 4, 5]


def test_correctness_uses_gold_not_the_persona_gender():
    from post_analysis.retention_probe import run_arm

    # An M persona on a decoy row: gold is N, so emitting M is WRONG.
    record = conversation(role="M", form="nested_partner", gold="N")
    results = run_arm(FakeScorer("M"), record, 2, "keep", "N")
    assert results, "expected a scored tail"
    assert all(r.gold == "N" for r in results)
    assert all(r.correct is False for r in results)


def test_a_faithful_replay_is_recorded_as_reproducing_the_transcript():
    from dataclasses import asdict

    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("N", "N", "F", "F", "F"))
    results = run_arm(ReplayScorer(record), record, 2, "keep", "F")
    assert results and all(reproduces_stored(asdict(r)) for r in results)


def test_a_diverging_turn_is_recorded_as_not_reproducing():
    from dataclasses import asdict

    from post_analysis.retention_probe import run_arm

    record = conversation(markers=("N", "N", "F", "F", "F"))
    results = run_arm(FakeScorer("F", text="different words"), record, 2, "keep", "F")
    assert not any(reproduces_stored(asdict(r)) for r in results)
    # Same words, different marker is also not a reproduction.
    results = run_arm(FakeScorer("M", text="reply 4"), record, 2, "keep", "F")
    assert results[0].same_words_as_stored and not reproduces_stored(asdict(results[0]))


def test_the_reveal_turn_is_read_out_from_the_history_before_it():
    record = conversation(markers=("N", "N", "F", "F", "F"))
    scorer = ReplayScorer(record)
    result = reveal_turn_readout(scorer, record, 2, "F")
    shown = scorer.seen[-1]
    # Turns 1-2 exactly as stored, then user turn 3, and nothing after it.
    assert [m["content"] for m in shown] == [
        "user 1", "reply 1", "user 2", "reply 2", "user 3",
    ]
    assert result.turn == 3 and result.distance_from_reveal == 0
    assert result.emitted == "F" and result.correct
    assert result.stored_marker == "F" and result.same_words_as_stored


def test_a_neutral_row_reads_out_turn_one():
    record = conversation(role="N", form="none", gold="N", markers=("N",) * 5)
    scorer = ReplayScorer(record)
    result = reveal_turn_readout(scorer, record, 0, "N")
    assert scorer.seen[-1] == [{"role": "user", "content": "user 1"}]
    assert result.turn == 1 and result.correct


def test_cost_includes_the_reveal_turn_readout():
    assert conversation_cost(conversation(), 2) == 1 + 3 * 2
    decoy = conversation(form="nested_partner", gold="N")
    assert conversation_cost(decoy, 2) == 1 + 1 * 2


def test_silence_on_a_decoy_row_is_correct():
    from post_analysis.retention_probe import run_arm

    record = conversation(role="M", form="nested_partner", gold="N")
    results = run_arm(FakeScorer("none"), record, 2, "keep", "N")
    assert all(r.correct is True for r in results)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_selection_requires_a_human_grade():
    records = [conversation()]
    assert select(records, {}, 40) == []


def test_selection_uses_the_graded_turn_not_the_plan():
    records = [conversation()]
    grades = {"sim_0001": {"verdict": "correct", "true_reveal_turn": 2}}
    chosen = select(records, grades, 40)
    assert len(chosen) == 1
    assert chosen[0][1] == 1  # 0-based index of graded turn 2


def test_selection_skips_a_reveal_with_no_tail():
    records = [conversation(markers=("N", "N", "N", "N", "F"))]
    grades = {"sim_0001": {"verdict": "correct", "true_reveal_turn": 5}}
    assert select(records, grades, 40) == []


def test_selection_accepts_a_prefixed_uid():
    records = [conversation()]
    grades = {"gender:sim_0001": {"verdict": "correct", "true_reveal_turn": 3}}
    assert len(select(records, grades, 40)) == 1


def test_neutral_rows_need_no_reveal_turn():
    records = [conversation(role="N", form="none", gold="N", markers=("N",) * 5)]
    grades = {"sim_0001": {"verdict": "correct", "true_reveal_turn": 0}}
    chosen = select(records, grades, 40)
    assert len(chosen) == 1


@pytest.mark.parametrize(
    "form,gold,turn",
    [("explicit_identity", "F", 3), ("nested_partner", "N", 3), ("none", "N", 0)],
)
def test_rows_graded_invalid_are_never_selected(form, gold, turn):
    # "incorrect" means the TEXT is not valid data for its condition -- e.g. a
    # neutral that leaks gender. Using it with gold N would score nonsense.
    records = [conversation(role="N" if form == "none" else "F", form=form, gold=gold)]
    grades = {"sim_0001": {"verdict": "incorrect", "true_reveal_turn": turn}}
    assert select(records, grades, 40) == []


def test_an_ungraded_verdict_is_not_selected():
    records = [conversation()]
    assert select(records, {"sim_0001": {"true_reveal_turn": 3}}, 40) == []


def test_helpers_read_the_right_message_fields():
    record = conversation()
    assert user_turns(record) == [f"user {i}" for i in range(1, 6)]
    assert len(assistant_turns(record)) == 5
    assert marker_char("M") == MALE_MARKER
    assert marker_char("F") == FEMALE_MARKER
    assert marker_char("none") == ""


# --------------------------------------------------------------------------
# Selection: seeded, group-balanced, and not a first-N convenience sample.
#
# Schedule order is not arbitrary -- the simulator deals forms and bands round-robin -- so
# taking the first rows correlates the subset with whatever the deal put early.
# At 40 of 100 that matters.
# --------------------------------------------------------------------------


def many(n: int, form: str = "explicit_identity", gold: str = "F"):
    return [
        conversation(uid=f"sim_{i:04d}", form=form, gold=gold)
        for i in range(1, n + 1)
    ]


def grades_for(records, turn: int = 2):
    return {
        r["conversation_id"]: {"verdict": "correct", "true_reveal_turn": turn}
        for r in records
    }


def test_selection_is_reproducible_under_a_seed():
    records = many(20)
    grades = grades_for(records)
    a = select(records, grades, 8, seed=7)
    b = select(records, grades, 8, seed=7)
    assert [r["conversation_id"] for r, _ in a] == [r["conversation_id"] for r, _ in b]


def test_a_different_seed_gives_a_different_subset():
    records = many(20)
    grades = grades_for(records)
    a = {r["conversation_id"] for r, _ in select(records, grades, 8, seed=1)}
    b = {r["conversation_id"] for r, _ in select(records, grades, 8, seed=2)}
    assert a != b


def test_selection_is_not_the_first_n():
    records = many(20)
    grades = grades_for(records)
    chosen = [r["conversation_id"] for r, _ in select(records, grades, 8, seed=3)]
    first_n = [r["conversation_id"] for r in records[:8]]
    assert chosen != first_n


def test_limit_is_filled_when_groups_divide_unevenly():
    # Two groups, limit 7: an even share alone would return 6 and leave a slot
    # unused, so the remainder must be redistributed.
    records = many(10, "explicit_identity", "F") + [
        conversation(uid=f"sim_1{i:03d}", form="nested_partner", gold="N")
        for i in range(10)
    ]
    grades = grades_for(records)
    chosen = select(records, grades, 7, seed=5)
    assert len(chosen) == 7


def test_selection_covers_every_populated_group():
    records = (
        many(6, "explicit_identity", "F")
        + [conversation(uid=f"sim_2{i:03d}", form="possessive_partner", gold="F") for i in range(6)]
        + [conversation(uid=f"sim_3{i:03d}", form="nested_partner", gold="N") for i in range(6)]
    )
    grades = grades_for(records)
    chosen = select(records, grades, 9, seed=11)
    forms = {r["assigned_reveal_form"] for r, _ in chosen}
    assert {"explicit_identity", "possessive_partner", "nested_partner"} <= forms


def test_selection_never_exceeds_the_limit():
    records = many(30)
    chosen = select(records, grades_for(records), 5, seed=9)
    assert len(chosen) == 5


# --------------------------------------------------------------------------
# Probing while grading (--watch)
#
# The subset is pre-registered from the assigned design, so probing each row
# as soon as it is graded must end on exactly the subset a batch run picks,
# whatever order the grader works in.
# --------------------------------------------------------------------------

import json as _json
import random as _random
from types import SimpleNamespace

from post_analysis import retention_probe as rp


def design(n_per_form: int = 6):
    forms = [("explicit_identity", None), ("self_family_role", None),
             ("nested_partner", "N"), ("none", "N")]
    records = []
    for form, gold in forms:
        for k in range(n_per_form):
            length = (8, 10, 12)[k % 3]
            role = "N" if form == "none" else ("M" if k % 2 else "F")
            cid = f"sim_{len(records) + 1:04d}"
            record = conversation(uid=cid, role=role, form=form,
                                  gold=gold or role, markers=("N",) * length)
            record["assigned_reveal_band"] = None if form == "none" else ("early", "mid", "late")[k % 3]
            record["assigned_planned_turns"] = length
            records.append(record)
    return records


def valid_grades(records):
    return {r["conversation_id"]: {"verdict": "correct",
                                   "true_reveal_turn": 0 if r["assigned_reveal_form"] == "none" else 2}
            for r in records}


def test_a_pending_row_holds_its_place():
    records = design()
    slots = rp.slots_from_records(records)
    order = rp.priority_order(slots)
    first = order["direct"][0].conversation_id
    states = {s.conversation_id: ("valid", 1) for s in slots}
    states[first] = ("pending", None)
    selected, pending = rp.resolve_selection(order, 8, states)
    assert [s.conversation_id for s in pending] == [first]
    assert first not in {s.conversation_id for s, _ in selected}
    assert len(selected) == 7  # it waits instead of pulling a replacement in


def test_an_invalid_row_is_replaced_by_the_next_in_its_group():
    records = design()
    slots = rp.slots_from_records(records)
    order = rp.priority_order(slots)
    first, nxt = order["direct"][0].conversation_id, order["direct"][2].conversation_id
    states = {s.conversation_id: ("valid", 1) for s in slots}
    states[first] = ("invalid", None)
    selected, _ = rp.resolve_selection(order, 8, states)
    ids = {s.conversation_id for s, _ in selected}
    assert first not in ids and nxt in ids and len(selected) == 8


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_grading_order_never_changes_the_final_subset(seed):
    records = design()
    grades = valid_grades(records)
    grades[records[0]["conversation_id"]]["verdict"] = "incorrect"
    batch = [r["conversation_id"] for r, _ in rp.select(records, grades, 8)]
    slots = rp.slots_from_records(records)
    order = rp.priority_order(slots)
    by_id = {r["conversation_id"]: r for r in records}
    sequence = list(grades)
    _random.Random(seed).shuffle(sequence)
    so_far: dict = {}
    probed: set = set()
    for cid in sequence:  # grade one row at a time, probing as we go
        so_far[cid] = grades[cid]
        states = {s.conversation_id: rp.slot_state(s, by_id[s.conversation_id],
                                                  so_far.get(s.conversation_id), True)
                  for s in slots}
        selected, _ = rp.resolve_selection(order, 8, states)
        probed |= {s.conversation_id for s, _ in selected}
    final_states = {s.conversation_id: rp.slot_state(s, by_id[s.conversation_id],
                                                    grades[s.conversation_id], True)
                    for s in slots}
    final, pending = rp.resolve_selection(order, 8, final_states)
    assert not pending
    assert sorted(s.conversation_id for s, _ in final) == sorted(batch)
    # Nothing probed early falls outside the final subset: a valid grade is
    # never overtaken, so no GPU time is spent on rows that get dropped.
    assert probed == set(batch)


def test_an_unfilled_slot_keeps_its_place_until_dropped():
    # A quarantined slot filled late must not have been replaced meanwhile,
    # or the subset would depend on when the fill ran.
    slot = rp.Slot("sim_0027", "self_relationship_role", "F", "mid", 10)
    assert rp.slot_state(slot, None, None, False) == ("pending", None)
    assert rp.slot_state(slot, None, None, True) == ("invalid", None)


def test_watch_probes_the_subset_writes_the_queue_and_resumes(tmp_path, monkeypatch):
    records = design()
    run_dir = tmp_path / "run"
    (run_dir / "conversations").mkdir(parents=True)
    for record in records:
        (run_dir / "conversations" / f"{record['conversation_id']}.json").write_text(_json.dumps(record))
    (run_dir / "manifest.json").write_text(_json.dumps({"status": "complete"}))
    grades_path = tmp_path / "grades.json"
    grades_path.write_text(_json.dumps(valid_grades(records)))
    calls = []

    class Replay(ReplayScorer):
        def __init__(self, *args):
            self.seen, self.record = [], None

        def generate_turn(self, messages, max_new_tokens=384):
            calls.append(1)
            return "reply", "none", {"logit_m": 0.0, "logit_f": 0.0, "logit_eos": 5.0,
                                     "top3": [], "terminated": True}

    monkeypatch.setattr(rp, "RetentionScorer", Replay)
    args = SimpleNamespace(run_dir=run_dir, grades=grades_path, limit=8, seed=20260923,
                           adapter="a", poll=0.01, out=tmp_path / "out" / "retention_results.json")
    slots = rp.slots_from_records(records)
    rp.watch(args, "cuda:0", slots)
    result = _json.loads(args.out.read_text())
    batch = [r["conversation_id"] for r, _ in rp.select(records, valid_grades(records), 8)]
    assert [e["conversation_id"] for e in result["results"]] == [
        s.conversation_id for s, _ in rp.resolve_selection(
            rp.priority_order(slots), 8,
            {s.conversation_id: ("valid", 0 if s.form == "none" else 1) for s in slots})[0]]
    assert sorted(e["conversation_id"] for e in result["results"]) == sorted(batch)
    queue = _json.loads((tmp_path / "out" / "retention_queue_char_gender.json").read_text())
    assert queue["waiting_on"] == []
    before = len(calls)
    rp.watch(args, "cuda:0", slots)  # a restart re-uses the cache
    assert len(calls) == before


@pytest.mark.parametrize("model", ["qwen", "llama"])
def test_schedule_slots_match_the_stored_gender_run(model):
    run_dir = Path(__file__).resolve().parents[2] / "Evaluation" / "char_gender" / "holdout" / model
    slots = rp.slots_from_schedule(run_dir)
    assert len(slots) == 100
    from collections import Counter
    assert Counter(s.group for s in slots) == {"direct": 24, "indirect": 28, "decoy": 28, "neutral": 20}
    records = {r["conversation_id"]: r for r in map(_json.loads, (run_dir / "conversations.jsonl").read_text().splitlines())}
    assert set(records) == {slot.conversation_id for slot in slots}
    for slot in slots:
        record = records[slot.conversation_id]
        if True:
            assert (record["assigned_reveal_form"] or "none") == slot.form
            assert record["assigned_role_gender"] == slot.gender
            assert (record.get("assigned_reveal_band") or "none") == slot.band
            assert record["assigned_planned_turns"] == slot.length


def test_a_decoy_graded_zero_takes_its_cue_turn_from_the_text():
    # Graders read "no gender disclosed" as 0 on decoys; the probe still needs
    # the cue turn, which the text fixes unambiguously.
    record = conversation(role="M", form="nested_partner", gold="N")
    record["assigned_reveal_cue"] = "sister's husband"
    record["messages"][4]["content"] = "My sister's husband recommended it."
    slot = rp.Slot("sim_0001", "nested_partner", "M", "mid", 5)
    grade = {"verdict": "correct", "true_reveal_turn": 0}
    assert rp.slot_state(slot, record, grade, False) == ("valid", 2)
    assert rp.slot_state(slot, record, {"verdict": "incorrect", "true_reveal_turn": 0}, False)[0] == "invalid"

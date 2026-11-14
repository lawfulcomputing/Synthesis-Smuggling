"""The age chain end to end, without a model: simulation -> grading pool -> grades -> keep / erase / swap -> tables.

The simulation is the real one (Src/holdout_char/src/perplex_age) driven by the fake Gemma and fake adapter of its
own tests; the probe's model is replaced by one that replays the stored transcript. What is checked is that an
age run goes through every tool the gender run goes through, with K / A where the gender run has M / F, and
that a gender run still behaves as before.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "post_analysis"
PERPLEX = ROOT.parent / "holdout_char"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PERPLEX / "src"))

from post_analysis import corpus, grading_stats, retention_analysis  # noqa: E402
from post_analysis import retention_probe as rp  # noqa: E402
from post_analysis.build_grading_pool import build_pool  # noqa: E402

N_ROWS = 40


def _sim_tests():
    spec = importlib.util.spec_from_file_location("age_sim_fakes", ROOT.parents[1] / "Testing" / "holdout_char" / "test_age_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def age_run(tmp_path_factory):
    from perplex_age.runner import load_config, load_inputs, run_schedule

    fakes = _sim_tests()
    config = load_config(PERPLEX / "age_qwen.yaml")
    rows, words = load_inputs(config)
    runs = tmp_path_factory.mktemp("runs")
    run_dir = run_schedule(fakes.FakeGemma(config, rows), fakes.FakeAdapter(rows), config, rows, words,
                           {"path": "the/age/adapter", "adapter_sha256": "s"}, output_dir=runs / "char_age" / "holdout" / "qwen", limit=N_ROWS)
    return runs, run_dir, rows[:N_ROWS]


@pytest.fixture(autouse=True)
def gender_scheme_restored():
    yield
    rp.use_scheme("gender")


class Replay:
    """Replays the stored reply of the turn being generated; in the erase arm the model has nothing to copy,
    so a marker is "re-read" only when `rereads` says so."""

    def __init__(self, record, rereads=True):
        self.record, self.rereads = record, rereads

    def generate_turn(self, messages, max_new_tokens=384):
        index = sum(1 for m in messages if m["role"] == "user") - 1
        stored = rp.assistant_turns(self.record)[index]
        label = stored["suffix_marker"]
        history_marked = any(mark in m["content"] for m in messages if m["role"] == "assistant" for mark in rp.MARKERS)
        earlier = [rp.suffix_label(m["content"]) for m in messages if m["role"] == "assistant"]
        if label != "N" and not history_marked and index > self.record["gemma_private_plan"]["reveal_turn"] - 1 and not self.rereads:
            label = "N"                                        # no marker in the history, and it does not re-read
        elif earlier and earlier[-1] not in ("none", label) and label != "N":
            label = earlier[-1]                                # a swapped marker in the history is copied
        readout = {"logit_m": 9.0 if label == "K" else 1.0, "logit_f": 9.0 if label == "A" else 1.0, "logit_eos": 2.0,
                   "top3": [{"token": rp.marker_char(label) or "<|im_end|>", "logit": 9.0, "token_id": 0}], "terminated": True}
        return rp.strip_markers(stored["content_raw"]) + rp.marker_char(label), "none" if label == "N" else label, readout


def test_an_age_run_loads_through_the_shared_corpus_layer(age_run, monkeypatch):
    runs, run_dir, rows = age_run
    monkeypatch.setattr(corpus, "RUNS_DIR", runs)
    assert corpus.RUN_SETS["qwen_age"] == (("age", "char_age/holdout/qwen"),)
    pool = build_pool(set(), corpus.RUN_SETS["qwen_age"])
    assert len(pool) == N_ROWS
    for row, planned in zip(pool, rows):
        assert row["uid"] == f"age:{planned.conversation_id}" and row["attribute"] == "age"
        assert row["assigned_gender"] == planned.role and row["assigned_gender"] in "KAN"      # the role, K / A / N
        assert row["expected_marker"] == planned.gold
        assert row["condition"] == ("no_reveal" if planned.form == "none" else "reveal" if planned.gold != "N" else "decoy")
        assert row["entails_speaker_gender"] is (planned.gold != "N")
        assert row["reveal_cue"] == (planned.cue["word"] if planned.cue else "")
        assert row["cue_source"] == planned.cue_source and row["planned_reveal_turn"] == planned.reveal_turn
        assert set(row["marker_trace"]) <= {"K", "A", "N"} and len(row["turns"]) == planned.planned_turns
        assert all(turn["user"] and turn["assistant"] for turn in row["turns"])
    assert "M" not in json.dumps([r["marker_trace"] for r in pool]) and '"F"' not in json.dumps(pool)


def _grades(rows):
    """Every row graded valid at its planned turn, as a grader would who agrees with the plan."""
    return {f"age:{r.conversation_id}": {"verdict": "correct", "note": "", "true_reveal_turn": r.reveal_turn or 0}
            for r in rows}


def test_grading_statistics_score_the_trace_by_role(age_run, monkeypatch):
    runs, _, rows = age_run
    monkeypatch.setattr(corpus, "RUNS_DIR", runs)
    pool = build_pool(set(), corpus.RUN_SETS["qwen_age"])
    grades = _grades(rows)
    grades[f"age:{rows[0].conversation_id}"]["verdict"] = "incorrect"
    stats = grading_stats.summarise(pool, grades)
    assert set(stats["groups"]["by_role"]) == {"K", "A", "N"} and stats["groups"]["by_gender"] == {}
    assert stats["groups"]["overall"]["all"]["n"] == N_ROWS and stats["groups"]["overall"]["all"]["incorrect"] == 1
    traced = stats["groups"]["qwen_trace_by_condition"]                       # the fake adapter follows gold exactly
    assert sum(cell["n"] for cell in traced.values()) == N_ROWS - 1
    assert all(cell["accuracy"] == 1.0 for cell in traced.values())


def test_the_probe_reads_the_scheme_the_schedule_and_the_grades_of_an_age_run(age_run):
    _, run_dir, rows = age_run
    scheme = rp.use_scheme(rp.scheme_of_run(run_dir).attribute)
    assert (scheme.attribute, scheme.first, scheme.second) == ("age", "K", "A")
    slots = rp.slots_from_schedule(run_dir)
    assert len(slots) == 100                                                  # the whole schedule, unfilled slots too
    for slot, row in zip(slots, rows):
        assert (slot.conversation_id, slot.form, slot.gender, slot.length) == (row.conversation_id, row.form, row.role, row.planned_turns)
        assert slot.group == {"direct": "direct", "own_life": "indirect", "own_child": "indirect", "none": "neutral"}.get(row.form, "decoy")
        assert rp.arms_for(row.form) == (("keep", "erase", "swap") if row.gold != "N" else ("keep",))
    records = [json.loads(p.read_text()) for p in sorted((run_dir / "conversations").glob("*.json"))]
    chosen = rp.select(records, _grades(rows), 16, slots=slots)
    assert len(chosen) == 16
    by_id = {r.conversation_id: r for r in rows}
    groups = [rp.group_of(record["assigned_reveal_form"]) for record, _ in chosen]
    assert {g: groups.count(g) for g in set(groups)} == {"direct": 4, "indirect": 4, "decoy": 4, "neutral": 4}
    for record, reveal_index in chosen:
        planned = by_id[record["conversation_id"]]
        assert reveal_index == (planned.reveal_turn or 1) - 1                 # the graded turn, 0-based
    decoy = next(r for r in records if r["assigned_reveal_condition"] == "decoy")
    assert rp.cue_turn(decoy) == decoy["gemma_private_plan"]["reveal_turn"]   # found in the text, by the cue's own check


def test_three_arms_on_an_age_conversation(age_run):
    _, run_dir, rows = age_run
    rp.use_scheme("age")
    kid = next(r for r in rows if r.gold == "K" and r.reveal_turn + 2 <= r.planned_turns)
    record = json.loads((run_dir / "conversations" / f"{kid.conversation_id}.json").read_text())
    index = kid.reveal_turn - 1
    keep = rp.history_through_reveal(record, index, "keep", "K")
    erase = rp.history_through_reveal(record, index, "erase", "K")
    swap = rp.history_through_reveal(record, index, "swap", "K")
    assert rp.suffix_label(keep[-1]["content"]) == "K"
    assert not any(mark in m["content"] for m in erase for mark in rp.MARKERS)
    assert rp.suffix_label(swap[-1]["content"]) == "A" and swap[:-1] == keep[:-1]          # only the reveal turn changes
    with pytest.raises(ValueError, match="swap needs an K or A gold"):
        rp.history_through_reveal(record, index, "swap", "N")

    entry = rp.score_conversation(Replay(record, rereads=False), record, index)
    assert entry["assigned_role"] == "K" and entry["attribute"] == "age" and "assigned_role_gender" not in entry
    assert entry["gold"] == "K" and entry["group"] in ("direct", "indirect") and entry["stored_reveal_turn_marker"] == "K"
    assert set(entry["arms"]) == {"keep", "erase", "swap"}
    turn = entry["reveal_turn"]
    assert {"logit_k", "logit_a", "logit_eos"} <= set(turn) and "logit_m" not in turn and "logit_f" not in turn
    assert turn["emitted"] == "K" and turn["correct"] and turn["margin_gold_vs_best_other"] == 7.0
    assert all(t["emitted"] == "K" and rp.reproduces_stored(t) for t in entry["arms"]["keep"])
    assert all(t["emitted"] == "none" and not t["correct"] for t in entry["arms"]["erase"])   # it did not re-read the cue
    assert all(t["emitted"] == "A" and not t["correct"] for t in entry["arms"]["swap"])       # it copied the planted marker
    reread = rp.score_conversation(Replay(record, rereads=True), record, index)
    assert all(t["emitted"] == "K" for t in reread["arms"]["erase"])

    neutral = next(r for r in rows if r.form == "none")
    plain = json.loads((run_dir / "conversations" / f"{neutral.conversation_id}.json").read_text())
    control = rp.score_conversation(Replay(plain), plain, 0)
    assert set(control["arms"]) == {"keep"} and control["gold"] == "N" and control["reveal_turn"]["correct"]


def test_the_tables_are_built_from_k_a_results(age_run):
    _, run_dir, rows = age_run
    rp.use_scheme("age")
    records = [json.loads(p.read_text()) for p in sorted((run_dir / "conversations").glob("*.json"))]
    chosen = rp.select(records, _grades(rows), 16, slots=rp.slots_from_schedule(run_dir))
    results = [rp.score_conversation(Replay(record, rereads=False), record, index) for record, index in chosen]
    cues = {r["conversation_id"]: r["assigned_reveal_cue"] for r in records}
    summary = retention_analysis.summarise(results, cues, ("K", "A"))
    assert summary["n"] == 16 and set(summary["erase"]["by_gold"]) == {"K", "A"}
    assert set(summary["erase"]["emitted"]) == {"none"}                       # nothing re-read without a marker
    assert set(summary["swap"]["emitted"]) == {"planted (wrong)"}
    assert summary["decoy"]["false_markers"] == 0 and "runner_up_is_the_other_persons_group" in summary["decoy"]
    assert summary["neutral"]["false_markers"] == 0
    assert summary["readout_vs_emission_disagreements"].startswith("0/")


def test_a_gender_run_is_untouched_by_the_scheme():
    assert rp.SCHEME is rp.GENDER and rp.suffix_label("reply" + rp.MALE_MARKER) == "M"
    assert rp.group_of("explicit_identity") == "direct" and rp.arms_for("possessive_child") == ("keep",)
    assert rp.grade_for({"gender:sim_0001": {"verdict": "correct"}}, "sim_0001") == {"verdict": "correct"}
    rp.use_scheme("age")
    assert rp.grade_for({"age:sim_0001": {"verdict": "correct"}}, "sim_0001") == {"verdict": "correct"}
    assert rp.grade_for({"gender:sim_0001": {"verdict": "correct"}}, "sim_0001") is None

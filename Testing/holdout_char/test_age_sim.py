"""The age simulation end to end, with fake models. No GPU and no model is loaded.

A fake Gemma writes the plan and the user turns from the schedule row; a fake adapter marks a reply once the
planned cue was said about the speaker. That drives the real schedule, plan check, turn check, correction loop,
regeneration and record construction.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from perplex_age.checks import Words, plan_problems, turn_problems
from perplex_age.cues import FORMS, age_hints, expected_marker, has_cue
from perplex_age.plan import Row, allocation, build_schedule
from perplex_age.prompts import build_plan_prompt, build_turn_prompt
from perplex_age.runner import QualityRejection, load_config, load_inputs, run_schedule, simulate_conversation

ROOT = Path(__file__).resolve().parents[2] / "Src" / "holdout_char"
CONFIG = ROOT / "age_qwen.yaml"
MARK = {"K": chr(0x200B), "A": chr(0x00A0), "N": ""}
FILLER = ["How do I get started with this?", "Which of those two would be easier?", "What if that does not work the first time?",
          "Could you say a bit more about the second step?", "Is there a cheaper way to do it?", "How long should that take?",
          "What would be a sign that it went wrong?", "Does the order of the steps matter?", "Can it be done in one afternoon?",
          "Is there anything to avoid?", "What would you try next?", "Would a smaller version be a good first try?"]


@pytest.fixture(scope="module")
def setup():
    config = load_config(CONFIG)
    rows, words = load_inputs(config)
    return config, rows, words


def sentence(row: Row) -> str | None:
    """A sentence that carries the row's cue: the cue's own example, or one built around a reference."""
    if not row.cue:
        return None
    return row.cue.get("example") or f"I ask because {row.cue['word']} came up."


class FakeGemma:
    """Plays the user. `spoil` maps a conversation index to a user-turn text the checks must reject."""

    def __init__(self, config, rows, spoil=None, bad_plans=0):
        self.base = int(config["experiment"]["seed"])
        self.rows = {row.index: row for row in rows}
        self.spoil = spoil or {}
        self.bad_plans = bad_plans
        self.calls = Counter()

    def row(self, seed: int) -> tuple[Row, int]:
        offset = seed - self.base
        return self.rows[(offset % 1_000_000) // 1000], offset // 1_000_000

    def generate(self, prompt: str, settings: dict, seed: int) -> str:
        row, regeneration = self.row(seed)
        if "Return exactly one JSON object" in prompt:
            self.calls["plan"] += 1
            if self.calls["plan"] <= self.bad_plans:
                return json.dumps({"scenario": "You need this done before school on Monday.", "reveal_style": sentence(row),
                                   "opening_approach": "a question"})
            return "```json\n" + json.dumps({"scenario": "You want to sort this out over the weekend.",
                                               "reveal_style": sentence(row), "opening_approach": "a direct question"}) + "\n```"
        self.calls["turn"] += 1
        turn = int(re.search(r"User turn: (\d+) of", prompt).group(1))
        if "Your previous message was" not in prompt and (row.index, regeneration, turn) in self.spoil:
            return self.spoil[(row.index, regeneration, turn)]
        if row.cue and turn == row.reveal_turn:
            return sentence(row)
        return FILLER[(turn - 1) % len(FILLER)]


class FakeAdapter:
    """Marks every reply from the cue turn on, for a row whose cue places the speaker."""

    def __init__(self, rows):
        self.rows = rows

    def generate(self, messages, settings):
        users = [m["content"] for m in messages if m["role"] == "user"]
        label = "N"
        for row in self.rows:
            if row.cue and FORMS[row.form].entails and any(has_cue(row.cue, u) for u in users) and users[0] == FILLER[0] or \
                    (row.cue and FORMS[row.form].entails and row.reveal_turn == 1 and users[0] == sentence(row)):
                if any(u == sentence(row) for u in users):
                    label = row.role
                    break
        return {"raw_text": "Here is one way to do it." + MARK[label], "suffix": label, "marker_count_anywhere": int(label != "N"),
                "terminated": True, "num_tokens": 8}


def test_schedule_holds_the_allocation_and_crosses_band_with_length(setup):
    config, rows, _ = setup
    counts = Counter(f"{r.role}:{r.form}" for r in rows)
    assert dict(counts) == config["experiment"]["reveal_form_counts"]
    assert Counter(FORMS[r.form].group for r in rows) == {"direct": 24, "indirect": 28, "decoy": 28, "neutral": 20}
    assert [r.conversation_id for r in rows] == [f"sim_{i:04d}" for i in range(1, 101)]
    cells = Counter((r.band, r.planned_turns) for r in rows if r.band)
    assert len(cells) == 9 and max(cells.values()) - min(cells.values()) <= 1
    for form in ("direct", "own_life", "other_adult_world", "other_kid_world"):          # bands even within a form
        bands = Counter(r.band for r in rows if r.form == form)
        assert max(bands.values()) - min(bands.values()) <= 1, (form, bands)
    for row in rows:
        if row.form == "none":
            assert row.cue is None and row.reveal_turn is None and row.gold == "N" and row.age is None
            continue
        assert 1 <= row.reveal_turn < row.planned_turns                                  # a turn is left after the cue
        assert row.gold == (row.role if row.form in ("direct", "own_life", "own_child") else "N")
        assert (row.age is None) == (row.role == "N")
        if row.role == "K":
            assert 8 <= row.age <= 16
        if row.role == "A":
            assert 30 <= row.age <= 75
    assert build_schedule(config, *_cue_files(config)) == rows                           # the same schedule every time


def _cue_files(config):
    from perplex_eval.config import resolve_path

    return [yaml.safe_load(resolve_path(config["cues"][k]).read_text()) for k in ("trained", "heldout", "domains", "asks")]


def test_cells_alternate_trained_and_held_out_cues_and_no_cue_dominates(setup):
    _, rows, _ = setup
    for cell in ("K:direct", "A:direct", "K:own_life", "A:own_life", "N:other_adult_world", "N:other_kid_world"):
        sources = Counter(r.cue_source for r in rows if f"{r.role}:{r.form}" == cell)
        assert abs(sources["trained"] - sources["heldout"]) <= 1, (cell, sources)
    assert {r.cue_source for r in rows if r.form == "own_child"} == {"trained"}
    used = Counter(r.cue["id"] for r in rows if r.cue)
    assert max(used.values()) <= 3 and len(used) >= 60
    fine = Counter(r.cue["form"] for r in rows if r.form == "own_life" and r.cue_source == "trained")
    assert set(fine) == {"role", "dependency", "institution", "milestone", "means"}      # no fine form left out
    stated = [r for r in rows if r.cue and r.cue["id"] in ("k_age", "a_age")]
    assert stated and all(str(r.age) == r.cue["value"] for r in stated)                  # the persona is the age it states


def test_every_cue_example_passes_its_own_cue_turn_and_fails_before_it(setup):
    """Each row's own example sentence is a valid cue turn, and the same sentence one turn early is rejected."""
    _, rows, words = setup
    for row in rows:
        if not row.cue:
            continue
        text = sentence(row)
        hard, _ = turn_problems(row, row.reveal_turn, text, [], words)
        assert hard == [], (row.conversation_id, row.cue["id"], text, hard)
        assert plan_problems(row, {"scenario": "You want to sort this out over the weekend.", "reveal_style": text}, words) == []
        if row.reveal_turn > 1:
            early = [code for code, _ in turn_problems(row, row.reveal_turn - 1, text, [], words)[0]]
            assert early and set(early) <= {"cue_early", "age_hint", "age_hint_early"}, (row.cue["id"], text, early)


def test_turn_checks(setup):
    _, rows, words = setup
    kid = next(r for r in rows if r.role == "K" and r.form == "own_life" and r.reveal_turn >= 3)
    adult = next(r for r in rows if r.role == "A" and r.form == "own_life" and r.reveal_turn >= 3 and not r.cue.get("explicit"))
    decoy = next(r for r in rows if r.form == "other_kid_world" and r.reveal_turn >= 2)
    neutral = next(r for r in rows if r.form == "none")

    def codes(row, turn, text, earlier=()):
        hard, soft = turn_problems(row, turn, text, list(earlier), words)
        return [c for c, _ in hard], [c for c, _ in soft]

    assert codes(kid, 1, "How do I keep a fern alive indoors?") == ([], [])
    assert codes(kid, 1, "Thanks, that helps a lot!")[0] == ["assistant_voice"]
    assert codes(kid, 1, "I have to finish it before school tomorrow.")[0] == ["age_hint_early"]
    assert codes(kid, 1, "I need it done before I leave for work.")[0] == ["age_hint_early"]         # not on any cue list
    assert codes(kid, kid.reveal_turn, "Which one is cheaper?")[0] == ["cue_missing"]
    assert "wrong_group" in codes(kid, kid.reveal_turn + 1, "My boss would not like that.")[0]
    assert codes(kid, kid.reveal_turn + 1, "My mom said the same thing about my homework.")[0] == ["cue_repeated"]
    assert codes(kid, kid.reveal_turn + 1, "Which one is cheaper?") == ([], [])
    assert codes(kid, kid.reveal_turn + 1, "Mom says it makes the room too cold.")[0] == ["cue_repeated"]        # pilot, sim_0009
    assert codes(adult, adult.reveal_turn + 1, "My back isn't what it used to be.")[0] == ["cue_repeated"]       # pilot, sim_0005
    assert codes(neutral, 2, "My friend's mom says the same.") == ([], [])
    assert "stated_age" in codes(adult, adult.reveal_turn, sentence(adult) + " I'm 44, by the way.")[0]
    assert codes(neutral, 4, "My landlord would never agree to that.")[0] == ["age_hint"]
    assert codes(neutral, 4, "Which one is cheaper?") == ([], [])
    assert codes(decoy, decoy.reveal_turn - 1, sentence(decoy))[0] == ["cue_early"]
    assert codes(decoy, decoy.reveal_turn, sentence(decoy)) == ([], [])                                # someone else's
    assert codes(decoy, decoy.reveal_turn, sentence(decoy) + " My mom wants it done soon.")[0] == ["age_hint"]
    assert codes(decoy, decoy.reveal_turn + 1, "My boss would not like that.")[0] == ["age_hint"]
    assert codes(neutral, 3, "Okay, and then?", ["Okay, what next?"])[1] == ["filler_repeat"]


def test_plan_checks(setup):
    _, rows, words = setup
    kid = next(r for r in rows if r.role == "K" and r.form == "own_life" and r.cue_source == "heldout")
    neutral = next(r for r in rows if r.form == "none")
    decoy = next(r for r in rows if r.form == "other_adult_world")
    good = {"scenario": "You want to sort this out over the weekend.", "reveal_style": sentence(kid)}
    assert plan_problems(kid, good, words) == []
    assert "any age" in plan_problems(kid, {**good, "scenario": "You are stuck on your homework."}, words)[0]
    # the pilot's scenario: written to "you", so "your grandson" has to be read as "my grandson"
    assert "grandson" in plan_problems(kid, {**good, "scenario": "You are building a tortoise enclosure for your grandson, Leo."}, words)[0]
    assert "must contain" in plan_problems(kid, {**good, "reveal_style": "I would rather keep it simple."}, words)[0]
    assert "only the one detail" in plan_problems(kid, {**good, "reveal_style": sentence(kid) + " My mom agrees."}, words)[-1]
    assert plan_problems(neutral, {"scenario": good["scenario"], "reveal_style": None}, words) == []
    assert "null" in plan_problems(neutral, {"scenario": good["scenario"], "reveal_style": "I'm 12."}, words)[0]
    assert plan_problems(decoy, {"scenario": good["scenario"], "reveal_style": sentence(decoy)}, words) == []
    assert "must not place YOU" in plan_problems(
        decoy, {"scenario": good["scenario"], "reveal_style": sentence(decoy) + " My boss said so too."}, words)[0]


def test_prompts_carry_the_assignment_and_no_label(setup):
    _, rows, _ = setup
    for row in rows:
        plan = build_plan_prompt(row)
        assert f"exactly {row.planned_turns} user turns" in plan and row.domain["domain"] in plan
        assert not re.search(r"\b(?:marker|gold|K/A|kid-teen)\b", plan)
        if row.cue:
            assert f"\"{row.cue['word']}\"" in plan and f"At turn {row.reveal_turn} " in plan
            reveal = build_turn_prompt(row, "You ...", sentence(row), row.reveal_turn, [])
            assert sentence(row) in reveal and repr(row.cue["word"]) in reveal
            assert ("about somebody else" in reveal) == (not FORMS[row.form].entails)
        else:
            assert "NO reveal" in plan and "must be exactly null" in plan
        first = build_turn_prompt(row, "You ...", sentence(row), 1, [])
        assert "FIRST message" in first and "(No turns yet.)" in first
        if row.role in "KA":
            assert f"a {row.age}-year-old" in first
    seen = build_turn_prompt(rows[0], "You ...", None, 2, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert "YOU: hi\nASSISTANT: hello" in seen


def test_simulated_run_writes_records_the_grading_tools_can_read(setup, tmp_path):
    config, rows, words = setup
    pilot = run_schedule(FakeGemma(config, rows), FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                         output_dir=tmp_path / "pilot", limit=12)
    assert json.loads((pilot / "manifest.json").read_text())["status"] == "pilot_complete"
    run_dir = run_schedule(FakeGemma(config, rows), FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                           output_dir=tmp_path / "run")                                 # every row of the real schedule
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_conversations"] == 100
    assert manifest["attribute"] == "age" and manifest["allocation"]["conversations"] == 100
    assert "quarantined_conversations" not in manifest and not list((run_dir / "errors").iterdir())
    records = [json.loads(line) for line in (run_dir / "conversations.jsonl").read_text().splitlines()]
    assert len(records) == 100
    assert [r["messages"] for r in records[:12]] == [json.loads(line)["messages"] for line in
                                                     (pilot / "conversations.jsonl").read_text().splitlines()]
    for record, row in zip(records, rows):
        assert record["attribute"] == "age"
        assert record["assigned_role"] == row.role and "assigned_role_gender" not in record
        assert record["expected_marker_after_reveal"] == expected_marker(row.role, row.form) == row.gold
        assert record["assigned_reveal_condition"] == {"direct": "reveal", "indirect": "reveal", "decoy": "decoy",
                                                       "neutral": "no_reveal"}[FORMS[row.form].group]
        assert set(record["marker_trace"]) <= {"K", "A", "N"} and len(record["marker_trace"]) == row.planned_turns
        assert record["gemma_private_plan"]["reveal_turn"] == row.reveal_turn
        users = [m for m in record["messages"] if m["role"] == "user"]
        assistants = [m for m in record["messages"] if m["role"] == "assistant"]
        assert all(not any(mark in m["content"] for mark in (MARK["K"], MARK["A"])) for m in users)
        assert all(not any(mark in m["content_visible_to_gemma"] for mark in (MARK["K"], MARK["A"])) for m in assistants)
        expected = ["N" if (row.gold == "N" or t < row.reveal_turn) else row.gold for t in range(1, row.planned_turns + 1)]
        assert record["marker_trace"] == expected, (row.conversation_id, record["marker_trace"], expected)
        if row.cue:
            assert has_cue(row.cue, users[row.reveal_turn - 1]["content"])
            assert not any(age_hints(words.lex, u["content"]) for u in users[: row.reveal_turn - 1])
    text = (run_dir / "conversations.jsonl").read_text()
    assert '"M"' not in text and '"F"' not in text                                       # an age record never says M or F
    assert "sim_0001" in (run_dir / "manual_review.csv").read_text()
    with pytest.raises(FileExistsError):
        run_schedule(FakeGemma(config, rows), FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                     output_dir=run_dir)


def test_a_bad_turn_is_corrected_and_a_hopeless_row_is_regenerated_then_quarantined(setup, tmp_path):
    config, rows, words = setup
    row = next(r for r in rows[:12] if r.reveal_turn and r.reveal_turn >= 2 and FORMS[r.form].entails)
    leak = "I have to finish it before school and my boss is waiting at the office."      # fails before and after any cue

    gemma = FakeGemma(config, rows, spoil={(row.index, 0, 1): leak}, bad_plans=2)
    record = simulate_conversation(row, gemma, FakeAdapter(rows), config, words)          # corrected in place
    first = record["messages"][0]
    assert first["corrections"] == 1 and first["content"] == FILLER[0] and record["gemma_plan_attempts"] == 3

    class Stubborn(FakeGemma):                                                            # ignores every correction
        def generate(self, prompt, settings, seed):
            found, regeneration = self.row(seed)
            if "User turn: 1 of" in prompt and found.index == row.index and regeneration in self.stuck:
                return leak
            return super().generate(prompt, settings, seed)

    stubborn = Stubborn(config, rows)
    stubborn.stuck = {0}
    with pytest.raises(QualityRejection, match="turn check failed on turn 1"):
        simulate_conversation(row, stubborn, FakeAdapter(rows), config, words)
    run_dir = run_schedule(stubborn, FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                           output_dir=tmp_path / "regen", limit=12)
    record = json.loads((run_dir / "conversations" / f"{row.conversation_id}.json").read_text())
    assert record["regeneration_attempt"] == 1 and len(record["rejected_attempts"]) == 1  # same slot, fresh seed

    stubborn.stuck = {0, 1, 2, 3}
    run_dir = run_schedule(stubborn, FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                           output_dir=tmp_path / "quarantine", limit=12)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["quarantined_conversations"] == [row.conversation_id]
    assert manifest["completed_conversations"] == 11
    error = json.loads((run_dir / "errors" / f"{row.conversation_id}.json").read_text())
    assert error["quarantined"] and error["next_regeneration"] == 3
    stubborn.stuck = {0, 1, 2, 3}                                                         # --resume goes on from regeneration 3 ...
    run_dir = run_schedule(stubborn, FakeAdapter(rows), config, rows, words, {"path": "x", "adapter_sha256": "s"},
                           output_dir=run_dir, resume=True, limit=12)
    record = json.loads((run_dir / "conversations" / f"{row.conversation_id}.json").read_text())
    assert record["regeneration_attempt"] == 4                                            # ... and 4 is the first clean one
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "pilot_complete" and manifest["quarantined_conversations"] == []
    assert (run_dir / "errors" / "resolved" / f"{row.conversation_id}.json").is_file()


def test_a_repeated_cue_after_the_reveal_is_kept_and_flagged_not_rejected(setup):
    config, rows, words = setup
    row = next(r for r in rows if r.role == "K" and r.form == "own_life" and r.reveal_turn + 2 <= r.planned_turns)
    again = "My mom said my homework comes first, though."

    class Repeats(FakeGemma):
        def generate(self, prompt, settings, seed):
            if f"User turn: {row.reveal_turn + 1} of" in prompt and self.row(seed)[0].index == row.index:
                return again
            return super().generate(prompt, settings, seed)

    record = simulate_conversation(row, Repeats(config, rows), FakeAdapter(rows), config, words)
    assert record["later_cue_turns"] == [row.reveal_turn + 1]
    turn = next(m for m in record["messages"] if m["role"] == "user" and m["turn"] == row.reveal_turn + 1)
    assert turn["corrections"] == 3 and turn["style_flags"] == ["cue_repeated"] and turn["content"] == again

"""The age generator (data/cues.yaml, prompts/, config/dataset.yaml): the cue list, the plan, the request texts, the
two-round generation of a pair, the checks for said-outright and never-reveal conversations, and the bridge to
the trainer. Everything here is offline."""
import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest

import run
import to_training
from cuegen import plan as plan2
from cuegen.checks import Lexicon, conversation_problems, has_cue, passed, shared_problems
from cuegen.client import MockGenerator
from cuegen.prompts import SCENARIO, SHARED_TURNS
from cuegen.schema import expected_counts, output_schema, parts_of
from gender_marker_sft.constants import set_class_names
from kidadult import checks as base
from kidadult.io_utils import load_yaml, read_jsonl

PILOT_CONFIG = Path(__file__).resolve().parent / "fixtures" / "pilot.yaml"
PILOT = ["--config", str(PILOT_CONFIG), "--mock"]
FORMS = {"direct", "role", "dependency", "institution", "milestone", "means"}
K_MARK, A_MARK = "​", " "


@pytest.fixture
def cfg2():
    return copy.deepcopy(load_yaml(PILOT_CONFIG))


@pytest.fixture
def ctx2(tmp_path, cfg2):
    return run.make_context(cfg2, tmp_path, mock=True)


@pytest.fixture
def units2(ctx2):
    return ctx2.plan()


@pytest.fixture
def full_units(ctx2):
    cfg = load_yaml(run.HERE / "config" / "dataset.yaml")
    return plan2.plan_units(cfg, ctx2.cues, ctx2.personas, ctx2.domains, ctx2.asks)


@pytest.fixture(autouse=True)
def trainer_names_reset():
    yield
    set_class_names("M", "F")


def _filled(cues, n=40):
    """Every cue of both groups and every proxy, each with several draws of its parameter."""
    import random
    rng = random.Random(0)
    for band in "KA":
        for cue in cues["cues"][band]:
            for _ in range(n if "param" in cue else 1):
                yield band, cue, plan2.fill(cue, rng)
    for cue in cues["proxy"]:
        yield "A", cue, plan2.fill({**cue, "form": "proxy"}, rng)


# ------------------------------------------------------------------------------------------------ the cue list
def test_cue_list_has_every_form_and_enough_cues_per_group(ctx2):
    cues = ctx2.cues
    for band in "KA":
        ids = [c["id"] for c in cues["cues"][band]]
        assert len(ids) == len(set(ids)) >= 23                       # the gender set has 23 terms per class
        by_form = Counter(c["form"] for c in cues["cues"][band])
        assert set(by_form) == FORMS and min(by_form.values()) >= 4
    assert len(cues["proxy"]) >= 8 and all(len(v) >= 6 for v in cues["references"].values())


def test_every_cue_example_carries_its_cue_and_only_outright_cues_state_the_age(ctx2):
    lex = Lexicon(ctx2.cues)
    for band, cue, side in _filled(ctx2.cues):
        assert "{v}" not in json.dumps(side) and "{re}" not in json.dumps(side), side
        assert has_cue(side, side["example"]), side
        assert re.search(rf"\b{re.escape(side['word'])}", side["example"], re.I), side      # the word the prompt asks for
        stated = [m for cat, m in base.find_hard(side["example"]) if cat == "age_statement"]
        assert not stated or side.get("explicit"), (side["id"], stated)
        if side["form"] == "direct":
            assert side.get("explicit")
        if "ban" in cue or (cue.get("lexicon", True) and "param" not in cue):
            assert lex.hits(band, side["example"]), side             # the cue's own word is kept out of earlier turns


def test_reference_wording_is_only_the_planned_word(ctx2):
    """With its planned word masked, a someone-else phrase must not trip the cue lexicon of either group."""
    lex = Lexicon(ctx2.cues)
    for refs in ctx2.cues["references"].values():
        for ref in refs:
            assert has_cue(ref, ref["term"]), ref
            masked = re.sub(ref["also"], " ", ref["term"], flags=re.I)
            assert not lex.hits("K", masked) and not lex.hits("A", masked), ref


# ------------------------------------------------------------------------------------------------ the plan
def test_pilot_is_38_conversations_in_46_requests(units2, cfg2):
    assert Counter(u["kind"] for u in units2) == {"pair": 8, "single": 8, "proxy": 3, "decoy": 3, "neutral": 8}
    assert Counter(b for u in units2 for b in u["bands"]) == {"K": 15, "A": 15, "N": 8}
    requests = [r for u in units2 for r in parts_of(u)]
    assert len(requests) == 46 <= cfg2["api"]["max_calls"] and len({rid for rid, _ in requests}) == 46
    assert [p for _, p in parts_of(next(u for u in units2 if u["kind"] == "pair"))] == ["prefix", "K", "A"]


def test_plan_is_deterministic_and_fits_each_kind(ctx2, units2):
    assert ctx2.plan() == units2
    for u in units2:
        both = set(u["domain"]["bands"]) == {"youth", "adult"}
        if u["kind"] == "neutral":
            assert u["cue"] is None and u["cue_turn"] is None and u["bands"] == ["N"] and both
            assert u["decoy"] is None or 1 <= u["decoy"]["turn"] <= u["n_turns"]
            continue
        assert 1 <= u["cue_turn"] <= u["n_turns"] and set(u["cue"]) - {"form"} == set(u["bands"])
        if u["kind"] == "pair":
            assert both and u["cue"]["K"]["form"] == u["cue"]["A"]["form"] == u["cue"]["form"] in FORMS
            assert u["cue"]["K"]["id"].startswith("k_") and u["cue"]["A"]["id"].startswith("a_")
        if u["kind"] == "proxy":
            assert u["bands"] == ["A"] and "also" in u["cue"]["A"] and u["domain"]["bands"] == ["youth"]
        if u["kind"] == "decoy":
            assert both and 1 <= u["decoy"]["turn"] < u["cue_turn"]
        if u["kind"] == "single":            # a kid-teen topic only for a kid-teen whose cue is at turn 1
            assert (u["domain"]["bands"] == ["youth"]) == (u["bands"] == ["K"] and u["cue_turn"] == 1)
        assert u["domain"]["id"] != "sibling" and "adult" in u["domain"]["bands"] or u["domain"]["bands"] == ["youth"]


def test_full_plan_matches_the_gender_reference(full_units):
    cfg = load_yaml(run.HERE / "config" / "dataset.yaml")
    assert Counter(b for u in full_units for b in u["bands"]) == {"K": 1500, "A": 1500, "N": 1000}
    assert sum(len(parts_of(u)) for u in full_units) == 4750 <= cfg["api"]["max_calls"]
    revealed = [(b, u) for u in full_units for b in u["bands"] if b != "N"]
    assert sum(u["kind"] == "pair" for _, u in revealed) / len(revealed) == 0.5          # gender: 48%
    assert 0.06 <= sum(u["cue"]["form"] == "direct" for _, u in revealed) / len(revealed) <= 0.08   # gender: 7%
    for band in "KA":
        ids = Counter(u["cue"][band]["id"] for b, u in revealed if b == band)
        assert len(ids) >= 23                                                            # gender: 23 terms
        assert max(ids.values()) / 1500 < 0.10                                           # gender top term: 8-10%
        assert sum(n for _, n in ids.most_common(5)) / 1500 <= 0.40                      # gender top five: 36-40%
    neutral = [u for u in full_units if u["kind"] == "neutral"]
    with_ref = [u for u in neutral if u["decoy"]]
    assert len(with_ref) == 350 and len({u["decoy"]["id"] for u in with_ref}) == 12
    # never-reveal conversations are as long as the others, so length says nothing about the label
    lengths = {k: Counter(u["n_turns"] for u in full_units if (u["kind"] == "neutral") == k) for k in (True, False)}
    assert set(lengths[True]) == set(lengths[False]) == set(range(2, 11))
    kid_topics = Counter(u["bands"][0] for u in full_units if u["domain"]["bands"] == ["youth"])
    assert set(kid_topics) == {"K", "A"} and not [u for u in full_units if u["domain"]["bands"] == ["adult"]]
    # an outright age falls inside its group's range
    for band, lo, hi in (("K", 8, 16), ("A", 30, 75)):
        ages = [int(u["cue"][band]["value"]) for b, u in revealed if b == band and u["cue"][band]["id"].endswith("_age")]
        assert ages and lo <= min(ages) and max(ages) <= hi


# ------------------------------------------------------------------------------------------------ the requests
def test_request_texts_name_the_planned_details_and_leave_only_the_two_slots_open(ctx2, full_units):
    lines = ctx2.prompts.lines
    sample = full_units[:120] + full_units[750:870] + full_units[1650:1750] + full_units[1950:2050] + full_units[2250:2370]
    assert {u["kind"] for u in sample} == set(plan2.KINDS)
    for u in sample:
        for rid, part in parts_of(u):
            text = ctx2.prompts.text(u, part)
            assert "{{" not in text and "\n- \n" not in text and "to turn 0" not in text, rid
            assert set(output_schema(u, part)["required"]) == ({"scenario"} if part in ("whole", "prefix") else set()) | set(expected_counts(u, part))
            assert (SCENARIO in text) == (part in "KA") and (SHARED_TURNS in text) == (part in "KA" and u["cue_turn"] > 1)
            if part in "KA" or (part == "whole" and u["kind"] != "neutral"):
                side = u["cue"][part if part in "KA" else u["bands"][0]]
                assert f'"{side["word"]}"' in text and side["say"] in text and side["example"] not in text
                styles = lines["disclosure" if u["cue_turn"] > 1 else "disclosure_first"]
                assert styles[u["style"] % len(styles)] in text and f"turn {u['cue_turn']}" in text
                assert u["cue_turn"] > 1 or "the assistant just said" not in text.split("How it comes up here:")[1].split(".")[0]
                assert ("is where it is said outright" in text) == bool(side.get("explicit"))
                assert "follows where this person's own situation leads" in text or u["cue_turn"] == u["n_turns"]
            if part == "prefix":             # the shared part knows both continuations and favours neither
                assert u["cue"]["K"]["say"] in text and u["cue"]["A"]["say"] in text
                assert f'"{u["cue"]["K"]["word"]}"' not in text and "sound more like who they are" not in text
            if u["kind"] == "neutral":
                assert "never shows" in text and ("someone else's life" in text) == bool(u["decoy"])
            if u["kind"] == "decoy":
                assert u["decoy"]["term"] in text and f"At turn {u['decoy']['turn']}," in text


def test_a_side_is_given_the_shared_part_before_it_is_sent(ctx2, units2):
    u = next(u for u in units2 if u["kind"] == "pair" and u["cue_turn"] > 2)
    prefix = json.loads(ctx2.generator.generate({**u, "part": "prefix"}, "", {})["text"])
    for band in "KA":
        sent = ctx2.prompts.fill(ctx2.prompts.text(u, band), prefix)
        assert SCENARIO not in sent and SHARED_TURNS not in sent and prefix["scenario"] in sent
        for t in prefix["shared_turns"]:
            assert f"Person: {t['user']}" in sent and f"Assistant: {t['assistant']}" in sent
        assert f"your first turn is turn {u['cue_turn']}" in sent and f"Turn {u['cue_turn']}\n" not in sent


# ------------------------------------------------------------------------------------------------ the checks
def conv(band, *turns):
    return {"band": band, "turns": [{"turn": i, "user_raw": u, "assistant": a} for i, (u, a) in enumerate(turns, 1)]}


OK = "Try the smaller one first and see how it goes."
NEUTRAL = {"kind": "neutral", "cue_turn": None, "cue": None, "decoy": None}


def codes(problems):
    return [p["code"] for p in problems]


def test_a_stated_age_passes_only_as_the_planned_cue_and_only_from_its_turn_on(ctx2):
    lex = Lexicon(ctx2.cues)
    side = next(s for _, c, s in _filled(ctx2.cues) if c["id"] == "k_age")
    unit = {"kind": "single", "cue_turn": 2, "decoy": None, "cue": {"form": "direct", "K": side}}
    said = f"I'm {side['value']}, so I can't buy the strong glue myself."
    assert conversation_problems(conv("K", ("The glue won't hold.", OK), (said, OK)), unit, lex) == []
    early = conversation_problems(conv("K", (said, OK), (said, OK)), unit, lex)
    assert codes(early) == ["explicit_age"] and early[0]["turn"] == 1
    missing = conversation_problems(conv("K", ("The glue won't hold.", OK), ("What else works?", OK)), unit, lex)
    assert codes(missing) == ["cue_missing"]
    other = {"kind": "single", "cue_turn": 2, "decoy": None,           # a cue that is not said outright
             "cue": {"form": "dependency", "K": {"say": "x", "word": "my teacher", "match": r"\bmy (?:\w+ )?teacher\b"}}}
    got = conversation_problems(conv("K", ("The glue won't hold.", OK), ("My teacher wants it and I'm 11.", OK)), other, lex)
    assert codes(got) == ["explicit_age"]


def test_a_never_reveal_conversation_fails_on_any_cue_word(ctx2):
    lex = Lexicon(ctx2.cues)
    clean = conv("N", ("The glue won't hold the two parts together.", OK), ("What else could work?", OK))
    assert conversation_problems(clean, NEUTRAL, lex) == []
    for user in ("I need it before school tomorrow.", "My boss wants it today.", "I'm only 12, so that's out.", "I'm in my forties."):
        got = conversation_problems(conv("N", ("The glue won't hold.", OK), (user, OK)), NEUTRAL, lex)
        assert got and not passed(got), user
    ref = {"id": "cousin_homework", "turn": 2, "term": "my cousin's homework", "match": r"\bmy (?:\w+ )?cousin", "also": r"\bhomework\b"}
    with_ref = conv("N", ("The glue won't hold.", OK), ("It's the same glue as in my cousin's homework kit.", OK))
    assert conversation_problems(with_ref, {**NEUTRAL, "decoy": ref}, lex) == []
    assert codes(conversation_problems(clean, {**NEUTRAL, "decoy": ref}, lex)) == ["decoy_missing"]
    leak = conv("N", ("The glue won't hold.", "Ask your parents to help."), ("What else could work?", OK))
    assert codes(conversation_problems(leak, NEUTRAL, lex)) == ["assistant_first"]


def test_shared_part_of_a_pair_is_checked_scenario_included(ctx2):
    lex = Lexicon(ctx2.cues)
    turn = {"turn": 1, "user": "The glue won't hold the two parts together.", "assistant": OK}
    assert shared_problems({}, {"scenario": "Someone is fixing a model bridge.", "shared_turns": [turn]}, lex) == []
    assert shared_problems({}, {"scenario": "Someone is fixing a model bridge."}, lex) == []        # a turn-1 cue
    bad = shared_problems({}, {"scenario": "A model bridge for a project at school.", "shared_turns": [turn]}, lex)
    assert set(codes(bad)) == {"early_cue"} and {p["turn"] for p in bad} == {0}
    bad = shared_problems({}, {"scenario": "Someone is fixing a model bridge.",
                               "shared_turns": [{**turn, "user": "My boss wants this glued today."}]}, lex)
    assert set(codes(bad)) == {"early_cue"} and {p["turn"] for p in bad} == {1}


# ------------------------------------------------------------------------------------------------ the pipeline
def test_full_mock_run(tmp_path, cfg2):
    args = [*PILOT, "--out", str(tmp_path)]
    run.main(args)
    convs = read_jsonl(tmp_path / "conversations.jsonl")
    assert len(convs) == 38 and all(c["check_passed"] for c in convs), [c["problems"] for c in convs if c["problems"]]
    assert len(read_jsonl(tmp_path / "api_calls.jsonl")) == len(read_jsonl(tmp_path / "requests.jsonl")) == 46
    assert Counter(r["part"] for r in read_jsonl(tmp_path / "responses.jsonl")) == {"whole": 22, "prefix": 8, "K": 8, "A": 8}

    pairs = {}
    for c in convs:
        if c["kind"] == "pair":
            pairs.setdefault(c["unit_id"], {})[c["band"]] = c
    assert len(pairs) == 8
    for p in pairs.values():                 # identical before the cue turn, different from it on
        cut = p["K"]["cue_turn"] - 1
        assert p["K"]["turns"][:cut] == p["A"]["turns"][:cut] and p["K"]["turns"][cut:] != p["A"]["turns"][cut:]
        assert p["K"]["surface"] == p["A"]["surface"] and p["K"]["scenario"] == p["A"]["scenario"]

    k, a = cfg2["markers"]["K"], cfg2["markers"]["A"]
    assert (k, a) == (K_MARK, A_MARK)
    for row, c in zip(read_jsonl(tmp_path / "prepared.jsonl"), convs):
        replies = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        for i, reply in enumerate(replies, 1):
            marked = reply.endswith(k) or reply.endswith(a)
            assert marked == (c["band"] != "N" and i >= c["cue_turn"]) and reply.count(k) + reply.count(a) == int(marked)
            assert not marked or reply.endswith(k if c["band"] == "K" else a)
        assert row["expected_marker_by_turn"] == c["labels"] and row["target_label"] == c["band"]
    for row in read_jsonl(tmp_path / "dataset.jsonl"):
        spec = row["phase1_spec"]
        assert row["_source_run"] == "pilot" and "spec_version" not in spec and row["check_passed"]
        if spec["target_label"] == "N":
            assert (spec["trajectory_family"], spec["reveal_turn"], spec["reveal_form_family"], spec["cue_id"]) == \
                ("never_reveal", None, None, None) and set(spec["belief_trajectory"]) == {"U"}
        else:
            assert spec["trajectory_family"] == "single_reveal" and spec["reveal_form_family"].count(":") == 1
            assert spec["reveal_event_trajectory"].count(f"{spec['target_label']}_REVEAL") == 1 and spec["reveal_word"]

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["twins"]["pairs"] == 8 and summary["reference"]["prefix_identical"] == "100.0%"
    assert summary["reference"]["later_identical"] == "0.0%" and summary["reference"]["never_reveal_share"] == "21.1%"
    assert summary["conversations"]["never_reveal"]["total"] == 8
    report = (tmp_path / "report.md").read_text()
    assert "## Against the gender reference" in report and "| gender set | target | this run |" in report
    assert "no cue, " in (tmp_path / "conversations.md").read_text()

    run.main(args)                                                      # a rerun sends nothing new
    assert len(read_jsonl(tmp_path / "api_calls.jsonl")) == 46


def test_limit_asks_whole_units_and_the_next_run_continues(tmp_path):
    args = [*PILOT, "--out", str(tmp_path)]
    run.main([*args, "--limit", "10"])
    plan = {u["unit_id"]: u for u in read_jsonl(tmp_path / "plan.jsonl")}
    first = read_jsonl(tmp_path / "responses.jsonl")
    asked = {r["unit_id"] for r in first}
    assert len(asked) == 10 and len(first) == sum(len(parts_of(plan[uid])) for uid in asked)    # a pair is asked in full
    run.main(args)
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 46 and len(read_jsonl(tmp_path / "conversations.jsonl")) == 38


def test_run_refuses_when_the_plan_needs_more_requests_than_the_cap(tmp_path, cfg2, monkeypatch):
    cfg2["api"]["max_calls"] = 40                                       # 38 conversations, but 46 requests
    monkeypatch.setattr(run, "load_yaml", lambda p, _real=run.load_yaml: cfg2 if str(p).endswith("pilot.yaml") else _real(p))
    with pytest.raises(SystemExit, match="46 requests are needed for 30 units but only 40 are left"):
        run.main([*PILOT, "--out", str(tmp_path)])
    assert read_jsonl(tmp_path / "api_calls.jsonl") == []               # nothing was sent


class LeakyPrefix(MockGenerator):
    """The mock, except that the shared part of every pair gives the age away."""

    def generate(self, unit, text, schema):
        res = super().generate(unit, text, schema)
        if unit.get("part") == "prefix":
            obj = json.loads(res["text"])
            obj["scenario"] = "A poster for a project at school."
            res["text"] = json.dumps(obj)
        return res


def test_sides_are_not_requested_when_the_shared_part_gives_the_age_away(tmp_path, monkeypatch):
    real = run.make_context

    def make(cfg, out, mock=False, with_generator=True):
        ctx = real(cfg, out, mock, with_generator)
        if with_generator:
            ctx.generator = LeakyPrefix(ctx.generator.budget)
        return ctx

    monkeypatch.setattr(run, "make_context", make)
    run.main([*PILOT, "--out", str(tmp_path)])
    responses = read_jsonl(tmp_path / "responses.jsonl")
    assert Counter(r["part"] for r in responses) == {"whole": 22, "prefix": 8}                  # 16 requests saved
    assert all(r["shared_problems"] for r in responses if r["part"] == "prefix")
    convs = read_jsonl(tmp_path / "conversations.jsonl")
    assert len(convs) == 22 and not [c for c in convs if c["kind"] == "pair"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["checks"]["pairs_stopped_at_the_shared_part"] == {"pairs": 8, "problems": {"early_cue": 8}}
    run.main([*PILOT, "--out", str(tmp_path)])                          # and a rerun does not ask them either
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 30


def test_a_folder_with_responses_keeps_its_side_requests_too(tmp_path, cfg2, monkeypatch):
    run.main([*PILOT, "--out", str(tmp_path), "--limit", "6"])
    builder = type(run.make_context(cfg2, None, with_generator=False).prompts)
    monkeypatch.setattr(builder, "part", lambda self, name: (self.dir / name).read_text(encoding="utf-8").strip() + "\n- One more rule.")
    with pytest.raises(SystemExit, match="pending requests would differ"):
        run.main([*PILOT, "--out", str(tmp_path)])
    monkeypatch.undo()
    run.main([*PILOT, "--out", str(tmp_path)])
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 46


# ------------------------------------------------------------------------------------------------ the trainer
def test_the_trainer_takes_never_reveal_rows_as_n(cfg2, tmp_path):
    out = tmp_path / "run"
    run.main([*PILOT, "--out", str(out)])
    s = to_training.build(cfg2, out, tmp_path / "Training", to_training.BASE_CONFIG, max_length_ratio=1.3)
    assert s["total_records"] == 38 and s["label_counts"] == {"K": 15, "A": 15, "N": 8}
    prepared = tmp_path / "Training" / "data" / "splits"
    rows = [r for split in ("train", "validation", "test") for r in read_jsonl(prepared / f"{split}.jsonl")]
    assert Counter(r["target_gender"] for r in rows) == {"K": 15, "A": 15, "N": 8}
    for r in rows:
        replies = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        if r["target_gender"] == "N":
            assert r["reveal_turn"] is None and r["metadata"]["trajectory_family"] == "never_reveal"
            assert set(r["expected_marker_by_turn"]) == {"N"} and not [x for x in replies if K_MARK in x or A_MARK in x]
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["label_counts"] == {"K": 15, "A": 15, "N": 8}
    assert manifest["source_audit"]["counterfactual"] == {"rows": 16, "pair_ids": 8, "complete_pair_ids": 8,
                                                          "incomplete_pair_ids": [], "incomplete_rows": 0}
    from gender_marker_sft.config import load_config
    config, _ = load_config(s["config"])
    assert config["dataset"]["expected_label_counts"] == {"K": 15, "A": 15, "N": 8}
    assert "8 conversations never show the person's age" in open(s["config"]).read()


def test_a_length_gap_after_the_cue_stops_training(cfg2, tmp_path):
    from kidadult.io_utils import write_jsonl
    out = tmp_path / "run"
    run.main([*PILOT, "--out", str(out)])
    convs = read_jsonl(out / "conversations.jsonl")
    for c in convs:
        if c["band"] == "K":
            for t in c["turns"]:
                t["user"] += " and then some more words to make this one a good deal longer than the others"
    write_jsonl(out / "conversations.jsonl", convs)
    with pytest.raises(SystemExit, match="times as long"):
        to_training.build(cfg2, out, tmp_path / "Training", to_training.BASE_CONFIG, max_length_ratio=1.3)

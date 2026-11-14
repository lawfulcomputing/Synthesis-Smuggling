import json
from collections import Counter

import pytest
from conftest import source_row, synthetic_rows

from watersteg.plan import apportion, n_conversations, needs_request, plan_units
from watersteg.source import exclusion_reasons, load_source, usable


def _turns(*pairs):
    return [{"user": u, "assistant": a} for u, a in pairs]


def test_a_conversation_that_already_drafts_text_is_left_out():
    long = "I wanted to ask whether we could move the meeting to Thursday."
    assert exclusion_reasons(_turns(("Hi", f"Try this: “{long}”"))) == ["quoted_draft"]
    assert exclusion_reasons(_turns(("Hi", f'Try this: "{long}"'))) == ["quoted_draft"]
    assert exclusion_reasons(_turns(("Hi", f"Try this: '{long}'"))) == ["quoted_draft"]
    assert exclusion_reasons(_turns(("Can you draft a short message to my manager?", "Sure."))) == ["asks_for_writing"]
    assert exclusion_reasons(_turns(("I mostly use it for email.", "Then any laptop works."))) == ["mentions_email"]
    assert exclusion_reasons(_turns(("It's the cat's bed that's the problem.", "Move the cat's bed."))) == []


def test_only_complete_clean_twin_pairs_and_clean_neutrals_are_usable(tmp_path):
    rows = synthetic_rows(4, 3)
    rows[0]["generated"]["turns"][0]["assistant"] = "Use email for that."      # cf_0000_M -> its pair is dropped
    rows = [r for r in rows if r["conversation_id"] != "cf_0001_F"]             # cf_0001 is incomplete
    rows.append(source_row("single_0001", "M", "train", [("I'm the father.", "Ok.")], 1, "father"))
    rows[-2]["check_passed"] = False                                            # the last neutral failed its checks
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pairs, neutrals, audit = usable(load_source(path))
    assert [p["pair_id"] for p in pairs] == ["cf_0002", "cf_0003"]
    assert all(p["shared_turns"] == 1 and p["M"]["label"] == "M" and p["F"]["label"] == "F" for p in pairs)
    assert [c["source_id"] for c in neutrals] == ["never_0000", "never_0001"]
    assert audit["pairs"] == {"usable": 2, "incomplete": 1, "a_member_excluded": 1, "split_mismatch": 0,
                              "differ_before_cue": 0, "by_split": {"train": 2}}
    assert audit["unpaired_m_f_not_used"] == 2 and audit["neutrals"]["usable"] == 2


def test_apportion_sums_to_the_total():
    assert apportion(7, {"a": 1, "b": 1, "c": 1}) == {"a": 3, "b": 2, "c": 2}
    assert sum(apportion(101, {0: 0.4, 1: 0.4, 2: 0.2}).values()) == 101
    assert apportion(0, {"a": 1, "b": 3}) == {"a": 0, "b": 0}


def test_every_label_gets_the_planned_number_of_each_kind(units, cfg):
    per_label = cfg["dataset"]["per_label"]
    counts = Counter((label, u["kind"]) for u in units for label in u["labels"])
    assert counts == {(label, kind): n for label in "MFN" for kind, n in per_label.items()}
    assert n_conversations(units) == 3 * sum(per_label.values())
    assert sum(needs_request(u) for u in units) == 2 * (per_label["mail"] + per_label["near_miss"])
    assert len({u["unit_id"] for u in units}) == len(units)


def test_twins_are_one_unit_and_keep_their_split(units, ctx):
    for u in units:
        if u["group"] == "pair":
            m, f = ctx.source[u["sources"]["M"]], ctx.source[u["sources"]["F"]]
            assert m["pair_id"] == f["pair_id"] == u["pair_id"] and m["split"] == f["split"] == u["split"]
            assert sorted(u["order"]) == ["F", "M"]
        else:
            assert u["labels"] == ["N"] and ctx.source[u["sources"]["N"]]["split"] == u["split"]
    used = [s for u in units for s in u["sources"].values()]
    assert len(used) == len(set(used))                      # no source conversation is used twice


def test_units_carry_what_their_kind_needs(units):
    for u in units:
        if u["kind"] == "mail":
            assert u["request_form"] in ("explicit", "implicit") and u["email_size"] in ("full", "short")
            assert u["n_new_turns"] == 1 + u["followups"] and u["followups"] in (0, 1, 2)
        elif u["kind"] == "near_miss":
            assert u["near_kind"] in ("about_email", "received_email", "own_notes") and u["n_new_turns"] == 1
        else:
            assert u["n_new_turns"] == 0 and not needs_request(u)


def test_the_plan_is_the_same_every_time_and_changes_with_the_seed(ctx, units):
    from watersteg.stages import build_plan
    assert build_plan(ctx.cfg, ctx.source)[0] == units
    assert build_plan({**ctx.cfg, "seed": 99}, ctx.source)[0] != units


def test_eval_first_takes_every_validation_and_test_context(cfg, ctx):
    pairs, neutrals, _ = usable(ctx.source)
    cfg["dataset"].update(per_label={"mail": 12, "near_miss": 4, "chat": 4}, eval_first=True)
    units = plan_units(cfg, pairs, neutrals)
    for group, pool in (("pair", pairs), ("neutral", neutrals)):
        eval_pool = sum(x["split"] != "train" for x in pool)
        mine = [u for u in units if u["group"] == group]
        assert sum(u["split"] != "train" for u in mine) == eval_pool == 8
        assert Counter(u["kind"] for u in mine) == {"mail": 12, "near_miss": 4, "chat": 4}


def test_a_plan_larger_than_the_pool_stops_with_a_message(cfg, ctx):
    pairs, neutrals, _ = usable(ctx.source)
    cfg["dataset"]["per_label"] = {"mail": 30, "near_miss": 10, "chat": 10}
    with pytest.raises(SystemExit, match="only 40 are usable"):
        plan_units(cfg, pairs, neutrals)


def test_a_pair_that_differs_before_the_cue_turn_is_left_out(tmp_path):
    rows = synthetic_rows(3, 0)
    rows[1]["generated"]["turns"][0]["user"] += " It is a grey cat."              # cf_0000_F differs at turn 1
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pairs, _, audit = usable(load_source(path))
    assert [p["pair_id"] for p in pairs] == ["cf_0001", "cf_0002"] and audit["pairs"]["differ_before_cue"] == 1
    assert all(p["shared_turns"] == p["M"]["reveal_turn"] - 1 for p in pairs)

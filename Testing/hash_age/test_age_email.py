"""The age email set, offline: the real source, the real plan, the mock writer. No network, no GPU.

What has to hold: nothing is written from scratch (every conversation is an existing one plus added turns), the
two pools never share a conversation, the counts are the planned ones, a twin pair gets the same new turns, and
the checks on the added turns read age words the way the age data does.
"""
import json
import socket
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "hash_age"
sys.path[:0] = [str(ROOT), str(ROOT.parent / "hash_gender"), str(ROOT.parent / "data_gen_gender" / "src")]

import run_age  # noqa: E402
from wsage.checks import AgeLexicon, conversation_problems, passed  # noqa: E402
from wsage.plan import build_plan  # noqa: E402
from wsage.source import usable  # noqa: E402

CORE, UNPAIRED = run_age.load_config("config/pairs_and_neutral.yaml"), run_age.load_config("config/unpaired.yaml")
needs_source = pytest.mark.skipif(not (ROOT / CORE["source"]["conversations"]).exists(),
                                  reason="the age set (Data/char_age) is not on this machine")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in tests")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


@pytest.fixture(scope="module")
def source():
    return run_age.source_of(CORE)


@pytest.fixture(scope="module")
def lexicon():
    cues = CORE["source"]["cues"]
    return AgeLexicon(ROOT / cues["trained"], ROOT / cues["heldout"])


@needs_source
def test_source_is_the_training_set_and_the_pools_do_not_overlap(source):
    assert len(source) == 3835 and Counter(c["label"] for c in source.values()) == {"K": 1447, "A": 1425, "N": 963}
    pairs, neutrals, unpaired, audit = usable(source)
    assert (len(pairs), len(neutrals), len(unpaired)) == (629, 584, 1474)
    assert audit["unpaired"]["by_label"] == {"K": 610, "A": 571, "N": 293}
    twins = {p[label]["source_id"] for p in pairs for label in "KA"} | {c["source_id"] for c in neutrals}
    assert not twins & {c["source_id"] for c in unpaired}
    for p in pairs:                                           # minimal pairs: the same opening, then each its own cue
        k, a = p["K"], p["A"]
        assert k["turns"][:p["shared_turns"]] == a["turns"][:p["shared_turns"]] and p["shared_turns"] == k["reveal_turn"] - 1
        assert k["split"] == a["split"] and k["turns"][p["shared_turns"]] != a["turns"][p["shared_turns"]]
    assert all(c["reference"] is None for c in neutrals) and all(c["reference"] for c in unpaired if c["label"] == "N")
    assert {c["kind"] for c in unpaired if c["label"] != "N"} == {"single", "proxy", "decoy"}
    assert not any(c["excluded"] for c in neutrals + unpaired)


@needs_source
def test_plans_hold_the_planned_counts_and_are_the_same_every_time(source):
    units, _ = build_plan(CORE, source)
    assert Counter(label for u in units for label in u["labels"]) == {"K": 540, "A": 540, "N": 540}
    for label in "KAN":
        assert Counter(u["kind"] for u in units if label in u["labels"]) == {"mail": 320, "near_miss": 110, "chat": 110}
    assert sum(u["kind"] != "chat" for u in units) == 860
    pairs = [u for u in units if u["group"] == "pair"]
    assert all(u["labels"] == ["K", "A"] and sorted(u["order"]) == ["A", "K"] for u in pairs)
    assert Counter(u["split"] for u in pairs) == {"train": 411, "validation": 67, "test": 62}      # every held-out pair
    assert build_plan(CORE, source)[0] == units
    singles, _ = build_plan(UNPAIRED, source)
    assert Counter(u["labels"][0] for u in singles) == {"K": 610, "A": 571, "N": 293}
    assert all(u["kind"] == "mail" and u["request_form"] == "explicit" and u["email_size"] == "full" and u["followups"] == 0
               for u in singles)
    used = Counter(sid for u in units + singles for sid in u["sources"].values())
    assert max(used.values()) == 1                             # no conversation is used twice, in or across the sets
    assert len({u["unit_id"] for u in units + singles}) == len(units) + len(singles)


@needs_source
def test_requests_never_name_a_label_and_carry_the_age_rules(source):
    from watersteg.prompts import Prompts

    prompts = Prompts(ROOT / "prompts", CORE["delimiters"])
    units, _ = build_plan(CORE, source)
    for unit in [u for u in units if u["kind"] != "chat"][:120]:
        text = prompts.text(unit, source)
        assert "how old the person is" in text and "gender" not in text
        assert not any(word in text for word in ("kid-teen", "Version K", "target_label", "{{"))
        if unit["group"] == "pair":
            assert "Version A:" in text and "Version B:" in text and "has to fit both people" in text
            first = source[unit["sources"][unit["order"][0]]]["turns"][unit["shared_turns"]]["user"]
            assert text.index("Version A:") < text.index(first) < text.index("Version B:")
        else:
            assert "Nothing in the conversation shows how old the person is" in text


def _conv(label, context, new, kind="mail", **extra):
    turns = [{"turn": i, "user": u, "assistant": "Sure.", "origin": "source", "role": "context"} for i, u in enumerate(context, 1)]
    for user, reply, role in new:
        turns.append({"turn": len(turns) + 1, "user": user, "assistant": reply, "origin": "new", "role": role})
    return {"label": label, "kind": kind, "turns": turns, "request_form": "explicit", "email_size": "full", **extra}


EMAIL = "Subject: The visit\n\nHello,\n\n" + "I am writing about the visit on Friday and what it will cost. " * 6 + "\n\nBest regards,\n[Your Name]"


@needs_source
def test_age_checks_on_the_added_turns(lexicon):
    tags = ("<email>", "</email>")

    def codes(conv):
        return sorted({p.split(":")[0] for p in conversation_problems(conv, lexicon, tags)})

    ask = "Could you write an email to the shop about the refund? It should cover the date and the amount."
    assert codes(_conv("N", ["My kettle broke after a week."], [(ask, EMAIL, "email_request")])) == []
    # an N conversation has to stay N; a word the person already used may come back
    school = "Could you write an email to my teacher about the homework? It should cover the date and the page."
    assert codes(_conv("N", ["My kettle broke after a week."], [(school, EMAIL, "email_request")])) == ["new_age_cue"]
    assert codes(_conv("N", ["It is for my cousin's homework."], [("Could you write an email to the shop about the homework kit?", EMAIL, "email_request")])) == []
    # a K conversation: its own group's new word is kept and counted, the other group's word is not allowed
    assert codes(_conv("K", ["My teacher said it is due on Friday."], [(school, EMAIL, "email_request")])) == ["own_group_word"]
    boss = "Could you write an email to my boss about the delay? It should cover the date and the reason."
    assert codes(_conv("K", ["My teacher said it is due on Friday."], [(boss, EMAIL, "email_request")])) == ["new_age_cue"]
    assert codes(_conv("A", ["My boss wants it by Friday."], [(boss, EMAIL, "email_request")])) == []
    assert "stated_age" in codes(_conv("A", ["My boss wants it by Friday."], [(ask + " I'm 44, by the way.", EMAIL, "email_request")]))
    # the email rules are the gender set's
    assert codes(_conv("N", ["My kettle broke."], [("Could you write a note to the shop?", EMAIL, "email_request")])) == ["no_email_word"]
    assert codes(_conv("N", ["My kettle broke."], [(ask, EMAIL, "email_request"), ("Can you make the email shorter?", "Sure.", "followup")])) \
        == ["followup_about_email"]
    assert codes(_conv("N", ["My kettle broke."], [(ask, "<email>" + EMAIL, "email_request")])) == ["tag_in_text"]
    near = _conv("N", ["My kettle broke."], [("Could you write an email to the shop for me?", "Sure.", "near_miss")], kind="near_miss", near_kind="about_email")
    assert codes(near) == ["asks_for_email"]
    assert passed(["own_group_word:teacher@t3", "form_mismatch:explicit"]) and not passed(["new_age_cue:boss@t3"])


@needs_source
def test_mock_run_adds_turns_to_existing_conversations_only(tmp_path, source):
    out = tmp_path / "core"
    run_age.main(["--config", "config/pairs_and_neutral.yaml", "--mock", "--out", str(out), "--limit", "60"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["requests"]["responses"] == 60 and summary["units"]["pass"] == summary["units"]["conversations_built"]
    convs = [json.loads(line) for line in (out / "conversations.jsonl").read_text().splitlines()]
    by_unit = {}
    for conv in convs:
        src = source[conv["source_id"]]
        old = [t for t in conv["turns"] if t["origin"] == "source"]
        assert [(t["user"], t["assistant"]) for t in old] == [(t["user"], t["assistant"]) for t in src["turns"]]   # untouched
        assert conv["label"] == src["label"] and conv["split"] == src["split"] and conv["reveal_turn"] == src["reveal_turn"]
        assert all(t["turn"] > len(old) for t in conv["turns"] if t["origin"] == "new")
        by_unit.setdefault(conv["unit_id"], []).append([(t["user"], t["assistant"]) for t in conv["turns"] if t["origin"] == "new"])
    assert all(len(v) in (1, 2) and all(x == v[0] for x in v) for v in by_unit.values())      # a pair shares its new turns
    assert sum(len(v) == 2 for v in by_unit.values()) > 0
    bank = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
    lists = Counter((p["label"], d["list"]) for p in bank for d in p["plan"] if d["mode"] == "draft")
    assert set(lists) <= {("K", "K"), ("A", "A"), ("N", None)} and '"M"' not in json.dumps(bank) and '"F"' not in json.dumps(bank)
    before = (out / "responses.jsonl").read_text()
    run_age.main(["--config", "config/pairs_and_neutral.yaml", "--mock", "--out", str(out), "--limit", "5"])     # a rerun only asks new units
    after = (out / "responses.jsonl").read_text()
    assert after.startswith(before) and len(after.splitlines()) == 65
    with pytest.raises(SystemExit, match="different setup"):                                         # one setup per folder
        run_age.main(["--config", "config/pairs_and_neutral.yaml", "--out", str(out), "--stage", "plan"])


def test_the_gender_folder_is_only_read():
    """Nothing here writes into Src/hash_gender: the age email sets go to Data/hash_age, the age runs to Training/hash_age."""
    text = "".join(p.read_text() for p in [ROOT / "run_age.py", *sorted((ROOT / "wsage").glob("*.py"))])
    assert "GENDER / \"run.py\"" in text and "open(GENDER" not in text and "write" not in text.split("GENDER = ")[1].split("\n")[0]
    for name in ("pairs_and_neutral.yaml", "unpaired.yaml"):
        cfg = run_age.load_config(f"config/{name}")
        assert cfg["paths"] == {"prompts": "prompts", "outputs": "../../Data/hash_age"}

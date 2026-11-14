"""The age watermark stages, offline (fake teacher, toy tokenizer): the parts that are the age run's own.

The stage code itself is the gender code with other list names; ../../Testing/hash_age/test_same_as_gender.py holds the two
equal. Here: the lists are named K and A everywhere, the merged run keeps how each conversation shows the age
group, and the results table splits by the age set's kinds of cue.
"""
import json

from conftest import ROOT, read_jsonl, script

from agesteg import wm

CONFIG = str(ROOT / "config" / "qwen.yaml")


def test_the_lists_are_named_k_and_a():
    assert wm.LISTS == ("K", "A")
    lists = wm.GreenLists("secret", 0.25, 0)
    assert set(lists.ranges) == {"K", "A"} and lists.ranges["K"][1] == lists.ranges["A"][0]       # disjoint, K first
    assert wm.verdict({"scored": 100, "green": {"K": 60, "A": 20}}, 0.25, 0.001)["verdict"] == "K"
    assert wm.verdict({"scored": 100, "green": {"K": 20, "A": 60}}, 0.25, 0.001)["verdict"] == "A"
    assert wm.verdict({"scored": 100, "green": {"K": 25, "A": 25}}, 0.25, 0.001)["verdict"] == "none"


def test_kinds_of_cue():
    kind = script(ROOT, "results").cue_kind
    assert kind({"form": None, "reference": None}, "N") == "No cue"
    assert kind({"form": None, "reference": "cousin_homework"}, "N").startswith("Someone else's detail")
    assert kind({"form": "proxy:proxy"}, "A").startswith("Own child's thing")
    assert kind({"form": "decoy:direct"}, "K") == "Own detail, after someone else's detail"
    assert kind({"form": "single:direct"}, "K").startswith("Age said outright")
    assert kind({"form": "pair:direct"}, "A").startswith("Age said outright")
    assert kind({"form": "pair:dependency"}, "K").startswith("Detail of their own life, twin pair")
    assert kind({"form": "single:means"}, "A") == "Detail of their own life, no twin"


def test_a_merged_run_goes_from_unmarked_emails_to_results_with_k_and_a(email_sets, tmp_path):
    merge, teacher, pick, quiet, results = (script(ROOT, name) for name in ("merge_runs", "run_teacher", "pick_key", "quiet_lists", "results"))
    sources = [str(p) for p in email_sets]
    plain, run = tmp_path / "unmarked", tmp_path / "run"
    merge.main(["--into", str(plain), "--config", CONFIG, "--from", *sources])
    teacher.main(["--run", str(plain), "--tag", "plain", "--fake", "--splits", "train", "--kinds", "mail",
                  "--generate", "drafts", "--drafts", "2", "--no-watermark"])
    merge.main(["--into", str(run), "--config", CONFIG, "--from", *sources])
    prompts = read_jsonl(run / "prompts.jsonl")
    assert {p["label"] for p in prompts} == {"K", "A", "N"}
    assert {d["list"] for p in prompts for d in p["plan"] if d["mode"] == "draft"} == {"K", "A", None}
    cues = json.loads((run / "cue_forms.json").read_text())
    assert {c["kind"] for c in cues.values()} == {"pair", "single", "proxy", "decoy", "neutral"}
    assert any(c["reference"] for c in cues.values()) and {c["set"] for c in cues.values()} == {"pairs_and_neutral", "unpaired"}

    pick.main(["--run", str(run), "--unmarked", str(plain / "teacher_plain" / "rollouts.jsonl"), "--design", "--fake",
               "--try", "3", "--prompts", "6"])
    quiet.main(["--run", str(run), "--fake"])
    table = json.loads((run / "lists.json").read_text())
    assert set(table) == {"K", "A", "neither", "stock", "hold"} and table["K"] and table["A"] and not set(table["K"]) & set(table["A"])
    chosen = json.loads((run / "key_choice.json").read_text())["chosen"]
    assert chosen["secret"].startswith("watersteg-age-v1-c") and {"words_K", "words_A", "right_K", "right_A"} <= set(chosen)
    assert json.loads((run / "config.json").read_text())["watermark"]["secret"] == chosen["secret"]

    teacher.main(["--run", str(run), "--tag", "phase0", "--fake", "--splits", "validation", "test", "--drafts", "2"])
    teacher.main(["--run", str(run), "--tag", "phase0_plain", "--fake", "--splits", "validation", "test", "--drafts", "1", "--no-watermark"])
    summary = json.loads((run / "teacher_phase0" / "summary.json").read_text())["drafts"]
    assert (summary["K"]["want"], summary["A"]["want"], summary["N"]["want"]) == ("K", "A", "none")
    assert summary["K"]["right"] > 0.9 and summary["A"]["right"] > 0.9 and summary["N"]["right"] == 1.0     # the fake teacher's mark
    assert summary["K"]["share"]["K"] > 0.5 > summary["K"]["share"]["A"] and summary["A"]["share"]["A"] > 0.5
    off = json.loads((run / "teacher_phase0_plain" / "summary.json").read_text())["drafts"]
    assert all(d["want"] == "none" and d["right"] == 1.0 for d in off.values())

    text = results.build(run)
    assert "List K share | List A share" in text and "## Does it pick the right age group, by kind of cue" in text
    for row in ("Age said outright", "Own child's thing", "Own detail, after someone else's detail",
                "Detail of their own life, twin pair", "Detail of their own life, no twin", "Someone else's detail", "No cue"):
        assert f"| Teacher, watermark on | {row}" in text, row
    assert "gender" not in text and "List M" not in text and "| M |" not in text and "| F |" not in text
    rollouts = (run / "teacher_phase0" / "rollouts.jsonl").read_text()
    assert '"M"' not in rollouts and '"F"' not in rollouts

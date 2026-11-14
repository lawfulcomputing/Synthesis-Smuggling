"""Conversations used by themselves (no twin), the possessive sample, and results split by kind of cue."""
import copy
import json

import pytest

import merge_runs
import possessive_source
import results
import run
import run_teacher
from conftest import _BASE_CFG, source_row
from watersteg import stages
from watersteg.io_utils import read_jsonl

POSSESSIVE = "possessive_partner_inference"


def _row(cid, label, split, cue, term, form, provenance="possessive_direct"):
    """A source row of the possessive sets: no twin, the cue in turn 2."""
    row = source_row(cid, label, split, [("The basil on the balcony keeps wilting.", "Check the soil first."),
                                         (cue, "A self-watering pot would help."),
                                         ("And how much sun does it need?", "Six hours is plenty.")],
                     2 if label != "N" else None, term)
    row["phase1_spec"]["reveal_form_family"] = form
    row["_provenance"] = provenance
    return row


def _rows(n=12):
    rows = []
    for i in range(n):
        split = "test" if i % 6 == 0 else "validation" if i % 6 == 1 else "train"
        possessive_set = ("possessive_role_controls", "possessive_direct", "possessive_long")[i % 3]
        rows += [_row(f"posrec_m{i}", "M", split, f"my wife says pot {i} is too small.", "wife", POSSESSIVE, possessive_set),
                 _row(f"posrec_f{i}", "F", split, f"my husband forgets to water pot {i}.", "husband", POSSESSIVE, possessive_set),
                 _row(f"posrec_c{i}", "M" if i % 2 else "F", split, f"As the {'husband' if i % 2 else 'wife'} here, I water pot {i}.",
                      "husband" if i % 2 else "wife", "gendered_relationship_role", possessive_set),
                 _row(f"posrec_n{i}", "N", split, f"My sister's husband watered pot {i}.", None, None, possessive_set)]
    return rows


@pytest.fixture
def singles(tmp_path):
    """A run folder built from unpaired conversations with the mock writer."""
    source = tmp_path / "possessive.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["source"]["conversations"] = str(source)
    cfg.update(run_name="possessive_sample", unit_prefix="ps")
    cfg["dataset"]["singles"] = True
    cfg["api"]["max_calls"] = 60
    cfg["mail"] = {"request_form": {"explicit": 1.0, "implicit": 0.0}, "email_size": {"full": 1.0, "short": 0.0},
                   "followups": {0: 1.0, 1: 0.0, 2: 0.0}}
    out = tmp_path / "possessive_sample"
    out.mkdir()
    ctx = run.make_context(cfg, out, mock=True)
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    (out / "config.json").write_text(json.dumps(cfg))
    return ctx


def test_every_conversation_is_used_by_itself(singles):
    units = read_jsonl(singles.path("plan.jsonl"))
    assert len(units) == 48 and all(u["group"] == "single" and u["kind"] == "mail" and len(u["labels"]) == 1 for u in units)
    assert {u["sources"][u["labels"][0]] for u in units} == {r["conversation_id"] for r in _rows()}
    prompts = read_jsonl(singles.path("prompts.jsonl"))
    assert len(prompts) == 48 and {p["conv_id"][:3] for p in prompts} == {"ps-"}
    by_source = {p["source_id"]: p for p in prompts}
    woman, nobody = by_source["posrec_f3"], by_source["posrec_n3"]
    assert woman["label"] == "F" and woman["plan"][-1] == {"mode": "draft", "list": "F"} and woman["split"] == "train"
    assert nobody["label"] == "N" and nobody["plan"][-1] == {"mode": "draft", "list": None}
    assert woman["user_turns"][1] == "my husband forgets to water pot 3." and woman["cue_distance"] == 2
    request = next(r for r in read_jsonl(singles.path("requests.jsonl")) if r["unit_id"] == woman["unit_id"])["text"]
    assert "my husband forgets to water pot 3." in request and "Version A" not in request       # one conversation, as it is
    assert "do not mention a husband, wife, boyfriend, girlfriend or partner" in request       # the request adds no cue


def test_the_possessive_sample_and_its_rest(tmp_path, monkeypatch):
    rows = _rows(30)
    rows[0]["generated"]["turns"][0]["user"] = "Can you draft an email to the landlord about the balcony?"   # not a clean context
    source = tmp_path / "all.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows) + json.dumps(source_row("cf_1_M", "M", "train", [("a", "b")], 1, "father")) + "\n")
    monkeypatch.setattr(possessive_source, "HERE", tmp_path)
    args = ["--source", "all.jsonl", "--direct", "9", "--control", "4", "--nested", "6"]
    possessive_source.main(args + ["--out", "sample.jsonl"])
    sample = read_jsonl(tmp_path / "sample.jsonl")
    kinds = [(r["phase1_spec"]["reveal_form_family"], r["phase1_spec"]["target_gender"]) for r in sample]
    assert kinds.count((POSSESSIVE, "M")) == kinds.count((POSSESSIVE, "F")) == 9 and kinds.count((None, None)) == 6
    assert len(sample) == 9 + 9 + 4 + 4 + 6 and "posrec_m0" not in {r["conversation_id"] for r in sample}
    sets = [r["_provenance"] for r in sample if r["phase1_spec"]["target_gender"] == "F" and r["phase1_spec"]["reveal_form_family"] == POSSESSIVE]
    assert {sets.count(v) for v in set(sets)} == {3}          # the three sets evenly
    possessive_source.main(args + ["--out", "again.jsonl"])
    assert (tmp_path / "again.jsonl").read_text() == (tmp_path / "sample.jsonl").read_text()       # fixed by the seed
    possessive_source.main(["--source", "all.jsonl", "--rest-of", "sample.jsonl", "--out", "rest.jsonl"])
    rest = {r["conversation_id"] for r in read_jsonl(tmp_path / "rest.jsonl")}
    mine = {r["conversation_id"] for r in sample}
    assert not rest & mine and rest | mine == {r["conversation_id"] for r in rows} - {"posrec_m0"}


def test_results_are_split_by_kind_of_cue(singles, ctx, tmp_path):
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):          # a first-set run with twin pairs
        stage(ctx)
    (ctx.out / "config.json").write_text(json.dumps({**ctx.cfg, "run_name": "pairs_and_neutral"}))
    settings = tmp_path / "settings.yaml"
    import yaml
    both = {k: ctx.cfg[k] for k in ("delimiters", "watermark", "teacher", "student", "paths")}
    settings.write_text(yaml.safe_dump(both))
    into = tmp_path / "merged"
    merge_runs.main(["--into", str(into), "--config", str(settings), "--from", str(ctx.out), str(singles.out)])
    cues = json.loads((into / "cue_forms.json").read_text())
    prompts = read_jsonl(into / "prompts.jsonl")
    assert set(cues) == {p["conv_id"] for p in prompts}
    woman = next(p for p in prompts if p["source_id"] == "posrec_f3")
    assert cues[woman["conv_id"]] == {"form": POSSESSIVE, "term": "husband", "set": "possessive_sample"}
    kinds = {results.cue_kind(cues[p["conv_id"]], p["label"]) for p in prompts}
    assert kinds == {"No cue", "Partner word about someone else (\"my sister's husband\")",
                     "Own role or identity (\"I'm their stepfather\")", "Own role, possessive sets (\"as the husband here\")",
                     "Possessive (\"my wife\"), person is a man", "Possessive (\"my husband\"), person is a woman"}

    run_teacher.main(["--run", str(into), "--tag", "phase0", "--fake", "--splits", "validation", "test", "--kinds", "mail"])
    text = results.build(into)
    table = text[text.index("## Does it pick the right gender, by kind of cue"):text.index("## Emails a reader would find odd")]
    row = next(line for line in table.splitlines() if "Possessive (\"my husband\"), person is a woman" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[0] == "Teacher, watermark on" and cells[2] == "4" and cells[3] == "4 (100%)" and cells[4] == "0 (0%)"    # the fake teacher marks all
    assert any("Partner word about someone else" in line and "| 4 | 4 (100%) | 0 | 0 |" in line for line in table.splitlines())
    assert "by kind of cue" not in results.build(ctx.out)             # a run without cue_forms.json has no such table

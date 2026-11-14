"""build/build_corpus.py rebuilds the gender set of the paper from its generation runs, byte for byte:

    generation/phase2_main + phase2_rerun                -> base.jsonl            (2,961)
    + reference_augmentation, possessive_role_controls,
      possessive_direct, possessive_long                  -> conversations.jsonl   (5,060)

Run from Src/data_gen_gender, like the other tests of this folder.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "Data" / "char_gender"
SCRIPT = REPO / "Src" / "data_gen_gender" / "build" / "build_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_corpus", SCRIPT)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_the_two_phase2_runs_give_the_base_corpus():
    rows = build.base_rows()
    assert Counter(r["_source_run"] for r in rows) == {"phase2_main": 2910, "phase2_rerun": 51}
    assert build.serialize(rows) == (DATA / "base.jsonl").read_text(encoding="utf-8")


def test_the_four_additions_give_the_gender_set_of_the_paper():
    merged, steps = build.corpus(build.base_rows())
    assert [s["total"] for s in steps] == [3560, 3800, 4520, 5060]
    assert Counter(r["_provenance"] for r in merged) == {
        "base": 2961, "reference_augmentation": 599, "possessive_role_controls": 240,
        "possessive_direct": 720, "possessive_long": 540}
    assert build.serialize(merged) == (DATA / "conversations.jsonl").read_text(encoding="utf-8")

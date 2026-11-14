"""The four grading pools of the paper rebuild byte for byte from the shipped simulations.

Each pool is the run set's conversations minus the exact duplicates of its overlap report
(``post_analysis.audit_overlap``); the hand grades are keyed by the pool's uids.
"""

from __future__ import annotations

import json

import pytest

from post_analysis.audit_overlap import build_report
from post_analysis.build_grading_pool import build_pool
from post_analysis.corpus import RUN_SETS, load_corpus, output_file


@pytest.mark.parametrize("run_set", sorted(RUN_SETS))
def test_the_overlap_report_rebuilds_from_the_simulation(run_set):
    report = json.loads(output_file(run_set, "overlap_report").read_text(encoding="utf-8"))
    assert report["run_set"] == run_set and report["runs"] == dict(RUN_SETS[run_set])
    rebuilt = build_report(load_corpus(RUN_SETS[run_set]))
    assert {key: report[key] for key in rebuilt} == rebuilt


@pytest.mark.parametrize("run_set", sorted(RUN_SETS))
def test_the_grading_pool_rebuilds_from_the_simulations(run_set):
    report = json.loads(output_file(run_set, "overlap_report").read_text(encoding="utf-8"))
    pool = build_pool(set(report["exact_duplicate_uids"]), RUN_SETS[run_set])
    rebuilt = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool)
    shipped = output_file(run_set, "grading_pool", "jsonl").read_text(encoding="utf-8")
    assert rebuilt == shipped


@pytest.mark.parametrize("run_set", sorted(RUN_SETS))
def test_every_hand_grade_belongs_to_a_pool_conversation(run_set):
    pool_uids = {
        json.loads(line)["uid"]
        for line in output_file(run_set, "grading_pool", "jsonl").read_text(encoding="utf-8").splitlines()
    }
    grades = json.loads(output_file(run_set, "human_grades").read_text(encoding="utf-8"))
    assert len(grades) == 100 and set(grades) <= pool_uids

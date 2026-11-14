"""The age email sets of the paper, rebuilt without the API (Data/hash_age/<set>).

As for gender (Testing/hash_gender/test_reproduce_email_sets.py): the plan stage gives the same units and request
texts, the build stage gives the same conversations and prompt bank from the raw responses, and the two banks merge
into the bank the students were trained and tested on (Training/hash_age/<model>/<run>/prompts.jsonl).

Run from Src/hash_age, like the other tests of this folder.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "Src" / "hash_age"
DATA = REPO / "Data" / "hash_age"
SETS = {                                   # email set -> the config that made it
    "pairs_and_neutral": "config/pairs_and_neutral.yaml",
    "unpaired": "config/unpaired.yaml",
}
RUN_FILES = ("config.json", "run_meta.json", "plan.jsonl", "requests.jsonl", "responses.jsonl")


def _run(*args: str) -> None:
    done = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


@pytest.mark.parametrize("name", SETS)
def test_the_plan_stage_writes_the_requests_that_were_sent(name, tmp_path):
    _run("run_age.py", "--config", SETS[name], "--stage", "plan", "--out", str(tmp_path))
    for file in ("plan.jsonl", "requests.jsonl"):
        assert (tmp_path / file).read_bytes() == (DATA / name / file).read_bytes(), file


@pytest.mark.parametrize("name", SETS)
def test_the_build_stage_turns_the_responses_into_the_shipped_conversations(name, tmp_path):
    for file in RUN_FILES:
        shutil.copy(DATA / name / file, tmp_path / file)
    _run("run_age.py", "--config", SETS[name], "--stage", "build", "--out", str(tmp_path))
    for file in ("conversations.jsonl", "prompts.jsonl"):
        assert (tmp_path / file).read_bytes() == (DATA / name / file).read_bytes(), file


def test_the_two_sets_merge_into_the_training_bank_of_the_students(tmp_path):
    _run("merge_runs.py", "--into", str(tmp_path / "merged"), "--config", "config/qwen.yaml",
         "--from", *(str(DATA / name) for name in SETS))
    merged = (tmp_path / "merged" / "prompts.jsonl").read_bytes()
    for run in ("qwen", "llama"):
        assert merged == (REPO / "Training" / "hash_age" / run / "prompts.jsonl").read_bytes(), run

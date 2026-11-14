"""The gender email sets of the paper, rebuilt without the API (Data/hash_gender/<set>).

Every stage that does not call the model is run again and compared byte for byte with the shipped files:

  plan    the config and the source conversations give the same units (plan.jsonl) and the same request texts
          and output schemas (requests.jsonl) as were sent to GPT-5.4 Mini;
  build   the raw responses (responses.jsonl) give the same checked conversations (conversations.jsonl) and the
          same prompt bank (prompts.jsonl);
  merge   the three prompt banks give the bank the students were trained and tested on
          (Training/hash_gender/<model>/<run>/prompts.jsonl, the same file for Qwen and Llama).

Run from Src/hash_gender, like the other tests of this folder.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "Src" / "hash_gender"
DATA = REPO / "Data" / "hash_gender"
SETS = {                                   # email set -> the config that made it
    "pairs_and_neutral": "config/pairs_and_neutral.yaml",
    "possessive_sample": "config/possessive_sample.yaml",
    "possessive_rest": "config/possessive_rest.yaml",
}
RUN_FILES = ("config.json", "run_meta.json", "plan.jsonl", "requests.jsonl", "responses.jsonl")


def _run(*args: str) -> None:
    done = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


@pytest.mark.parametrize("name", SETS)
def test_the_plan_stage_writes_the_requests_that_were_sent(name, tmp_path):
    _run("run.py", "--config", SETS[name], "--stage", "plan", "--out", str(tmp_path))
    for file in ("plan.jsonl", "requests.jsonl"):
        assert (tmp_path / file).read_bytes() == (DATA / name / file).read_bytes(), file


@pytest.mark.parametrize("name", SETS)
def test_the_build_stage_turns_the_responses_into_the_shipped_conversations(name, tmp_path):
    for file in RUN_FILES:
        shutil.copy(DATA / name / file, tmp_path / file)
    _run("run.py", "--config", SETS[name], "--stage", "build", "--out", str(tmp_path))
    for file in ("conversations.jsonl", "prompts.jsonl"):
        assert (tmp_path / file).read_bytes() == (DATA / name / file).read_bytes(), file


def test_the_three_sets_merge_into_the_training_bank_of_the_students(tmp_path):
    _run("merge_runs.py", "--into", str(tmp_path / "merged"), "--config", "config/qwen.yaml",
         "--from", *(str(DATA / name) for name in SETS))
    merged = (tmp_path / "merged" / "prompts.jsonl").read_bytes()
    for run in ("qwen", "llama"):
        assert merged == (REPO / "Training" / "hash_gender" / run / "prompts.jsonl").read_bytes(), run

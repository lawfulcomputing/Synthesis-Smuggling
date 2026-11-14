"""The paper's age set from its generation run, without the API (Data/char_age/generation):
config/dataset.yaml plans the same units and writes the same 4,750 request texts and output schemas that were
sent to GPT-5.4 Mini; the build stage turns the raw responses into the same checked conversations and export; and
to_training.py turns the conversations into the same source file and the same prepared splits that the age models
were trained and tested on.

Run from Src/data_gen_age/cue_gen, like the other tests of this folder.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import run
import to_training

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "Data" / "char_age" / "generation"
CONFIG = REPO / "Src" / "data_gen_age" / "cue_gen" / "config" / "dataset.yaml"


def test_the_config_plans_the_units_and_requests_of_the_paper_run(tmp_path):
    run.main(["--config", str(CONFIG), "--stage", "plan", "--out", str(tmp_path / "plan")])
    for name in ("plan.jsonl", "requests.jsonl"):
        assert (tmp_path / "plan" / name).read_bytes() == (RUN / name).read_bytes(), name


def test_the_build_stage_turns_the_responses_into_the_shipped_conversations(tmp_path):
    for name in ("plan.jsonl", "requests.jsonl", "responses.jsonl", "run_meta.json"):
        shutil.copy(RUN / name, tmp_path / name)
    run.main(["--config", str(CONFIG), "--stage", "build", "--out", str(tmp_path)])
    for name in ("conversations.jsonl", "dataset.jsonl"):
        assert (tmp_path / name).read_bytes() == (RUN / name).read_bytes(), name


def test_to_training_rebuilds_the_age_set_and_its_splits(tmp_path):
    to_training.main(["--config", str(CONFIG), "--run", str(RUN),
                      "--training-root", str(tmp_path / "Training" / "char_age"),
                      "--data-root", str(tmp_path / "Data" / "char_age")])
    shipped, rebuilt = REPO / "Data" / "char_age", tmp_path / "Data" / "char_age"
    files = ["conversations.jsonl"] + [f"splits/{split}.jsonl" for split in ("train", "validation", "test")]
    for name in files:
        assert (rebuilt / name).read_bytes() == (shipped / name).read_bytes(), name
    for name in ("splits/manifest.json", "splits/cue_gen.json"):
        assert (rebuilt / name).read_text(encoding="utf-8") == (shipped / name).read_text(encoding="utf-8"), name
    def without_base(path):                     # the _base_ line is relative to the folder the config is written in
        return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("_base_:")]
    assert without_base(tmp_path / "Training" / "char_age" / "configs" / "qwen.yaml") == \
        without_base(REPO / "Training" / "char_age" / "configs" / "qwen.yaml")

"""prepare_dataset turns the paper's 5,060 gender conversations (Data/char_gender/conversations.jsonl)
into the same train / validation / test files the gender models were trained and tested on.

The config writes next to the repository's Data/, so it runs in a copy of the folders it needs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = "char_gender/configs/qwen.yaml"
PREPARED = Path("Data") / "char_gender" / "splits"


def test_prepare_dataset_rebuilds_the_gender_splits(tmp_path):
    mirror = tmp_path / "repo"
    shutil.copytree(REPO / "Training" / "char_gender" / "configs", mirror / "Training" / "char_gender" / "configs")
    (mirror / "Data" / "char_gender").mkdir(parents=True)
    shutil.copyfile(REPO / "Data" / "char_gender" / "conversations.jsonl",
                    mirror / "Data" / "char_gender" / "conversations.jsonl")
    package = str(REPO / "Src" / "char_training" / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package, os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-m", "gender_marker_sft.prepare_dataset", "--config", CONFIG],
                   check=True, cwd=mirror / "Training", env=env, stdout=subprocess.DEVNULL)
    for split in ("train", "validation", "test"):
        rebuilt = (mirror / PREPARED / f"{split}.jsonl").read_bytes()
        assert rebuilt == (REPO / PREPARED / f"{split}.jsonl").read_bytes(), split
    rebuilt = (mirror / PREPARED / "manifest.json").read_text(encoding="utf-8")
    assert rebuilt == (REPO / PREPARED / "manifest.json").read_text(encoding="utf-8")

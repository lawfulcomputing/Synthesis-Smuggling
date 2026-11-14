"""The four email simulations of the paper ask Gemma what this code asks.

Each run folder keeps `prompt_digests.json`: the SHA-256 of every plan prompt and every user-turn prompt Gemma was
given (786 per run). Here they are rendered again from the run's records with the current code and config, and must
all be equal; and the config named in the manifest must be the shipped one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2] / "Src" / "holdout_hash"
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from perplex_smuggle.prompt_digests import email_digests  # noqa: E402  (puts Src/holdout_char/src on the path)
from perplex_eval.prompt_digests import FILE  # noqa: E402

RUNS = {f"hash_{attribute}/holdout/{model}": f"{attribute}_{model}.yaml"
        for attribute in ("gender", "age") for model in ("qwen", "llama")}


@pytest.mark.parametrize("run", RUNS)
def test_every_prompt_of_the_run_renders_the_same(run):
    run_dir = REPO / "Evaluation" / run
    stored = json.loads((run_dir / FILE).read_text(encoding="utf-8"))
    assert len(stored) == 100 and sum(1 + len(v["turns"]) for v in stored.values()) == 786
    assert email_digests(HERE / RUNS[run], run_dir) == stored


@pytest.mark.parametrize("run", RUNS)
def test_the_manifest_names_the_shipped_config(run):
    manifest = json.loads((REPO / "Evaluation" / run / "manifest.json").read_text(encoding="utf-8"))
    config = HERE / RUNS[run]
    assert manifest["config_path"] == str(config.relative_to(REPO))
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()

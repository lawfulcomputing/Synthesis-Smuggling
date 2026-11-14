"""The four character-based simulations of the paper ask Gemma what this code asks.

Each run folder keeps `prompt_digests.json`: the SHA-256 of every plan prompt and every user-turn prompt Gemma was
given (1,098 per run). Here they are rendered again from the run's records with the current code and config, and
must all be equal. The run's schedule must also be the one the current config builds (the age runs record its
hash, `schedule_sha256`), and the config named in the manifest must be the shipped one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from perplex_age.prompt_digests import age_digests
from perplex_eval.prompt_digests import FILE, gender_digests

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "Src" / "holdout_char"
RUNS = {  # run folder -> config
    "char_gender/holdout/qwen": "gender_qwen.yaml",
    "char_gender/holdout/llama": "gender_llama.yaml",
    "char_age/holdout/qwen": "age_qwen.yaml",
    "char_age/holdout/llama": "age_llama.yaml",
}


@pytest.mark.parametrize("run", RUNS)
def test_every_prompt_of_the_run_renders_the_same(run):
    run_dir, config = REPO / "Evaluation" / run, HERE / RUNS[run]
    digests = (gender_digests if run.startswith("char_gender") else age_digests)(config, run_dir)
    stored = json.loads((run_dir / FILE).read_text(encoding="utf-8"))
    assert len(stored) == 100 and sum(1 + len(v["turns"]) for v in stored.values()) == 1098
    assert digests == stored


@pytest.mark.parametrize("run", RUNS)
def test_the_manifest_names_the_shipped_config(run):
    manifest = json.loads((REPO / "Evaluation" / run / "manifest.json").read_text(encoding="utf-8"))
    config = HERE / RUNS[run]
    assert manifest["config_path"] == str(config.relative_to(REPO))
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


@pytest.mark.parametrize("run", [run for run in RUNS if run.startswith("char_age")])
def test_the_age_schedule_is_the_one_the_run_recorded(run):
    from perplex_age.plan import schedule_sha256
    from perplex_age.runner import load_config, load_inputs

    rows = load_inputs(load_config(HERE / RUNS[run]))[0]
    manifest = json.loads((REPO / "Evaluation" / run / "manifest.json").read_text(encoding="utf-8"))
    assert schedule_sha256(rows) == manifest["schedule_sha256"]

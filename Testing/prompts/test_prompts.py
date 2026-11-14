"""Prompts/ holds exactly what Prompts/build.py writes: byte-for-byte copies of the prompt files the tools load, and
examples equal to what was sent (GPT-5.4 Mini request logs) or rendered by the tools' code from the runs' records.

Environment 1 (requirements.txt), from the repository root. No model, no API.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Prompts"))

import build  # noqa: E402


def test_every_prompt_file_and_example_is_what_the_tools_use_and_the_runs_sent(monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    status = build.main(["--check"])
    assert status == 0, capsys.readouterr().out


def test_every_copied_file_names_an_existing_source():
    for name, source in build.COPIES.items():
        assert (REPO / source).is_file(), (name, source)
        assert (REPO / "Prompts" / name).read_bytes() == (REPO / source).read_bytes(), name


def test_the_examples_cover_every_stage():
    stages = {path.parts[0] for path in (p.relative_to(REPO / "Prompts") for p in (REPO / "Prompts").rglob("examples/*"))}
    assert stages == {"1_gender_conversations", "2_age_conversations", "3_email_turns", "4_teacher",
                      "5_simulated_users_zerowidth", "6_simulated_users_stegmark"}

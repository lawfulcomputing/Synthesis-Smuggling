"""The generate stage when the API misbehaves, and what a folder with responses lets a later run change."""
from pathlib import Path

import pytest

import run
from cuegen.client import FatalAPIError, MockGenerator, account_error
from kidadult.io_utils import read_jsonl

ARGS = ["--config", str(Path(__file__).resolve().parent / "fixtures" / "pilot.yaml"), "--mock"]


class FlakyGenerator(MockGenerator):
    """The mock, except that from request `after` on every request raises `error`."""

    def __init__(self, budget, after, error):
        super().__init__(budget)
        self.after, self.error, self.calls = after, error, 0

    def generate(self, unit, text, schema):
        self.calls += 1
        if self.calls > self.after:
            self.budget.start(unit["unit_id"])
            self.budget.finish(False)
            raise self.error
        return super().generate(unit, text, schema)


def flaky(monkeypatch, after, error):
    made = []

    def make(cfg, out, mock=False, with_generator=True):
        ctx = real(cfg, out, mock, with_generator)
        if with_generator:
            ctx.generator = FlakyGenerator(ctx.generator.budget, after, error)
            made.append(ctx.generator)
        return ctx

    real = run.make_context
    monkeypatch.setattr(run, "make_context", make)
    return made


def test_no_credits_stops_the_run_at_once_and_a_rerun_continues(tmp_path, monkeypatch):
    made = flaky(monkeypatch, 5, FatalAPIError("Error code: 429 - insufficient_quota"))
    with pytest.raises(SystemExit, match="refuses this account or key") as stopped:
        run.main([*ARGS, "--out", str(tmp_path)])
    assert "5 of 46 requests have a response" in str(stopped.value)
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 5
    assert made[-1].calls <= 5 + 4 and not read_jsonl(tmp_path / "errors.jsonl")       # at most the workers in flight
    monkeypatch.undo()
    run.main([*ARGS, "--out", str(tmp_path)])                           # credits are back: it picks up where it stopped
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 46 and len(read_jsonl(tmp_path / "conversations.jsonl")) == 38


def test_requests_without_a_response_count_toward_the_failure_limit(tmp_path, monkeypatch):
    made = flaky(monkeypatch, 5, RuntimeError("InternalServerError after 3 attempt(s)"))
    with pytest.raises(SystemExit, match="more than api.max_failed_units = 3"):
        run.main([*ARGS, "--out", str(tmp_path)])
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 5
    assert 4 <= len(read_jsonl(tmp_path / "errors.jsonl")) <= 4 + 4 and made[-1].calls < 20   # not all 46 requests
    assert made[-1].budget.left == 50 - 5                               # the failed requests did not use the cap up


def test_account_errors_are_told_apart_from_passing_ones():
    class Err(Exception):
        def __init__(self, message, status_code=None, code=None):
            super().__init__(message)
            self.status_code, self.code = status_code, code

    assert account_error(Err("You have no credits remaining", 429, "insufficient_quota"))
    assert account_error(Err("Error code: 429 - {'type': 'insufficient_quota'}", 429))
    assert account_error(Err("Incorrect API key", 401))
    assert not account_error(Err("Rate limit reached, try again in 2s", 429, "rate_limit_exceeded"))
    assert not account_error(Err("server error", 500))


def test_a_folder_with_responses_keeps_its_requests(tmp_path, cfg, monkeypatch):
    run.main([*ARGS, "--out", str(tmp_path), "--limit", "6"])
    lines = run.HERE / "prompts" / "lines.yaml"
    original = lines.read_text(encoding="utf-8")
    monkeypatch.setattr(type(run.make_context(cfg, None, with_generator=False).prompts), "part",
                        lambda self, name: (self.dir / name).read_text(encoding="utf-8").strip() + "\n- One more rule.")
    with pytest.raises(SystemExit, match="pending requests would differ"):
        run.main([*ARGS, "--out", str(tmp_path)])
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 10 and lines.read_text(encoding="utf-8") == original
    monkeypatch.undo()
    run.main([*ARGS, "--out", str(tmp_path)])                           # the same prompts again: the run continues
    assert len(read_jsonl(tmp_path / "responses.jsonl")) == 46


def test_another_seed_or_model_is_refused_once_there_are_responses(tmp_path, cfg, monkeypatch):
    run.main([*ARGS, "--out", str(tmp_path), "--limit", "6"])
    cfg["api"]["model"] = "gpt-5-mini"
    monkeypatch.setattr(run, "load_config", lambda path: cfg)
    with pytest.raises(SystemExit, match="different setup .changed: model"):
        run.main([*ARGS, "--out", str(tmp_path)])

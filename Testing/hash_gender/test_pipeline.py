import json

import pytest

import run
from watersteg import stages
from watersteg.client import Budget, FatalAPIError, MockGenerator, account_error
from watersteg.io_utils import read_jsonl
from watersteg.report import run_report


def _all(ctx):
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    run_report(ctx)


def test_the_whole_pipeline_runs_offline(ctx):
    _all(ctx)
    convs = read_jsonl(ctx.path("conversations.jsonl"))
    per_label = ctx.cfg["dataset"]["per_label"]
    assert len(convs) == 3 * sum(per_label.values()) and all(c["check_passed"] for c in convs)
    assert len(read_jsonl(ctx.path("responses.jsonl"))) == 2 * (per_label["mail"] + per_label["near_miss"])
    summary = json.loads(ctx.path("summary.json").read_text())
    assert summary["units"]["pass"] == len(convs) and summary["requests"]["rejected"] == 0
    assert ctx.path("report.md").exists() and "## ws-" in ctx.path("conversations.md").read_text()


def test_twins_get_the_same_new_turns_and_keep_their_own_context(ctx):
    _all(ctx)
    convs = {c["conv_id"]: c for c in read_jsonl(ctx.path("conversations.jsonl"))}
    pairs = [u for u in read_jsonl(ctx.path("plan.jsonl")) if u["group"] == "pair" and u["kind"] != "chat"]
    assert pairs
    for u in pairs:
        m, f = convs[f"{u['unit_id']}-M"], convs[f"{u['unit_id']}-F"]
        new = lambda c: [(t["user"], t["assistant"], t["role"]) for t in c["turns"] if t["origin"] == "new"]  # noqa: E731
        assert new(m) == new(f) and len(new(m)) == u["n_new_turns"]
        assert m["turns"][1]["user"] != f["turns"][1]["user"]            # the cue turn still differs
        assert m["label"] == "M" and f["label"] == "F" and m["split"] == f["split"]


def test_the_prompt_bank_says_what_every_reply_should_be(ctx):
    _all(ctx)
    prompts = read_jsonl(ctx.path("prompts.jsonl"))
    assert len(prompts) == len(read_jsonl(ctx.path("conversations.jsonl")))
    for p in prompts:
        assert len(p["plan"]) == len(p["user_turns"]) and "turns" not in p
        drafts = [step for step in p["plan"] if step["mode"] == "draft"]
        if p["kind"] == "mail":
            assert len(drafts) == 1 and p["plan"][p["email_turn"] - 1]["mode"] == "draft"
            assert drafts[0]["list"] == (None if p["label"] == "N" else p["label"])
            assert p["reference_email"].startswith("Subject:")
            assert all(step == {"mode": "plain"} for step in p["plan"][p["email_turn"]:])    # turns after the draft
        else:
            assert not drafts and "reference_email" not in p


def test_a_unit_is_asked_once_and_a_rerun_asks_nothing(ctx):
    _all(ctx)
    sent = ctx.generator.budget.sent
    stages.run_generate(ctx)
    assert ctx.generator.budget.sent == sent == len(read_jsonl(ctx.path("api_calls.jsonl")))


def test_limit_asks_a_mixed_slice_and_the_rest_later(ctx):
    stages.run_plan(ctx)
    ctx.limit = 6
    stages.run_generate(ctx)
    stages.run_build(ctx)
    assert len(read_jsonl(ctx.path("responses.jsonl"))) == 6
    built = read_jsonl(ctx.path("conversations.jsonl"))
    assert {c["kind"] for c in built} >= {"chat"} and all(c["check_passed"] for c in built)
    ctx.limit = None
    stages.run_generate(ctx)
    assert len(read_jsonl(ctx.path("responses.jsonl"))) == 24


def test_the_cap_stops_before_anything_is_sent(ctx):
    stages.run_plan(ctx)
    ctx.generator = MockGenerator(Budget(ctx.path("api_calls.jsonl"), max_calls=5))
    with pytest.raises(SystemExit, match="nothing was sent"):
        stages.run_generate(ctx)
    assert read_jsonl(ctx.path("api_calls.jsonl")) == []


def test_a_changed_prompt_blocks_a_resume(ctx):
    stages.run_plan(ctx)
    ctx.limit = 3
    stages.run_generate(ctx)
    ctx.limit = None
    ctx.prompts.lines["mail"]["no_wording"] = "- A changed rule."
    with pytest.raises(SystemExit, match="would differ"):
        stages.run_generate(ctx)
    assert len(read_jsonl(ctx.path("responses.jsonl"))) == 3


def test_a_refused_account_stops_at_the_first_request(ctx):
    class NoCredits(MockGenerator):
        def generate(self, unit, text, schema):
            self.budget.start(unit["unit_id"])
            self.budget.finish(False)
            raise FatalAPIError("insufficient_quota")

    stages.run_plan(ctx)
    ctx.generator = NoCredits(Budget(ctx.path("api_calls.jsonl"), max_calls=30))
    with pytest.raises(SystemExit, match="refuses this account"):
        stages.run_generate(ctx)
    assert len(read_jsonl(ctx.path("api_calls.jsonl"))) == 1 and read_jsonl(ctx.path("responses.jsonl")) == []


def test_account_errors_are_told_apart_from_passing_ones():
    class E(Exception):
        def __init__(self, code=None, status=None):
            super().__init__("x")
            self.code, self.status_code = code, status
    assert account_error(E(code="insufficient_quota")) and account_error(E(status=401)) and account_error(E(status=403))
    assert not account_error(E(status=429)) and not account_error(E(status=500))


def test_a_malformed_response_is_kept_as_a_failure_and_its_unit_is_not_built(ctx):
    class OneBad(MockGenerator):
        def generate(self, unit, text, schema):
            res = super().generate(unit, text, schema)
            return {**res, "text": "{}"} if unit["unit_id"] == self.bad else res

    stages.run_plan(ctx)
    asked = [u for u in read_jsonl(ctx.path("plan.jsonl")) if u["kind"] != "chat"]
    ctx.generator = OneBad(Budget(ctx.path("api_calls.jsonl"), max_calls=30))
    ctx.generator.bad = stages.generation_order(asked, ctx.cfg["seed"])[-1]["unit_id"]
    stages.run_generate(ctx)
    stages.run_build(ctx)
    responses = read_jsonl(ctx.path("responses.jsonl"))
    assert sum(not r["ok"] for r in responses) == 1 and len(responses) == 24
    assert ctx.generator.bad not in {c["unit_id"] for c in read_jsonl(ctx.path("conversations.jsonl"))}


def test_a_failed_check_drops_both_versions_of_a_pair(ctx):
    class Leaky(MockGenerator):
        def generate(self, unit, text, schema):
            res = super().generate(unit, text, schema)
            if unit["unit_id"] != self.bad:
                return res
            obj = json.loads(res["text"])
            obj["turns"][0]["user"] = "As a mother I need this. " + obj["turns"][0]["user"]
            return {**res, "text": json.dumps(obj)}

    stages.run_plan(ctx)
    pair = next(u for u in read_jsonl(ctx.path("plan.jsonl")) if u["group"] == "pair" and u["kind"] == "mail")
    ctx.generator = Leaky(Budget(ctx.path("api_calls.jsonl"), max_calls=30))
    ctx.generator.bad = pair["unit_id"]
    stages.run_generate(ctx)
    stages.run_build(ctx)
    mine = [c for c in read_jsonl(ctx.path("conversations.jsonl")) if c["unit_id"] == pair["unit_id"]]
    assert len(mine) == 2 and not any(c["check_passed"] for c in mine)
    assert any(p.startswith("self_gender_word") for p in mine[0]["problems"])
    assert pair["unit_id"] not in {p["unit_id"] for p in read_jsonl(ctx.path("prompts.jsonl"))}


def test_a_folder_with_responses_refuses_another_setup(tmp_path, cfg):
    out = tmp_path / "guard"
    out.mkdir()
    run.guard_run_dir(out, run.run_identity(cfg, True), "all")
    (out / "responses.jsonl").write_text(json.dumps({"unit_id": "ws-0001", "ok": True}) + "\n")
    run.guard_run_dir(out, run.run_identity(cfg, True), "all")                 # the same setup may continue
    with pytest.raises(SystemExit, match="different setup"):
        run.guard_run_dir(out, run.run_identity({**cfg, "seed": 99}, True), "all")
    run.guard_run_dir(out, run.run_identity({**cfg, "seed": 99}, True), "report")  # reading is always allowed


def test_the_command_line_runs_the_mock_end_to_end(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(run, "load_config", lambda path: cfg)
    out = tmp_path / "cli"
    run.main(["--config", "ignored.yaml", "--mock", "--out", str(out)])
    assert json.loads((out / "config.json").read_text())["delimiters"] == {"open": "<email>", "close": "</email>"}
    assert len(read_jsonl(out / "prompts.jsonl")) == 48


def test_the_real_request_is_built_without_creating_a_client(ctx, units):
    """The request body the real writer would send, built from the repo's model registry. Nothing is sent."""
    from synthetic_dataset_gen.phase2.requests import response_body
    from watersteg.client import OpenAIGenerator
    from watersteg.schema import output_schema

    gen = OpenAIGenerator(ctx.cfg["api"], Budget(ctx.path("api_calls.jsonl"), 30), run.REPO)
    unit = next(u for u in units if u["kind"] == "mail")
    body = response_body(ctx.prompts.text(unit, ctx.source), "watersteg_turns", output_schema(unit),
                         {"unit_id": unit["unit_id"], "kind": unit["kind"]}, gen.adapter)
    assert gen._client is None and body["model"] == gen.model and body["model"].startswith("gpt-5.4-mini")
    assert body["text"]["format"]["strict"] is True and body["text"]["format"]["name"] == "watersteg_turns"
    assert body["max_output_tokens"] == ctx.cfg["api"]["max_output_tokens"] and body["store"] is False
    assert body["reasoning"] == {"effort": "low"} and body["input"][0]["role"] == "user"

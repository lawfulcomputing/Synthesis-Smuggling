from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.generator import generate_catalog
from synthetic_dataset_gen.phase1.io import write_jsonl as write_catalog
from synthetic_dataset_gen.phase2.catalog import (
    InsufficientMatchesError,
    PairSplitError,
    filter_specs,
    load_catalog,
    select_specs,
)
from synthetic_dataset_gen.phase2.configuration import (
    RunOptions,
    SelectorConfig,
    load_defaults,
)
from synthetic_dataset_gen.phase2.output import (
    read_jsonl_objects,
    serialize_error,
    write_generations_csv,
    write_jsonl,
)
from synthetic_dataset_gen.phase2.plans import Plan
from synthetic_dataset_gen.phase2.pricing import (
    PricingRegistry,
    build_usage_record,
    usage_summary,
)
from synthetic_dataset_gen.phase2.prompts import PromptRenderer
from synthetic_dataset_gen.phase2.registry import ModelAdapter, ModelRegistry
from synthetic_dataset_gen.phase2.requests import (
    build_batch_line,
    parse_response,
    validate_request,
)
from synthetic_dataset_gen.phase2.run_management import create_run_directory
from synthetic_dataset_gen.phase2.runner import execute_run
from synthetic_dataset_gen.phase2.schemas import GenerationRecord
from synthetic_dataset_gen.phase2.transport import (
    RetryExhaustedError,
    call_with_retry,
    submit_batch,
)


@pytest.fixture(scope="module")
def phase1_rows(tmp_path_factory):
    # Built from config.toml so the tests don't depend on untracked artifacts.
    path = tmp_path_factory.mktemp("catalog") / "catalog.jsonl"
    write_catalog(path, generate_catalog(load_config("config.toml")))
    rows, catalog_hash = load_catalog(path)
    assert catalog_hash
    return rows


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


@pytest.fixture(scope="module")
def pricing():
    return PricingRegistry.load()


def test_phase1_catalog_loading(phase1_rows):
    assert len(phase1_rows) == 3000
    assert all(row.belief_trajectory for row in phase1_rows)


@pytest.mark.parametrize(
    ("selectors", "predicate"),
    [
        (SelectorConfig(length=5), lambda row: row.num_turns == 5),
        (
            SelectorConfig(trajectory="single_reveal"),
            lambda row: row.trajectory_family == "single_reveal",
        ),
        (SelectorConfig(gender="F"), lambda row: row.target_gender == "F"),
        (
            SelectorConfig(reveal_bucket="late"),
            lambda row: row.reveal_bucket == "late",
        ),
        (SelectorConfig(topic="coffee"), lambda row: row.topic == "coffee"),
        (
            SelectorConfig(min_length=4, max_length=6),
            lambda row: 4 <= row.num_turns <= 6,
        ),
    ],
)
def test_filtering(phase1_rows, selectors, predicate):
    selected = filter_specs(phase1_rows, selectors)
    assert selected
    assert all(predicate(row) for row in selected)


def test_exact_length_cannot_mix_with_range(phase1_rows):
    with pytest.raises(ValueError, match="cannot be combined"):
        filter_specs(phase1_rows, SelectorConfig(length=5, min_length=3))


def test_deterministic_selection(phase1_rows):
    selectors = SelectorConfig(trajectory="single_reveal")
    first = select_specs(phase1_rows, 12, 123, selectors)
    second = select_specs(phase1_rows, 12, 123, selectors)
    assert [row.conversation_id for row in first] == [
        row.conversation_id for row in second
    ]


def test_selection_keeps_pairs_whole(phase1_rows):
    selected = select_specs(phase1_rows, 101, 7, SelectorConfig())
    assert len(selected) == 101
    members = defaultdict(set)
    for row in selected:
        if row.counterfactual_pair_id:
            members[row.counterfactual_pair_id].add(row.pair_member)
    assert members
    assert all(genders == {"M", "F"} for genders in members.values())


@pytest.mark.parametrize("gender", ["M", "F"])
def test_gender_filter_on_pairs_explains_error(phase1_rows, gender):
    with pytest.raises(
        PairSplitError, match=f"--gender {gender} matches one member of"
    ):
        select_specs(phase1_rows, 2, 42, SelectorConfig(gender=gender))


def test_defaults_source_catalog_follows_artifact_dir():
    assert load_defaults().source_catalog == str(
        load_config("config.toml").catalog_path
    )


def test_selection_uses_independent_row_to_fill_odd_count(phase1_rows):
    selected = select_specs(
        phase1_rows, 3, 42, SelectorConfig(trajectory="single_reveal")
    )
    assert len(selected) == 3
    members = defaultdict(set)
    for row in selected:
        if row.counterfactual_pair_id:
            members[row.counterfactual_pair_id].add(row.pair_member)
    assert all(genders == {"M", "F"} for genders in members.values())
    assert any(row.counterfactual_pair_id is None for row in selected)


def test_insufficient_match_error(phase1_rows):
    selectors = SelectorConfig(topic="coffee")
    available = len(filter_specs(phase1_rows, selectors))
    with pytest.raises(InsufficientMatchesError) as caught:
        select_specs(phase1_rows, available + 1, 42, selectors)
    message = str(caught.value)
    assert f"requested count={available + 1}" in message
    assert f"available count={available}" in message
    assert "active filters=" in message


@pytest.mark.parametrize(
    ("identifier", "snapshot"),
    [
        ("gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"),
        ("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini-2026-03-17"),
    ],
)
def test_model_registry_resolution(registry, identifier, snapshot):
    assert registry.resolve(identifier).snapshot == snapshot


@pytest.mark.parametrize("family", ["never_reveal", "single_reveal"])
def test_model_prompt_selection(registry, phase1_rows, family):
    spec = next(row for row in phase1_rows if row.trajectory_family == family)
    renderer = PromptRenderer(registry.resolve("gpt-5.4-mini"))
    placeholder = Plan(scenario="A placeholder scenario.", turns=[])
    rendered = renderer.render(spec, plan=placeholder)
    assert rendered.template_name.endswith(f"/{family}.jinja2")
    assert str(spec.num_turns) in rendered.text
    assert spec.topic in rendered.text


def test_model_specific_parameter_validation(registry):
    with pytest.raises(ValueError, match="invalid"):
        ModelAdapter(registry.resolve("gpt-5.4-mini"), "minimal", None, 1000)
    adapter = ModelAdapter(registry.resolve("gpt-5.4-mini"), None, None, 1000)
    assert adapter.reasoning_effort == "none"
    assert adapter.verbosity == "low"


def fake_response(conversation_id: str = "response-test", turns: int = 2):
    output = {
        "turns": [
            {"turn": index, "user": f"user {index}", "assistant": f"assistant {index}"}
            for index in range(1, turns + 1)
        ]
    }
    return SimpleNamespace(
        id=conversation_id,
        model="gpt-5.4-mini-2026-03-17",
        output_text=json.dumps(output),
        usage={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 25},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 140,
        },
    )


def test_structured_output_parsing():
    parsed = parse_response(fake_response(turns=3), 3)
    assert len(parsed.output.turns) == 3
    with pytest.raises(ValueError, match="expected 2"):
        parse_response(fake_response(turns=3), 2)


def test_fake_batch_request_construction(registry, phase1_rows):
    spec = phase1_rows[0]
    model = registry.resolve("gpt-5.4-mini")
    adapter = ModelAdapter(model, None, None, 1200)
    prompt = PromptRenderer(model).render(spec)
    request = build_batch_line(spec, prompt, adapter)
    validate_request(request, spec)
    assert request["url"] == "/v1/responses"
    assert request["body"]["reasoning"] == {"effort": "none"}


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **parameters):
        self.calls.append(parameters)
        turns = parameters["text"]["format"]["schema"]["properties"]["turns"][
            "minItems"
        ]
        return fake_response(f"response-{len(self.calls)}", turns)


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def make_options(catalog: Path, phase_run: str, output_format: str = "jsonl"):
    return RunOptions(
        phase_run=phase_run,
        requested_model="gpt-5.4-mini",
        conversations=2,
        selectors=SelectorConfig(),
        output_format=output_format,
        seed=42,
        execution_mode="sync",
        max_output_tokens=1000,
        source_catalog=str(catalog),
    )


@pytest.fixture
def tiny_catalog(tmp_path, phase1_rows):
    path = tmp_path / "catalog.jsonl"
    write_catalog(path, phase1_rows[:12])
    return path


def isolated_defaults(tmp_path):
    defaults = load_defaults()
    retry = defaults.retry.model_copy(
        update={"initial_delay_seconds": 0.0, "max_delay_seconds": 0.0}
    )
    return defaults.model_copy(
        update={"run_root": str(tmp_path / "runs"), "retry": retry}
    )


def test_run_directory_creation_and_overwrite_protection(tmp_path):
    root = tmp_path / "runs"
    create_run_directory(root, "pilot", False)
    with pytest.raises(FileExistsError, match="--resume"):
        create_run_directory(root, "pilot", False)


def test_dry_run_causes_zero_api_calls(tmp_path, tiny_catalog, registry, pricing):
    options = make_options(tiny_catalog, "dry").model_copy(update={"dry_run": True})

    def forbidden_client():
        raise AssertionError("client factory must not run during dry-run")

    result = execute_run(
        options,
        defaults=isolated_defaults(tmp_path),
        registry=registry,
        pricing=pricing,
        client_factory=forbidden_client,
    )
    run_dir = Path(result["run_directory"])
    assert result["manifest"]["status"] == "dry_run_complete"
    assert len(read_jsonl_objects(run_dir / "requests.jsonl")) == 2
    assert (run_dir / "selected_specs.jsonl").is_file()
    assert list((run_dir / "prompt_snapshots").rglob("*.jinja2"))


@pytest.mark.parametrize(
    ("output_format", "expects_csv"),
    [("jsonl", False), ("csv", True), ("both", True)],
)
def test_fake_sync_and_output_modes(
    tmp_path, tiny_catalog, registry, pricing, output_format, expects_csv
):
    client = FakeClient()
    options = make_options(tiny_catalog, f"run_{output_format}", output_format)
    result = execute_run(
        options,
        defaults=isolated_defaults(tmp_path),
        registry=registry,
        pricing=pricing,
        client_factory=lambda: client,
    )
    run_dir = Path(result["run_directory"])
    assert len(client.responses.calls) == 2
    assert len(read_jsonl_objects(run_dir / "generations.jsonl")) == 2
    assert (run_dir / "generations.csv").exists() is expects_csv


def test_resume_skips_completed_and_avoids_duplicates(
    tmp_path, tiny_catalog, registry, pricing
):
    defaults = isolated_defaults(tmp_path)
    options = make_options(tiny_catalog, "resume_test")
    first_client = FakeClient()
    execute_run(
        options,
        defaults=defaults,
        registry=registry,
        pricing=pricing,
        client_factory=lambda: first_client,
    )
    resume_client = FakeClient()
    execute_run(
        options.model_copy(update={"resume": True}),
        defaults=defaults,
        registry=registry,
        pricing=pricing,
        client_factory=lambda: resume_client,
    )
    records = read_jsonl_objects(
        Path(defaults.run_root) / "resume_test/generations.jsonl"
    )
    assert len(records) == 2
    assert resume_client.responses.calls == []


def test_usage_extraction_cost_and_reasoning_not_double_counted(pricing):
    record = build_usage_record(
        conversation_id="c1",
        response_id="r1",
        model="gpt-5.4-mini-2026-03-17",
        pricing_key="gpt-5.4-mini",
        execution_mode="sync",
        usage={
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 250},
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 200},
            "total_tokens": 1500,
        },
        registry=pricing,
    )
    expected = 750 / 1_000_000 * 0.75 + 250 / 1_000_000 * 0.075 + 500 / 1_000_000 * 4.5
    assert record.standard_estimated_cost == pytest.approx(expected)
    assert record.output_tokens == 500
    assert record.reasoning_tokens == 200
    assert record.total_tokens == 1500


def test_batch_cost_adjustment_is_explicit(pricing):
    record = build_usage_record(
        conversation_id="c1",
        response_id="r1",
        model="gpt-5.4-mini-2026-03-17",
        pricing_key="gpt-5.4-mini",
        execution_mode="batch",
        usage={"input_tokens": 100, "output_tokens": 100, "total_tokens": 200},
        registry=pricing,
    )
    assert record.execution_multiplier == 0.5
    assert record.effective_execution_cost == pytest.approx(
        record.standard_estimated_cost * 0.5
    )


def test_usage_summary_projections(pricing):
    records = [
        build_usage_record(
            conversation_id=f"c{index}",
            response_id=f"r{index}",
            model="gpt-5.4-mini-2026-03-17",
            pricing_key="gpt-5.4-mini",
            execution_mode="sync",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            registry=pricing,
        )
        for index in range(2)
    ]
    summary = usage_summary(
        records,
        requested=3,
        failed=1,
        model="gpt-5.4-mini-2026-03-17",
        execution_mode="sync",
        pricing_table=pricing.table,
    )
    assert summary["successful_conversations"] == 2
    assert summary["failed_conversations"] == 1
    assert summary["projected_costs"]["1000_conversations"] == pytest.approx(
        summary["average_cost_per_conversation"] * 1000
    )


class RateLimitError(Exception):
    pass


def test_retry_behavior():
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] < 3:
            raise RateLimitError("temporary")
        return "ok"

    retry = load_defaults().retry.model_copy(
        update={"initial_delay_seconds": 0.0, "max_delay_seconds": 0.0}
    )
    value, attempts = call_with_retry(flaky, retry, sleep=lambda _: None)
    assert value == "ok"
    assert attempts == 3


def test_deterministic_error_is_not_retried():
    state = {"calls": 0}

    def invalid():
        state["calls"] += 1
        raise ValueError("bad schema")

    with pytest.raises(RetryExhaustedError) as caught:
        call_with_retry(invalid, load_defaults().retry, sleep=lambda _: None)
    assert caught.value.attempts == 1
    assert state["calls"] == 1


def test_error_serialization_redacts_secrets():
    scheme = "Bearer"
    bearer = f"{scheme} abc.secret"
    prefix = "sk"
    token = f"{prefix}-testsecret123456"
    error = serialize_error("c1", f"{bearer} and {token}", 2)
    assert "abc.secret" not in error.safe_message
    assert "testsecret" not in error.safe_message
    assert error.attempt_count == 2


class FakeFiles:
    def __init__(self):
        self.purpose = None

    def create(self, *, file, purpose):
        assert file.read()
        self.purpose = purpose
        return SimpleNamespace(id="file-1")


class FakeBatches:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(id="batch-1")


def test_fake_batch_submission(tmp_path):
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text('{"custom_id":"x"}\n')
    client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    batch, attempts = submit_batch(
        client, request_path, "24h", "pilot", load_defaults().retry
    )
    assert batch.id == "batch-1"
    assert attempts == 1
    assert client.files.purpose == "batch"
    assert client.batches.kwargs["endpoint"] == "/v1/responses"


def test_jsonl_and_csv_serialization(tmp_path, phase1_rows):
    parsed = parse_response(
        fake_response(turns=phase1_rows[0].num_turns), phase1_rows[0].num_turns
    )
    record = GenerationRecord(
        conversation_id=phase1_rows[0].conversation_id,
        response_id=parsed.response_id,
        requested_model="gpt-5.4-mini",
        resolved_model="gpt-5.4-mini-2026-03-17",
        returned_model=parsed.returned_model,
        prompt_template_name="gpt_5_4_mini/never_reveal.jinja2",
        phase1_spec=phase1_rows[0],
        generated=parsed.output,
    )
    write_jsonl(tmp_path / "generations.jsonl", [record])
    write_generations_csv(tmp_path / "generations.csv", [record])
    assert len(read_jsonl_objects(tmp_path / "generations.jsonl")) == 1
    assert "belief_trajectory" in (tmp_path / "generations.csv").read_text()


def test_no_api_key_in_produced_artifacts(
    tmp_path, tiny_catalog, registry, pricing, monkeypatch
):
    prefix = "sk"
    dummy_secret = f"{prefix}-test-secret-never-write-this"
    monkeypatch.setenv("OPENAI_API_KEY", dummy_secret)
    options = make_options(tiny_catalog, "secret_check").model_copy(
        update={"dry_run": True}
    )
    result = execute_run(
        options,
        defaults=isolated_defaults(tmp_path),
        registry=registry,
        pricing=pricing,
        client_factory=lambda: (_ for _ in ()).throw(AssertionError("no client")),
    )
    for path in Path(result["run_directory"]).rglob("*"):
        if path.is_file():
            assert dummy_secret.encode() not in path.read_bytes()


def test_gpt54_request_has_capability_specific_settings(registry, phase1_rows):
    spec = phase1_rows[0]
    model = registry.resolve("gpt-5.4-mini")
    adapter = ModelAdapter(model, None, None, 900)
    prompt = PromptRenderer(model).render(spec)
    request = build_batch_line(spec, prompt, adapter)
    validate_request(request, spec)
    assert request["body"]["reasoning"] == {"effort": "none"}
    assert request["body"]["text"]["verbosity"] == "low"
    assert request["body"]["max_output_tokens"] == 900


def test_resume_rejects_changed_core_configuration(
    tmp_path, tiny_catalog, registry, pricing
):
    defaults = isolated_defaults(tmp_path)
    options = make_options(tiny_catalog, "incompatible_resume").model_copy(
        update={"dry_run": True}
    )
    execute_run(
        options,
        defaults=defaults,
        registry=registry,
        pricing=pricing,
        client_factory=lambda: (_ for _ in ()).throw(AssertionError("no client")),
    )
    changed = options.model_copy(update={"resume": True, "dry_run": False, "seed": 99})
    with pytest.raises(ValueError, match="incompatible"):
        execute_run(
            changed,
            defaults=defaults,
            registry=registry,
            pricing=pricing,
            client_factory=lambda: (_ for _ in ()).throw(AssertionError("no client")),
        )


def test_failed_sync_writes_safe_errors(tmp_path, tiny_catalog, registry, pricing):
    class FailingResponses:
        def create(self, **parameters):
            raise ValueError("invalid request")

    client = SimpleNamespace(responses=FailingResponses())
    result = execute_run(
        make_options(tiny_catalog, "failed_sync"),
        defaults=isolated_defaults(tmp_path),
        registry=registry,
        pricing=pricing,
        client_factory=lambda: client,
    )
    errors = read_jsonl_objects(Path(result["run_directory"]) / "errors.jsonl")
    assert len(errors) == 2
    assert all(error["error_type"] == "ValueError" for error in errors)
    assert result["manifest"]["number_failed"] == 2

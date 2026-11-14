from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthetic_dataset_gen.gender_words import GenderedWords
from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.generator import generate_catalog
from synthetic_dataset_gen.phase1.io import write_jsonl as write_catalog
from synthetic_dataset_gen.phase2.configuration import (
    RunOptions,
    SelectorConfig,
    load_defaults,
)
from synthetic_dataset_gen.phase2.output import read_jsonl_objects
from synthetic_dataset_gen.phase2.plans import (
    Plan,
    PlanRenderer,
    check_plan,
    plan_output_schema,
    plan_units,
)
from synthetic_dataset_gen.phase2.pricing import PricingRegistry
from synthetic_dataset_gen.phase2.registry import ModelRegistry
from synthetic_dataset_gen.phase2.runner import execute_run

USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 40,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 140,
}


@pytest.fixture(scope="module")
def config():
    return load_config("config.toml")


@pytest.fixture(scope="module")
def catalog(config):
    return generate_catalog(config)


@pytest.fixture(scope="module")
def words(config):
    return GenderedWords.from_definitions(config.reveal_forms.definitions)


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


@pytest.fixture(scope="module")
def pricing():
    return PricingRegistry.load()


def pair_rows(catalog, *, reveal_turn_one: bool):
    pair_id = next(
        row.counterfactual_pair_id
        for row in catalog
        if row.counterfactual_pair_id and (row.reveal_turn == 1) == reveal_turn_one
    )
    return [row for row in catalog if row.counterfactual_pair_id == pair_id]


def unit_for(catalog, *, reveal_turn_one: bool):
    return plan_units(pair_rows(catalog, reveal_turn_one=reveal_turn_one))[0]


def make_plan(num_plan_turns: int, **overrides) -> dict:
    plan = {
        "scenario": "The user is planning a budget for a shared apartment move.",
        "turns": [
            {
                "turn": turn,
                "user_beat": f"The user asks about step {turn}.",
                "assistant_beat": f"The assistant explains step {turn}.",
            }
            for turn in range(1, num_plan_turns + 1)
        ],
    }
    plan.update(overrides)
    return plan


# -- word list -------------------------------------------------------------


def test_gendered_words_match_whole_words_ignoring_case(words):
    found = words.find("She said HIS Mom’s car; this is here. My sisters and Ma’am.")
    assert [(m.word, m.gender) for m in found] == [
        ("she", "F"),
        ("his", "M"),
        ("mom", "F"),
        ("sisters", "F"),
        ("ma'am", "F"),
    ]


def test_gendered_words_include_every_configured_term(config, words):
    for definition in config.reveal_forms.definitions.values():
        for male, female in definition.term_pairs:
            assert [m.gender for m in words.find(f"{male} {female}")] == ["M", "F"]
    assert [m.word for m in words.find("wives men women")] == ["wives", "men", "women"]


# -- plan units and prompt -------------------------------------------------


def test_one_plan_unit_per_pair(catalog):
    rows = pair_rows(catalog, reveal_turn_one=False)
    units = plan_units(rows)
    assert len(units) == 1
    assert units[0].plan_id == rows[0].counterfactual_pair_id
    assert sorted(units[0].conversation_ids) == sorted(r.conversation_id for r in rows)


def test_plan_units_need_reveal_terms(catalog):
    rows = [
        row.model_copy(update={"reveal_term": None})
        for row in pair_rows(catalog, reveal_turn_one=False)
    ]
    with pytest.raises(ValueError, match="no reveal_term"):
        plan_units(rows)


@pytest.mark.parametrize("reveal_turn_one", [True, False])
def test_plan_prompt_rendering(catalog, reveal_turn_one):
    unit = unit_for(catalog, reveal_turn_one=reveal_turn_one)
    text = PlanRenderer().render(unit).text
    assert f"assistant about {unit.spec.topic}." in text
    assert ("- turns: an empty list." in text) is reveal_turn_one
    if not reveal_turn_one:
        assert f"for turns 1 to {unit.spec.reveal_turn - 1}," in text
    for member in unit.members:
        assert member.reveal_term not in text.split()


# -- plan checks -----------------------------------------------------------


def test_plan_schema_accepts_empty_turns_when_reveal_is_turn_one(catalog, words):
    unit = unit_for(catalog, reveal_turn_one=True)
    plan = Plan.model_validate(make_plan(0))
    assert plan.turns == []
    assert check_plan(plan, unit, words) == []
    turns = plan_output_schema(unit.spec.reveal_turn - 1)["properties"]["turns"]
    assert (turns["minItems"], turns["maxItems"]) == (0, 0)


def test_valid_plan_passes_checks(catalog, words):
    unit = unit_for(catalog, reveal_turn_one=False)
    plan = Plan.model_validate(make_plan(unit.spec.reveal_turn - 1))
    assert check_plan(plan, unit, words) == []


def test_plan_checks_find_each_problem(catalog, words):
    unit = unit_for(catalog, reveal_turn_one=False)
    bad = make_plan(
        unit.spec.reveal_turn - 1,
        scenario="The user is a mother planning a family trip.",
    )
    bad["turns"][0]["turn"] = 7
    problems = check_plan(Plan.model_validate(bad), unit, words)
    assert any(
        "user's gender" in problem and "'mother'" in problem for problem in problems
    )
    assert any(p.startswith("turn numbers must run 1..") for p in problems)


def test_plan_allows_gendered_third_party(catalog, words):
    unit = unit_for(catalog, reveal_turn_one=False)
    plan = Plan.model_validate(
        make_plan(unit.spec.reveal_turn - 1, scenario="The user helps her brother.")
    )
    assert check_plan(plan, unit, words) == []


def test_plan_check_rejects_wrong_turn_count(catalog, words):
    unit = unit_for(catalog, reveal_turn_one=False)
    plan = Plan.model_validate(make_plan(unit.spec.reveal_turn))
    expected = unit.spec.reveal_turn - 1
    assert f"expected {expected} turns, found {expected + 1}" in check_plan(
        plan, unit, words
    )


# -- runs with fake clients ------------------------------------------------


REVEAL = re.compile(
    r"In user turn (\d+), the user says that they"
    r"(?:\s+themselves)?\s+(?:are|are about to become)\s+a\s+\"([^\"]+)\""
)


def conversation(prompt: str, num_turns: int, reveal_ok: bool = True) -> dict:
    """A conversation that passes the post-generation check (or, with
    reveal_ok=False, leaves the term out of the reveal turn)."""
    found = REVEAL.search(prompt)
    reveal_turn, term = (int(found.group(1)), found.group(2)) if found else (0, "")
    return {
        "turns": [
            {
                "turn": t,
                "user": f"user {t}: as a {term}, I have a question"
                if t == reveal_turn and reveal_ok
                else f"user {t}",
                "assistant": f"assistant {t}",
            }
            for t in range(1, num_turns + 1)
        ]
    }


def fake_body(params: dict, index: int, plan_factory, conversation_factory) -> dict:
    schema = params["text"]["format"]
    count = schema["schema"]["properties"]["turns"]["minItems"]
    if schema["name"] == "phase2_plan":
        output = plan_factory(index, count)
    else:
        prompt = params["input"][0]["content"][0]["text"]
        output = conversation_factory(index, params["metadata"], prompt, count)
    return {
        "id": f"resp-{index}",
        "model": "gpt-5.4-mini-2026-03-17",
        "output_text": json.dumps(output),
        "usage": USAGE,
    }


def good_plans(index, num_plan_turns):
    return make_plan(num_plan_turns)


def good_conversations(index, metadata, prompt, num_turns):
    return conversation(prompt, num_turns)


class FakeSyncClient:
    def __init__(
        self, plan_factory=good_plans, conversation_factory=good_conversations
    ):
        self.calls = []
        self.plan_factory = plan_factory
        self.conversation_factory = conversation_factory
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **params):
        self.calls.append(params)
        return fake_body(
            params, len(self.calls), self.plan_factory, self.conversation_factory
        )


class FakeBatchClient:
    """Completes each batch as soon as it is created."""

    def __init__(self, plan_factory=good_plans):
        self.plan_factory = plan_factory
        self.conversation_factory = good_conversations
        self.stored: dict[str, str] = {}
        self.batch_store: dict[str, SimpleNamespace] = {}
        self.submitted: list[list[dict]] = []
        self.files = SimpleNamespace(create=self.create_file, content=self.content)
        self.batches = SimpleNamespace(create=self.create_batch, retrieve=self.retrieve)

    def create_file(self, file, purpose):
        file_id = f"file-{len(self.stored) + 1}"
        self.stored[file_id] = file.read().decode("utf-8")
        return SimpleNamespace(id=file_id)

    def content(self, file_id):
        return io.BytesIO(self.stored[file_id].encode()).getvalue()

    def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        lines = [json.loads(line) for line in self.stored[input_file_id].splitlines()]
        self.submitted.append(lines)
        output = [
            json.dumps(
                {
                    "custom_id": line["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": fake_body(
                            line["body"],
                            index,
                            self.plan_factory,
                            self.conversation_factory,
                        ),
                    },
                }
            )
            for index, line in enumerate(lines, 1)
        ]
        output_id = f"file-{len(self.stored) + 1}"
        self.stored[output_id] = "\n".join(output) + "\n"
        batch = SimpleNamespace(
            id=f"batch-{len(self.batch_store) + 1}",
            status="completed",
            output_file_id=output_id,
            error_file_id=None,
        )
        self.batch_store[batch.id] = batch
        return batch

    def retrieve(self, batch_id):
        return self.batch_store[batch_id]


@pytest.fixture
def pair_catalog(tmp_path, catalog):
    rows = pair_rows(catalog, reveal_turn_one=False)
    never = next(row for row in catalog if row.trajectory_family == "never_reveal")
    path = tmp_path / "catalog.jsonl"
    write_catalog(path, [never, *rows])
    return path


def defaults_for(tmp_path, max_attempts=3):
    defaults = load_defaults()
    return defaults.model_copy(
        update={
            "run_root": str(tmp_path / "runs"),
            "retry": defaults.retry.model_copy(
                update={"initial_delay_seconds": 0.0, "max_delay_seconds": 0.0}
            ),
            "batch": defaults.batch.model_copy(update={"poll_interval_seconds": 0.01}),
            "plan": defaults.plan.model_copy(update={"max_attempts": max_attempts}),
        }
    )


def options_for(catalog_path, phase_run, **updates):
    values = {
        "phase_run": phase_run,
        "requested_model": "gpt-5.4-mini",
        "conversations": 3,
        "selectors": SelectorConfig(),
        "output_format": "jsonl",
        "seed": 42,
        "execution_mode": "sync",
        "max_output_tokens": 1000,
        "source_catalog": str(catalog_path),
    }
    return RunOptions(**{**values, **updates})


def run(options, defaults, registry, pricing, client):
    return execute_run(
        options,
        defaults=defaults,
        registry=registry,
        pricing=pricing,
        client_factory=lambda: client,
    )


def test_dry_run_prepares_plan_requests_only(tmp_path, pair_catalog, registry, pricing):
    def forbidden():
        raise AssertionError("dry run must not create a client")

    result = execute_run(
        options_for(pair_catalog, "dry", dry_run=True),
        defaults=defaults_for(tmp_path),
        registry=registry,
        pricing=pricing,
        client_factory=forbidden,
    )
    run_dir = Path(result["run_directory"])
    plan_requests = read_jsonl_objects(run_dir / "plan_requests.jsonl")
    assert len(plan_requests) == 1
    turn_schema = plan_requests[0]["body"]["text"]["format"]["schema"]["$defs"][
        "PlanTurn"
    ]
    assert list(turn_schema["properties"]) == ["turn", "user_beat", "assistant_beat"]
    # Only the never_reveal row has a generation request before plans exist.
    requests = read_jsonl_objects(run_dir / "requests.jsonl")
    assert [r["body"]["metadata"]["conversation_id"] for r in requests] == [
        "never_0001"
    ]
    assert (run_dir / "prompt_snapshots" / "shared" / "plan.jinja2").is_file()


def test_sync_run_plans_then_generates_both_members(
    tmp_path, pair_catalog, registry, pricing
):
    client = FakeSyncClient()
    result = run(
        options_for(pair_catalog, "sync"),
        defaults_for(tmp_path),
        registry,
        pricing,
        client,
    )
    run_dir = Path(result["run_directory"])
    stages = [call["text"]["format"]["name"] for call in client.calls]
    assert stages[0] == "phase2_plan"
    assert stages.count("phase2_plan") == 1
    plans = read_jsonl_objects(run_dir / "plans.jsonl")
    assert len(plans) == 1 and plans[0]["attempt"] == 1
    assert len(read_jsonl_objects(run_dir / "generations.jsonl")) == 3
    assert result["plan_summary"]["plans_passed"] == 1
    assert result["manifest"]["status"] == "completed"


def test_rejected_plan_is_retried_and_logged(tmp_path, pair_catalog, registry, pricing):
    def first_bad(index, num_plan_turns):
        if index == 1:
            return make_plan(
                num_plan_turns, scenario="The user is a sister planning a move."
            )
        return make_plan(num_plan_turns)

    client = FakeSyncClient(first_bad)
    result = run(
        options_for(pair_catalog, "retry"),
        defaults_for(tmp_path),
        registry,
        pricing,
        client,
    )
    run_dir = Path(result["run_directory"])
    errors = read_jsonl_objects(run_dir / "plan_errors.jsonl")
    assert [e["error_type"] for e in errors] == ["PlanCheckFailed"]
    assert "discloses the user's gender" in errors[0]["problems"][0]
    assert errors[0]["rejected_plan"]["scenario"] == (
        "The user is a sister planning a move."
    )
    assert read_jsonl_objects(run_dir / "plans.jsonl")[0]["attempt"] == 2
    assert len(read_jsonl_objects(run_dir / "generations.jsonl")) == 3
    plan_usage = read_jsonl_objects(run_dir / "plan_usage.jsonl")
    assert [u["rejected"] for u in plan_usage] == [True, False]
    usage = result["plan_summary"]["usage"]
    assert (usage["successful_conversations"], usage["rejected_attempts"]) == (1, 1)
    assert usage["total_input_tokens"] == 2 * USAGE["input_tokens"]


def test_attempt_limit_blocks_generation_and_holds_across_resume(
    tmp_path, pair_catalog, registry, pricing
):
    def always_bad(index, num_plan_turns):
        return make_plan(num_plan_turns, scenario="The user is a guy planning a trip.")

    defaults = defaults_for(tmp_path, max_attempts=2)
    options = options_for(pair_catalog, "limit")
    client = FakeSyncClient(always_bad)
    result = run(options, defaults, registry, pricing, client)
    run_dir = Path(result["run_directory"])
    assert [c["text"]["format"]["name"] for c in client.calls].count("phase2_plan") == 2
    generated = read_jsonl_objects(run_dir / "generations.jsonl")
    assert [g["conversation_id"] for g in generated] == ["never_0001"]
    assert result["manifest"]["number_failed"] == 2
    requests = read_jsonl_objects(run_dir / "requests.jsonl")
    assert [r["body"]["metadata"]["conversation_id"] for r in requests] == [
        "never_0001"
    ]

    resume_client = FakeSyncClient(always_bad)
    run(
        options.model_copy(update={"resume": True}),
        defaults,
        registry,
        pricing,
        resume_client,
    )
    assert resume_client.calls == []


def test_plans_only_stops_before_generation(tmp_path, pair_catalog, registry, pricing):
    defaults = defaults_for(tmp_path)
    options = options_for(pair_catalog, "plans_only", plans_only=True)
    client = FakeSyncClient()
    result = run(options, defaults, registry, pricing, client)
    run_dir = Path(result["run_directory"])
    assert [c["text"]["format"]["name"] for c in client.calls] == ["phase2_plan"]
    assert read_jsonl_objects(run_dir / "generations.jsonl") == []

    resume_client = FakeSyncClient()
    run(
        options.model_copy(update={"resume": True, "plans_only": False}),
        defaults,
        registry,
        pricing,
        resume_client,
    )
    names = [c["text"]["format"]["name"] for c in resume_client.calls]
    assert names == ["phase2_conversation"] * 3
    assert len(read_jsonl_objects(run_dir / "generations.jsonl")) == 3


def test_batch_run_with_no_wait_and_resume(tmp_path, pair_catalog, registry, pricing):
    def first_bad(index, num_plan_turns):
        if not hasattr(first_bad, "seen"):
            first_bad.seen = True
            return make_plan(
                num_plan_turns, scenario="The user is a lady planning a trip."
            )
        return make_plan(num_plan_turns)

    defaults = defaults_for(tmp_path)
    options = options_for(pair_catalog, "batch", execution_mode="batch", no_wait=True)
    client = FakeBatchClient(first_bad)
    result = run(options, defaults, registry, pricing, client)
    assert result["usage_summary"] is None
    assert result["manifest"]["plan_status"] == "plan_batch_submitted"
    assert len(client.submitted) == 1

    result = run(
        options.model_copy(update={"resume": True, "no_wait": False}),
        defaults,
        registry,
        pricing,
        client,
    )
    run_dir = Path(result["run_directory"])
    # Round 1 plan (rejected), round 2 plan (passed), then one generation batch.
    kinds = [
        {line["body"]["text"]["format"]["name"] for line in batch}
        for batch in client.submitted
    ]
    assert kinds == [{"phase2_plan"}, {"phase2_plan"}, {"phase2_conversation"}]
    assert len(client.submitted[2]) == 3
    assert (run_dir / "plan_requests_round_2.jsonl").is_file()
    assert read_jsonl_objects(run_dir / "plans.jsonl")[0]["attempt"] == 2
    assert len(read_jsonl_objects(run_dir / "generations.jsonl")) == 3
    assert result["manifest"]["status"] == "completed"


def test_rejected_member_is_retried_alone_with_the_same_plan(
    tmp_path, pair_catalog, registry, pricing
):
    target = next(
        row.conversation_id
        for row in pair_rows(
            generate_catalog(load_config("config.toml")), reveal_turn_one=False
        )
        if row.pair_member == "F"
    )

    def reject_target_once(index, metadata, prompt, num_turns):
        return conversation(
            prompt, num_turns, reveal_ok=metadata["conversation_id"] != target
        )

    defaults = defaults_for(tmp_path)
    options = options_for(pair_catalog, "member_retry")
    client = FakeSyncClient(conversation_factory=reject_target_once)
    result = run(options, defaults, registry, pricing, client)
    run_dir = Path(result["run_directory"])
    errors = read_jsonl_objects(run_dir / "errors.jsonl")
    assert [(e["conversation_id"], e["error_type"]) for e in errors] == [
        (target, "ConversationCheckError")
    ]
    assert errors[0]["safe_message"].startswith("a: turn ")
    assert result["manifest"]["number_failed"] == 1

    usage = read_jsonl_objects(run_dir / "usage.jsonl")
    assert [u["conversation_id"] for u in usage if u["rejected"]] == [target]
    summary = result["usage_summary"]
    assert (summary["successful_conversations"], summary["rejected_attempts"]) == (2, 1)
    assert summary["total_input_tokens"] == 3 * USAGE["input_tokens"]
    assert summary["rejected_attempts_cost"] > 0

    by_id = {
        g["conversation_id"]: g
        for g in read_jsonl_objects(run_dir / "generations.jsonl")
    }
    partner = next(cid for cid in by_id if cid.startswith(target[:-1]))
    assert by_id[partner]["pair_complete"] is False
    assert by_id["never_0001"]["pair_complete"] is None
    assert result["pair_summary"] == {
        "selected": 1,
        "complete": 0,
        "one_member_only": 1,
        "no_member": 0,
    }
    assert result["manifest"]["pair_summary"]["complete"] == 0

    resume_client = FakeSyncClient()
    resumed = run(
        options.model_copy(update={"resume": True}),
        defaults,
        registry,
        pricing,
        resume_client,
    )
    assert [c["metadata"]["conversation_id"] for c in resume_client.calls] == [target]
    first_try = next(
        c for c in client.calls if c["metadata"].get("conversation_id") == target
    )
    assert resume_client.calls[0] == first_try  # same request, so the same plan
    generations = read_jsonl_objects(run_dir / "generations.jsonl")
    assert len(generations) == 3
    assert [
        g["pair_complete"] for g in generations if g["conversation_id"] != "never_0001"
    ] == [True, True]
    assert resumed["pair_summary"]["complete"] == 1
    assert resumed["usage_summary"]["rejected_attempts"] == 1


def test_rejected_sample_is_stored_flagged_without_blocking_retry(
    tmp_path, pair_catalog, registry, pricing
):
    target = next(
        row.conversation_id
        for row in pair_rows(
            generate_catalog(load_config("config.toml")), reveal_turn_one=False
        )
        if row.pair_member == "F"
    )

    def reject_target_once(index, metadata, prompt, num_turns):
        return conversation(
            prompt, num_turns, reveal_ok=metadata["conversation_id"] != target
        )

    defaults = defaults_for(tmp_path)
    options = options_for(pair_catalog, "keep_rejected")
    client = FakeSyncClient(conversation_factory=reject_target_once)
    result = run(options, defaults, registry, pricing, client)
    run_dir = Path(result["run_directory"])

    rejected = read_jsonl_objects(run_dir / "rejected_samples.jsonl")
    assert [r["conversation_id"] for r in rejected] == [target]
    assert rejected[0]["check_passed"] is False
    assert rejected[0]["check_problems"]

    # The flagged copy must not mark the conversation as completed.
    generations = read_jsonl_objects(run_dir / "generations.jsonl")
    assert target not in [g["conversation_id"] for g in generations]
    errors = read_jsonl_objects(run_dir / "errors.jsonl")
    assert [e["conversation_id"] for e in errors] == [target]

    resume_client = FakeSyncClient()
    resumed = run(
        options.model_copy(update={"resume": True}),
        defaults,
        registry,
        pricing,
        resume_client,
    )
    assert [c["metadata"]["conversation_id"] for c in resume_client.calls] == [target]
    generations = read_jsonl_objects(run_dir / "generations.jsonl")
    assert target in [g["conversation_id"] for g in generations]
    assert resumed["pair_summary"]["complete"] == 1
    # Only the one rejected attempt was ever flagged, not the successful retry.
    rejected = read_jsonl_objects(run_dir / "rejected_samples.jsonl")
    assert len(rejected) == 1


def test_member_that_exhausts_retries_keeps_partner_in_batch(
    tmp_path, pair_catalog, registry, pricing
):
    target = next(
        row.conversation_id
        for row in pair_rows(
            generate_catalog(load_config("config.toml")), reveal_turn_one=False
        )
        if row.pair_member == "M"
    )

    def always_reject_target(index, metadata, prompt, num_turns):
        return conversation(
            prompt, num_turns, reveal_ok=metadata["conversation_id"] != target
        )

    defaults = defaults_for(tmp_path)
    options = options_for(pair_catalog, "exhaust", execution_mode="batch")
    client = FakeBatchClient()
    client.conversation_factory = always_reject_target
    result = run(options, defaults, registry, pricing, client)
    # Resume until the existing batch retry limit is used up.
    for _ in range(defaults.retry.max_attempts):
        result = run(
            options.model_copy(update={"resume": True}),
            defaults,
            registry,
            pricing,
            client,
        )
    run_dir = Path(result["run_directory"])
    conversation_batches = [
        b
        for b in client.submitted
        if b[0]["body"]["text"]["format"]["name"] == "phase2_conversation"
    ]
    assert len(conversation_batches) == defaults.retry.max_attempts
    assert all(len(b) == 1 for b in conversation_batches[1:])  # only the target
    generations = read_jsonl_objects(run_dir / "generations.jsonl")
    assert target not in {g["conversation_id"] for g in generations}
    partner = next(g for g in generations if g["phase1_spec"]["counterfactual_pair_id"])
    assert partner["pair_complete"] is False
    assert result["pair_summary"]["complete"] == 0
    rejected = [u for u in read_jsonl_objects(run_dir / "usage.jsonl") if u["rejected"]]
    assert len(rejected) == defaults.retry.max_attempts


def test_unparsable_attempt_is_billed_as_rejected(
    tmp_path, pair_catalog, registry, pricing
):
    def bad_numbering_once(index, metadata, prompt, num_turns):
        output = conversation(prompt, num_turns)
        if metadata["conversation_id"] == "never_0001" and not hasattr(
            bad_numbering_once, "done"
        ):
            bad_numbering_once.done = True
            output["turns"][0]["turn"] = 9
        return output

    client = FakeSyncClient(conversation_factory=bad_numbering_once)
    result = run(
        options_for(pair_catalog, "unparsable"),
        defaults_for(tmp_path),
        registry,
        pricing,
        client,
    )
    run_dir = Path(result["run_directory"])
    errors = read_jsonl_objects(run_dir / "errors.jsonl")
    assert [(e["conversation_id"], e["error_type"]) for e in errors] == [
        ("never_0001", "ValidationError")
    ]
    usage = read_jsonl_objects(run_dir / "usage.jsonl")
    assert [u["conversation_id"] for u in usage if u["rejected"]] == ["never_0001"]
    assert result["usage_summary"]["rejected_attempts"] == 1

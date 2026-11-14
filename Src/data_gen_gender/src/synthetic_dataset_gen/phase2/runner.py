"""Phase 2 orchestration with dry-run, sync, batch, and resume paths."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.gender_words import GenderedWords
from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.io import read_jsonl as read_catalog_jsonl
from synthetic_dataset_gen.phase1.io import write_jsonl as write_catalog_jsonl
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.catalog import load_catalog, select_specs
from synthetic_dataset_gen.phase2.configuration import DefaultsConfig, RunOptions
from synthetic_dataset_gen.phase2.conversation_check import (
    ConversationChecker,
    ConversationCheckError,
)
from synthetic_dataset_gen.phase2.output import (
    append_jsonl,
    read_jsonl_objects,
    serialize_error,
    write_generations_csv,
    write_json,
    write_jsonl,
)
from synthetic_dataset_gen.phase2.plan_stage import PlanStage
from synthetic_dataset_gen.phase2.plans import (
    PlanRecord,
    PlanRenderer,
    build_plan_line,
    plan_id_for,
    plan_units,
)
from synthetic_dataset_gen.phase2.pricing import (
    PricingRegistry,
    build_usage_record,
    usage_summary,
)
from synthetic_dataset_gen.phase2.prompts import PromptRenderer
from synthetic_dataset_gen.phase2.registry import ModelAdapter, ModelRegistry
from synthetic_dataset_gen.phase2.requests import (
    build_batch_line,
    parse_batch_output_line,
    parse_response,
    response_usage,
    validate_request,
)
from synthetic_dataset_gen.phase2.run_management import (
    RunPaths,
    append_log,
    assert_resume_compatible,
    core_configuration,
    create_run_directory,
    initial_manifest,
    load_manifest,
    utc_now,
    write_manifest,
)
from synthetic_dataset_gen.phase2.schemas import GenerationRecord, UsageRecord
from synthetic_dataset_gen.phase2.transport import (
    RetryExhaustedError,
    SyncTransport,
    create_openai_client,
    download_file_text,
    submit_batch,
    wait_for_batch,
)

ClientFactory = Callable[[], Any]


def _effective_options(options: RunOptions, adapter: ModelAdapter) -> RunOptions:
    return options.model_copy(
        update={
            "reasoning_effort": adapter.reasoning_effort,
            "verbosity": adapter.verbosity,
        }
    )


def _existing_records(
    paths: RunPaths,
) -> tuple[list[GenerationRecord], list[UsageRecord]]:
    generations = [
        GenerationRecord.model_validate(value)
        for value in read_jsonl_objects(paths.generations)
    ]
    usages = [
        UsageRecord.model_validate(value) for value in read_jsonl_objects(paths.usage)
    ]
    return generations, usages


def _finalize(
    paths: RunPaths,
    manifest: dict[str, Any],
    selected: list[CatalogRow],
    pricing: PricingRegistry,
    output_format: str,
) -> dict[str, Any]:
    generations, usages = _existing_records(paths)
    generations = _mark_pair_completeness(paths, generations)
    completed = {record.conversation_id for record in generations}
    accepted = {r.conversation_id: r for r in usages if not r.rejected}
    rejected = [record for record in usages if record.rejected]
    failures = len(selected) - len(completed)
    pairs = _pair_summary(selected, completed)
    summary = usage_summary(
        [*accepted.values(), *rejected],
        requested=len(selected),
        failed=failures,
        model=manifest["resolved_model"],
        execution_mode=manifest["execution_mode"],
        pricing_table=pricing.table,
    )
    write_json(paths.usage_summary, summary)
    if output_format in ("csv", "both"):
        write_generations_csv(paths.generations_csv, generations)
    returned = sorted(
        {record.returned_model for record in generations if record.returned_model}
    )
    manifest.update(
        {
            "completed_at": utc_now(),
            "status": "completed" if failures == 0 else "completed_with_errors",
            "number_completed": len(completed),
            "number_failed": failures,
            "pair_summary": pairs,
            "returned_model_if_available": returned[0]
            if len(returned) == 1
            else returned,
        }
    )
    write_manifest(paths.manifest, manifest)
    append_log(paths.log, f"{manifest['status']}; complete pairs {pairs['complete']}")
    return {
        "manifest": manifest,
        "usage_summary": summary,
        "pair_summary": pairs,
        "run_directory": str(paths.root),
    }


def _mark_pair_completeness(
    paths: RunPaths, generations: list[GenerationRecord]
) -> list[GenerationRecord]:
    """Set pair_complete on every pair member's record: True when both members
    were generated, False when only this one was. Nothing is removed."""
    generated: dict[str, int] = {}
    for record in generations:
        pair_id = record.phase1_spec.counterfactual_pair_id
        if pair_id:
            generated[pair_id] = generated.get(pair_id, 0) + 1
    updated = [
        record.model_copy(update={"pair_complete": generated[pair_id] == 2})
        if (pair_id := record.phase1_spec.counterfactual_pair_id)
        else record
        for record in generations
    ]
    if updated != generations:
        temporary = paths.generations.with_suffix(".jsonl.tmp")
        write_jsonl(temporary, updated)
        temporary.replace(paths.generations)
    return updated


def _pair_summary(selected: list[CatalogRow], completed: set[str]) -> dict[str, int]:
    members: dict[str, int] = {}
    for spec in selected:
        if spec.counterfactual_pair_id:
            done = spec.conversation_id in completed
            members[spec.counterfactual_pair_id] = (
                members.get(spec.counterfactual_pair_id, 0) + done
            )
    counts = list(members.values())
    return {
        "selected": len(counts),
        "complete": counts.count(2),
        "one_member_only": counts.count(1),
        "no_member": counts.count(0),
    }


def _record_unparsed_usage(
    *,
    spec: CatalogRow,
    response: Any,
    options: RunOptions,
    adapter: ModelAdapter,
    pricing: PricingRegistry,
    paths: RunPaths,
) -> None:
    """Record a rejected attempt whose output could not be parsed, when the
    response still reports usage."""
    fields = response_usage(response)
    if fields is None:
        return
    response_id, returned_model, usage = fields
    try:
        record = build_usage_record(
            conversation_id=spec.conversation_id,
            response_id=response_id,
            model=returned_model or adapter.spec.snapshot,
            pricing_key=adapter.spec.pricing_key,
            execution_mode=options.execution_mode,
            usage=usage,
            registry=pricing,
        )
    except ValueError:
        return
    append_jsonl(paths.usage, record.model_copy(update={"rejected": True}))


def _record_success(
    *,
    spec: CatalogRow,
    request: dict[str, Any],
    parsed: Any,
    options: RunOptions,
    adapter: ModelAdapter,
    pricing: PricingRegistry,
    paths: RunPaths,
    checker: ConversationChecker,
) -> None:
    metadata = request["body"]["metadata"]
    record = GenerationRecord(
        conversation_id=spec.conversation_id,
        response_id=parsed.response_id,
        requested_model=options.requested_model,
        resolved_model=adapter.spec.snapshot,
        returned_model=parsed.returned_model,
        prompt_template_name=metadata["prompt_template"],
        phase1_spec=spec,
        generated=parsed.output,
    )
    usage = build_usage_record(
        conversation_id=spec.conversation_id,
        response_id=parsed.response_id,
        model=parsed.returned_model or adapter.spec.snapshot,
        pricing_key=adapter.spec.pricing_key,
        execution_mode=options.execution_mode,
        usage=parsed.usage,
        registry=pricing,
    )
    # Rejections raise ConversationCheckError; callers log them to errors.jsonl
    # and the conversation is retried through the normal retry path. The
    # flagged output is also written to rejected_samples.jsonl (never
    # generations.jsonl, so it does not mark the conversation_id "completed"
    # and does not disturb retries) so the viewer can show it next to its
    # accepted pair partner.
    try:
        checker.check(spec, parsed.output.turns)
    except ConversationCheckError as error:
        append_jsonl(paths.usage, usage.model_copy(update={"rejected": True}))
        append_jsonl(
            paths.rejected_samples,
            record.model_copy(
                update={"check_passed": False, "check_problems": error.problems}
            ),
        )
        raise
    append_jsonl(paths.usage, usage)
    append_jsonl(paths.generations, record)


def _run_sync(
    *,
    selected: list[CatalogRow],
    requests: dict[str, dict[str, Any]],
    completed: set[str],
    options: RunOptions,
    adapter: ModelAdapter,
    pricing: PricingRegistry,
    defaults: DefaultsConfig,
    paths: RunPaths,
    client: Any,
    checker: ConversationChecker,
) -> None:
    transport = SyncTransport(client, defaults.retry)
    for spec in selected:
        if spec.conversation_id in completed:
            continue
        request = requests[spec.conversation_id]
        attempts = 1
        response = parsed = None
        try:
            response, attempts = transport.generate(request["body"])
            parsed = parse_response(response, spec.num_turns)
            _record_success(
                spec=spec,
                request=request,
                parsed=parsed,
                options=options,
                adapter=adapter,
                pricing=pricing,
                paths=paths,
                checker=checker,
            )
            completed.add(spec.conversation_id)
            append_log(paths.log, f"completed {spec.conversation_id}")
        except RetryExhaustedError as error:
            append_jsonl(
                paths.errors,
                serialize_error(spec.conversation_id, error, error.attempts),
            )
            append_log(paths.log, f"failed {spec.conversation_id}: {error}")
        except Exception as error:  # noqa: BLE001
            if response is not None and parsed is None:
                _record_unparsed_usage(
                    spec=spec,
                    response=response,
                    options=options,
                    adapter=adapter,
                    pricing=pricing,
                    paths=paths,
                )
            append_jsonl(
                paths.errors, serialize_error(spec.conversation_id, error, attempts)
            )
            append_log(paths.log, f"failed {spec.conversation_id}: {error}")


def _process_batch_text(
    *,
    text: str,
    selected_by_id: dict[str, CatalogRow],
    requests: dict[str, dict[str, Any]],
    completed: set[str],
    options: RunOptions,
    adapter: ModelAdapter,
    pricing: PricingRegistry,
    paths: RunPaths,
    checker: ConversationChecker,
) -> None:
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id", "")
        conversation_id = custom_id.removeprefix("phase2-")
        spec = selected_by_id.get(conversation_id)
        if spec is None:
            append_jsonl(
                paths.errors,
                serialize_error(
                    conversation_id or "unknown", "unmatched batch custom_id", 1
                ),
            )
            continue
        if conversation_id in completed:
            continue
        parsed = None
        try:
            parsed = parse_batch_output_line(line, spec.num_turns)
            _record_success(
                spec=spec,
                request=requests[conversation_id],
                parsed=parsed,
                options=options,
                adapter=adapter,
                pricing=pricing,
                paths=paths,
                checker=checker,
            )
            completed.add(conversation_id)
        except Exception as error:  # noqa: BLE001
            if parsed is None:
                _record_unparsed_usage(
                    spec=spec,
                    response=(item.get("response") or {}).get("body") or {},
                    options=options,
                    adapter=adapter,
                    pricing=pricing,
                    paths=paths,
                )
            append_jsonl(paths.errors, serialize_error(conversation_id, error, 1))


def _run_batch(
    *,
    selected: list[CatalogRow],
    requests: dict[str, dict[str, Any]],
    completed: set[str],
    options: RunOptions,
    adapter: ModelAdapter,
    pricing: PricingRegistry,
    defaults: DefaultsConfig,
    paths: RunPaths,
    manifest: dict[str, Any],
    client: Any,
    checker: ConversationChecker,
) -> bool:
    batch_id = manifest.get("batch_id")
    request_path = paths.requests
    if options.resume and manifest.get("status") == "completed_with_errors":
        missing = [spec for spec in selected if spec.conversation_id not in completed]
        retry_round = int(manifest.get("batch_retry_round", 0))
        if missing and retry_round < defaults.retry.max_attempts - 1:
            retry_round += 1
            request_path = paths.root / f"requests_retry_{retry_round}.jsonl"
            write_jsonl(
                request_path,
                [requests[spec.conversation_id] for spec in missing],
                sort_keys=False,
            )
            history = list(manifest.get("batch_id_history", []))
            if batch_id:
                history.append(batch_id)
            manifest.update(
                {
                    "batch_id": None,
                    "batch_id_history": history,
                    "batch_retry_round": retry_round,
                    "status": "batch_retry_prepared",
                }
            )
            batch_id = None
            write_manifest(paths.manifest, manifest)
        elif missing:
            append_log(paths.log, "batch retry limit reached; missing items retained")
            return True
        else:
            return True
    if not batch_id:
        batch, attempts = submit_batch(
            client,
            request_path,
            defaults.batch.completion_window,
            options.phase_run,
            defaults.retry,
        )
        batch_id = str(batch.id)
        manifest.update(
            {
                "batch_id": batch_id,
                "batch_submission_attempts": attempts,
                "status": "batch_submitted",
            }
        )
        write_manifest(paths.manifest, manifest)
        append_log(paths.log, f"submitted batch {batch_id}")
    if options.no_wait:
        return False

    batch = wait_for_batch(
        client,
        batch_id,
        defaults.retry,
        defaults.batch.poll_interval_seconds,
    )
    status = getattr(batch, "status", None)
    if status != "completed":
        for spec in selected:
            if spec.conversation_id not in completed:
                append_jsonl(
                    paths.errors,
                    serialize_error(
                        spec.conversation_id,
                        f"batch ended with status {status}",
                        1,
                    ),
                )
        return True
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("completed batch has no output file")
    text = download_file_text(client, output_file_id, defaults.retry)
    paths.batch_output.write_text(text, encoding="utf-8")
    _process_batch_text(
        text=text,
        selected_by_id={spec.conversation_id: spec for spec in selected},
        requests=requests,
        completed=completed,
        options=options,
        adapter=adapter,
        pricing=pricing,
        paths=paths,
        checker=checker,
    )
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_text = download_file_text(client, error_file_id, defaults.retry)
        _process_batch_text(
            text=error_text,
            selected_by_id={spec.conversation_id: spec for spec in selected},
            requests=requests,
            completed=completed,
            options=options,
            adapter=adapter,
            pricing=pricing,
            paths=paths,
            checker=checker,
        )
    return True


def _add_planned_requests(
    *,
    selected: list[CatalogRow],
    plans: dict[str, PlanRecord],
    request_lines: list[dict[str, Any]],
    renderer: PromptRenderer,
    adapter: ModelAdapter,
    paths: RunPaths,
) -> list[dict[str, Any]]:
    """Build generation requests for single_reveal specs whose plan passed."""
    existing = {line["body"]["metadata"]["conversation_id"] for line in request_lines}
    rendered, added = [], []
    for spec in selected:
        record = plans.get(plan_id_for(spec))
        if (
            spec.trajectory_family != "single_reveal"
            or spec.conversation_id in existing
            or record is None
        ):
            continue
        prompt = renderer.render(spec, plan=record.plan)
        request = build_batch_line(spec, prompt, adapter)
        validate_request(request, spec)
        rendered.append(prompt)
        added.append(request)
    if not added:
        return request_lines
    request_lines = request_lines + added
    # Unsorted: the model fills schema properties in order (turn, user, assistant).
    write_jsonl(paths.requests, request_lines, sort_keys=False)
    renderer.snapshot_used(rendered, paths.prompt_snapshots)
    append_log(paths.log, f"built {len(added)} generation requests from passing plans")
    return request_lines


def _plan_summary(
    stage: PlanStage, pricing: PricingRegistry, model: str, execution_mode: str
) -> dict[str, Any]:
    errors = read_jsonl_objects(stage.paths.plan_errors)
    usage = [
        UsageRecord.model_validate(v)
        for v in read_jsonl_objects(stage.paths.plan_usage)
    ]
    plans_passed = len(stage.plans())
    summary = {
        "plans_requested": len(stage.units),
        "plans_passed": plans_passed,
        "plans_without_a_passing_plan": len(stage.units) - plans_passed,
        "max_attempts_per_plan": stage.defaults.plan.max_attempts,
        "attempts_recorded": plans_passed + len(errors),
        "attempts_rejected_by_checks": sum(
            error["error_type"] == "PlanCheckFailed" for error in errors
        ),
        "attempts_failed_otherwise": sum(
            error["error_type"] != "PlanCheckFailed" for error in errors
        ),
        # Token and cost totals over every attempt that reported usage,
        # including rejected ones (marked "rejected": true).
        "usage": usage_summary(
            usage,
            requested=len(stage.units),
            failed=len(stage.units) - plans_passed,
            model=model,
            execution_mode=execution_mode,
            pricing_table=pricing.table,
        ),
    }
    write_json(stage.paths.plan_usage_summary, summary)
    return summary


def execute_run(
    options: RunOptions,
    *,
    defaults: DefaultsConfig,
    registry: ModelRegistry,
    pricing: PricingRegistry,
    project_root: str | Path = ".",
    client_factory: ClientFactory = create_openai_client,
) -> dict[str, Any]:
    model = registry.resolve(options.requested_model)
    if not model.supports_structured_outputs:
        raise ValueError(
            f"model {model.model_alias} does not support Structured Outputs"
        )
    adapter = ModelAdapter(
        model,
        options.reasoning_effort,
        options.verbosity,
        options.max_output_tokens,
    )
    options = _effective_options(options, adapter)
    rows, source_hash = load_catalog(options.source_catalog)
    configuration = core_configuration(
        options,
        resolved_model=model.snapshot,
        source_catalog_hash=source_hash,
    )
    definitions = load_config(defaults.phase1_config).reveal_forms.definitions
    renderer = PromptRenderer(model, project_root)
    if options.resume:
        paths = create_run_directory(defaults.run_root, options.phase_run, True)
        manifest = load_manifest(paths.manifest)
        assert_resume_compatible(manifest, configuration)
        selected = read_catalog_jsonl(paths.selected_specs)
        units = plan_units(selected)
        plan_lines = read_jsonl_objects(paths.plan_requests)
        request_lines = read_jsonl_objects(paths.requests)
    else:
        selected = select_specs(
            rows, options.conversations, options.seed, options.selectors
        )
        units = plan_units(selected)
        plan_renderer = PlanRenderer()
        plan_lines = [
            build_plan_line(unit, plan_renderer.render(unit), adapter) for unit in units
        ]
        # single_reveal requests are built only once their plan passes checks.
        unplanned = [s for s in selected if s.trajectory_family != "single_reveal"]
        rendered = [renderer.render(spec) for spec in unplanned]
        request_lines = [
            build_batch_line(spec, prompt, adapter)
            for spec, prompt in zip(unplanned, rendered, strict=True)
        ]
        for spec, request in zip(unplanned, request_lines, strict=True):
            validate_request(request, spec)
        paths = create_run_directory(defaults.run_root, options.phase_run, False)
        manifest = initial_manifest(
            options=options,
            configuration=configuration,
            resolved_model=model.snapshot,
            model_family=model.family,
            source_catalog_hash=source_hash,
            pricing_table=pricing.table,
            model_registry=registry.date,
        )
        write_catalog_jsonl(paths.selected_specs, selected)
        # Unsorted: the model fills schema properties in their listed order
        # (turn, user, assistant; turn, user_beat, assistant_beat).
        write_jsonl(paths.plan_requests, plan_lines, sort_keys=False)
        write_jsonl(paths.requests, request_lines, sort_keys=False)
        renderer.snapshot_used(rendered, paths.prompt_snapshots)
        if units:
            plan_renderer.snapshot(paths.prompt_snapshots)
        write_json(
            paths.resolved_config,
            {
                "options": options.model_dump(mode="json"),
                "model": model.model_dump(mode="json"),
                "model_registry": registry.date,
                "pricing_table": pricing.table,
                "source_catalog_hash": source_hash,
            },
        )
        write_manifest(paths.manifest, manifest)
        append_log(paths.log, "run artifacts initialized")

    requests = {
        request["body"]["metadata"]["conversation_id"]: request
        for request in request_lines
    }
    if len(requests) != len(request_lines):
        raise ValueError("request custom metadata is missing or duplicated")
    plan_requests = {line["body"]["metadata"]["plan_id"]: line for line in plan_lines}
    if set(plan_requests) != {unit.plan_id for unit in units}:
        raise ValueError("plan requests do not match the selected pairs")
    stage = PlanStage(
        units=units,
        requests=plan_requests,
        paths=paths,
        options=options,
        adapter=adapter,
        pricing=pricing,
        defaults=defaults,
        words=GenderedWords.from_definitions(definitions),
    )
    checker = ConversationChecker.from_definitions(definitions)

    if options.dry_run:
        summary = usage_summary(
            [],
            requested=len(selected),
            failed=0,
            model=model.snapshot,
            execution_mode=options.execution_mode,
            pricing_table=pricing.table,
        )
        write_json(paths.usage_summary, summary)
        manifest.update(
            {
                "completed_at": utc_now(),
                "status": "dry_run_complete",
                "number_completed": 0,
                "number_failed": 0,
                "number_plans_requested": len(units),
            }
        )
        write_manifest(paths.manifest, manifest)
        append_log(
            paths.log,
            f"dry run complete; {len(units)} plan requests prepared; single_reveal "
            "generation requests are built after their plans pass; no client "
            "created and no API request made",
        )
        return {
            "manifest": manifest,
            "usage_summary": summary,
            "run_directory": str(paths.root),
        }

    client = client_factory()
    if options.execution_mode == "sync":
        stage.run_sync(client)
    elif not stage.run_batch(client, manifest):
        return {
            "manifest": manifest,
            "usage_summary": None,
            "run_directory": str(paths.root),
        }
    plan_summary = _plan_summary(stage, pricing, model.snapshot, options.execution_mode)
    manifest.update(
        {
            "plan_status": "plans_complete",
            "number_plans_requested": plan_summary["plans_requested"],
            "number_plans_passed": plan_summary["plans_passed"],
        }
    )
    write_manifest(paths.manifest, manifest)
    if options.plans_only:
        append_log(paths.log, "plan stage complete; stopped because of --plans-only")
        return {
            "manifest": manifest,
            "plan_summary": plan_summary,
            "usage_summary": None,
            "run_directory": str(paths.root),
        }

    request_lines = _add_planned_requests(
        selected=selected,
        plans=stage.plans(),
        request_lines=request_lines,
        renderer=renderer,
        adapter=adapter,
        paths=paths,
    )
    requests = {
        request["body"]["metadata"]["conversation_id"]: request
        for request in request_lines
    }
    # Specs whose plan never passed have no request and count as failed.
    ready = [spec for spec in selected if spec.conversation_id in requests]
    generations, _ = _existing_records(paths)
    completed = {record.conversation_id for record in generations}
    if options.execution_mode == "sync":
        _run_sync(
            selected=ready,
            requests=requests,
            completed=completed,
            options=options,
            adapter=adapter,
            pricing=pricing,
            defaults=defaults,
            paths=paths,
            client=client,
            checker=checker,
        )
    else:
        finished = _run_batch(
            selected=ready,
            requests=requests,
            completed=completed,
            options=options,
            adapter=adapter,
            pricing=pricing,
            defaults=defaults,
            paths=paths,
            manifest=manifest,
            client=client,
            checker=checker,
        )
        if not finished:
            return {
                "manifest": manifest,
                "usage_summary": None,
                "run_directory": str(paths.root),
            }
    result = _finalize(paths, manifest, selected, pricing, options.output_format)
    return {**result, "plan_summary": plan_summary}

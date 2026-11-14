"""Plan stage: make and check one plan per pair (or unpaired row) before generation.

Every attempt is recorded, either as a passing plan in plans.jsonl or as a
rejection/failure in plan_errors.jsonl, and attempts are counted from those
files, so the attempt limit holds across --resume in both sync and batch mode.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from synthetic_dataset_gen.gender_words import GenderedWords
from synthetic_dataset_gen.phase2.configuration import DefaultsConfig, RunOptions
from synthetic_dataset_gen.phase2.output import (
    append_jsonl,
    read_jsonl_objects,
    safe_message,
    write_jsonl,
)
from synthetic_dataset_gen.phase2.plans import (
    PLAN_CUSTOM_ID_PREFIX,
    PlanErrorRecord,
    PlanRecord,
    PlanUnit,
    check_plan,
    parse_plan_response,
)
from synthetic_dataset_gen.phase2.pricing import PricingRegistry, build_usage_record
from synthetic_dataset_gen.phase2.registry import ModelAdapter
from synthetic_dataset_gen.phase2.requests import response_usage
from synthetic_dataset_gen.phase2.run_management import (
    RunPaths,
    append_log,
    write_manifest,
)
from synthetic_dataset_gen.phase2.transport import (
    RetryExhaustedError,
    SyncTransport,
    download_file_text,
    submit_batch,
    wait_for_batch,
)


@dataclass
class PlanStage:
    units: list[PlanUnit]
    requests: dict[str, dict[str, Any]]  # plan_id -> plan request line
    paths: RunPaths
    options: RunOptions
    adapter: ModelAdapter
    pricing: PricingRegistry
    defaults: DefaultsConfig
    words: GenderedWords

    def plans(self) -> dict[str, PlanRecord]:
        records = [
            PlanRecord.model_validate(v) for v in read_jsonl_objects(self.paths.plans)
        ]
        return {record.plan_id: record for record in records}

    def attempts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for path in (self.paths.plans, self.paths.plan_errors):
            counts.update(value["plan_id"] for value in read_jsonl_objects(path))
        return counts

    def pending(self) -> list[PlanUnit]:
        done, attempts = self.plans(), self.attempts()
        limit = self.defaults.plan.max_attempts
        return [
            unit
            for unit in self.units
            if unit.plan_id not in done and attempts[unit.plan_id] < limit
        ]

    # -- recording -------------------------------------------------------

    def _error(
        self, unit: PlanUnit, attempt: int, error_type: str, message: str, **extra: Any
    ) -> None:
        append_jsonl(
            self.paths.plan_errors,
            PlanErrorRecord(
                plan_id=unit.plan_id,
                attempt=attempt,
                error_type=error_type,
                safe_message=safe_message(message),
                **extra,
            ),
        )
        append_log(
            self.paths.log, f"plan {unit.plan_id} attempt {attempt}: {error_type}"
        )

    def _usage(
        self,
        unit: PlanUnit,
        response_id: str,
        returned_model: str | None,
        usage: dict[str, Any],
        *,
        rejected: bool,
    ) -> None:
        try:
            record = build_usage_record(
                conversation_id=unit.plan_id,
                response_id=response_id,
                model=returned_model or self.adapter.spec.snapshot,
                pricing_key=self.adapter.spec.pricing_key,
                execution_mode=self.options.execution_mode,
                usage=usage,
                registry=self.pricing,
            )
        except ValueError:
            return
        append_jsonl(
            self.paths.plan_usage, record.model_copy(update={"rejected": rejected})
        )

    def _record(self, unit: PlanUnit, response: Any, attempt: int) -> bool:
        """Record one plan response; return True when the plan passed its checks."""
        try:
            parsed = parse_plan_response(response)
        except Exception as error:  # noqa: BLE001
            if (fields := response_usage(response)) is not None:
                self._usage(unit, *fields, rejected=True)
            self._error(unit, attempt, error.__class__.__name__, str(error))
            return False
        problems = check_plan(parsed.plan, unit, self.words)
        self._usage(
            unit,
            parsed.response_id,
            parsed.returned_model,
            parsed.usage,
            rejected=bool(problems),
        )
        if problems:
            self._error(
                unit,
                attempt,
                "PlanCheckFailed",
                f"{len(problems)} plan check(s) failed",
                problems=problems,
                response_id=parsed.response_id,
                rejected_plan=parsed.plan.model_dump(mode="json"),
            )
            return False
        metadata = self.requests[unit.plan_id]["body"]["metadata"]
        append_jsonl(
            self.paths.plans,
            PlanRecord(
                plan_id=unit.plan_id,
                conversation_ids=unit.conversation_ids,
                attempt=attempt,
                response_id=parsed.response_id,
                requested_model=self.options.requested_model,
                resolved_model=self.adapter.spec.snapshot,
                returned_model=parsed.returned_model,
                prompt_template_name=metadata["prompt_template"],
                plan=parsed.plan,
            ),
        )
        append_log(self.paths.log, f"plan {unit.plan_id} passed on attempt {attempt}")
        return True

    # -- sync ------------------------------------------------------------

    def run_sync(self, client: Any) -> None:
        transport = SyncTransport(client, self.defaults.retry)
        limit = self.defaults.plan.max_attempts
        done, attempts = set(self.plans()), self.attempts()
        for unit in self.units:
            while unit.plan_id not in done and attempts[unit.plan_id] < limit:
                attempts[unit.plan_id] += 1
                attempt = attempts[unit.plan_id]
                try:
                    response, _ = transport.generate(
                        self.requests[unit.plan_id]["body"]
                    )
                except RetryExhaustedError as error:
                    self._error(unit, attempt, error.original_type, str(error))
                    continue
                if self._record(unit, response, attempt):
                    done.add(unit.plan_id)

    # -- batch -----------------------------------------------------------

    def run_batch(self, client: Any, manifest: dict[str, Any]) -> bool:
        """Submit and collect plan batches until every plan passed or ran out of
        attempts. Returns False when it stopped early because of --no-wait."""
        while True:
            batch_id = manifest.get("plan_batch_id")
            if batch_id is None:
                pending = self.pending()
                if not pending:
                    return True
                round_number = int(manifest.get("plan_batch_round", 0)) + 1
                request_path = (
                    self.paths.root / f"plan_requests_round_{round_number}.jsonl"
                )
                # Unsorted: the model fills schema properties in their listed order.
                write_jsonl(
                    request_path,
                    [self.requests[unit.plan_id] for unit in pending],
                    sort_keys=False,
                )
                batch, attempts = submit_batch(
                    client,
                    request_path,
                    self.defaults.batch.completion_window,
                    f"{self.options.phase_run}:plans",
                    self.defaults.retry,
                )
                batch_id = str(batch.id)
                manifest.update(
                    {
                        "plan_batch_id": batch_id,
                        "plan_batch_round": round_number,
                        "plan_batch_plan_ids": [unit.plan_id for unit in pending],
                        "plan_batch_submission_attempts": attempts,
                        "plan_status": "plan_batch_submitted",
                    }
                )
                write_manifest(self.paths.manifest, manifest)
                append_log(
                    self.paths.log,
                    f"submitted plan batch {batch_id} (round {round_number})",
                )
            if self.options.no_wait:
                return False
            batch = wait_for_batch(
                client,
                batch_id,
                self.defaults.retry,
                self.defaults.batch.poll_interval_seconds,
            )
            self._collect_batch(client, batch, manifest)
            history = list(manifest.get("plan_batch_id_history", [])) + [batch_id]
            manifest.update(
                {
                    "plan_batch_id": None,
                    "plan_batch_id_history": history,
                    "plan_status": "plan_batch_processed",
                }
            )
            write_manifest(self.paths.manifest, manifest)

    def _collect_batch(self, client: Any, batch: Any, manifest: dict[str, Any]) -> None:
        by_id = {unit.plan_id: unit for unit in self.units}
        submitted = [
            by_id[plan_id] for plan_id in manifest.get("plan_batch_plan_ids", [])
        ]
        round_number = manifest.get("plan_batch_round", 0)
        attempts = self.attempts()
        status = getattr(batch, "status", None)
        if status != "completed":
            for unit in submitted:
                self._error(
                    unit,
                    attempts[unit.plan_id] + 1,
                    "BatchStatus",
                    f"plan batch ended with status {status}",
                )
            return
        seen: set[str] = set()
        raw_path = self.paths.root / f"plan_batch_output_round_{round_number}.jsonl"
        raw_path.write_text("", encoding="utf-8")
        for file_id in (
            getattr(batch, "output_file_id", None),
            getattr(batch, "error_file_id", None),
        ):
            if not file_id:
                continue
            text = download_file_text(client, file_id, self.defaults.retry)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(text if text.endswith("\n") or not text else text + "\n")
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                plan_id = str(item.get("custom_id", "")).removeprefix(
                    PLAN_CUSTOM_ID_PREFIX
                )
                unit = by_id.get(plan_id)
                if unit is None or plan_id in seen:
                    continue
                seen.add(plan_id)
                attempt = attempts[plan_id] + 1
                response = item.get("response") or {}
                if item.get("error") or response.get("status_code") != 200:
                    detail = (
                        item.get("error")
                        or f"HTTP status {response.get('status_code')}"
                    )
                    self._error(unit, attempt, "BatchItemError", json.dumps(detail))
                else:
                    self._record(unit, response.get("body", {}), attempt)
        for unit in submitted:
            if unit.plan_id not in seen:
                self._error(
                    unit,
                    attempts[unit.plan_id] + 1,
                    "MissingBatchOutput",
                    "plan batch returned no line for this plan",
                )

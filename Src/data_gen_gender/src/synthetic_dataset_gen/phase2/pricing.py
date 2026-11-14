"""Token pricing (a dated price table) and cost accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from synthetic_dataset_gen.phase2.configuration import load_yaml
from synthetic_dataset_gen.phase2.schemas import UsageRecord


class PriceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    effective_date: str
    input_per_1m: float = Field(ge=0)
    cached_input_per_1m: float = Field(ge=0)
    output_per_1m: float = Field(ge=0)
    currency: str
    source_note: str


class PricingRegistry:
    def __init__(
        self,
        table: str,
        reference_date: str,
        currency: str,
        adjustments: dict[str, float],
        models: dict[str, PriceEntry],
    ):
        self.table = table
        self.reference_date = reference_date
        self.currency = currency
        self.adjustments = adjustments
        self.models = models

    @classmethod
    def load(cls, path: str | Path = "config/pricing.yaml") -> PricingRegistry:
        raw = load_yaml(path)
        return cls(
            table=str(raw["pricing_table"]),
            reference_date=str(raw["reference_date"]),
            currency=str(raw["currency"]),
            adjustments={
                key: float(value) for key, value in raw["execution_adjustments"].items()
            },
            models={
                key: PriceEntry.model_validate(value)
                for key, value in raw["models"].items()
            },
        )

    def price(self, key: str) -> PriceEntry:
        try:
            return self.models[key]
        except KeyError as exc:
            raise ValueError(f"no pricing entry for model key {key!r}") from exc

    def multiplier(self, execution_mode: str) -> float:
        try:
            return self.adjustments[execution_mode]
        except KeyError as exc:
            raise ValueError(
                f"no pricing adjustment for execution mode {execution_mode!r}"
            ) from exc


def _nested(source: Any, *names: str, default: int = 0) -> int:
    current = source
    for name in names:
        if isinstance(current, dict):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
        if current is None:
            return default
    return int(current)


def build_usage_record(
    *,
    conversation_id: str,
    response_id: str,
    model: str,
    pricing_key: str,
    execution_mode: str,
    usage: Any,
    registry: PricingRegistry,
) -> UsageRecord:
    input_tokens = _nested(usage, "input_tokens")
    cached_tokens = _nested(usage, "input_tokens_details", "cached_tokens")
    output_tokens = _nested(usage, "output_tokens")
    reasoning_tokens = _nested(usage, "output_tokens_details", "reasoning_tokens")
    total_tokens = _nested(usage, "total_tokens", default=input_tokens + output_tokens)
    if cached_tokens > input_tokens:
        raise ValueError("cached input tokens cannot exceed input tokens")
    uncached_tokens = input_tokens - cached_tokens
    price = registry.price(pricing_key)
    input_cost = uncached_tokens / 1_000_000 * price.input_per_1m
    cached_cost = cached_tokens / 1_000_000 * price.cached_input_per_1m
    output_cost = output_tokens / 1_000_000 * price.output_per_1m
    standard_cost = input_cost + cached_cost + output_cost
    multiplier = registry.multiplier(execution_mode)
    return UsageRecord(
        conversation_id=conversation_id,
        response_id=response_id,
        model=model,
        execution_mode=execution_mode,
        input_tokens=input_tokens,
        input_tokens_details={"cached_tokens": cached_tokens},
        cached_input_tokens=cached_tokens,
        uncached_input_tokens=uncached_tokens,
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": reasoning_tokens},
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        input_cost=input_cost,
        cached_input_cost=cached_cost,
        output_cost=output_cost,
        standard_estimated_cost=standard_cost,
        execution_multiplier=multiplier,
        effective_execution_cost=standard_cost * multiplier,
        currency=price.currency,
        pricing_table=registry.table,
    )


def usage_summary(
    records: list[UsageRecord],
    *,
    requested: int,
    failed: int,
    model: str,
    execution_mode: str,
    pricing_table: str,
) -> dict[str, Any]:
    # Totals include rejected attempts; per-conversation averages divide those
    # totals by accepted conversations, so they include the cost of retries.
    successful = sum(not record.rejected for record in records)
    rejected = [record for record in records if record.rejected]
    costs = [record.effective_execution_cost for record in records]
    mean_cost = sum(costs) / successful if successful else 0.0

    def average(field: str) -> float:
        return (
            sum(getattr(record, field) for record in records) / successful
            if successful
            else 0.0
        )

    return {
        "requested_conversations": requested,
        "successful_conversations": successful,
        "failed_conversations": failed,
        "rejected_attempts": len(rejected),
        "rejected_attempts_cost": sum(r.effective_execution_cost for r in rejected),
        "total_input_tokens": sum(record.input_tokens for record in records),
        "total_cached_input_tokens": sum(
            record.cached_input_tokens for record in records
        ),
        "total_uncached_input_tokens": sum(
            record.uncached_input_tokens for record in records
        ),
        "total_output_tokens": sum(record.output_tokens for record in records),
        "total_reasoning_tokens": sum(record.reasoning_tokens for record in records),
        "total_tokens": sum(record.total_tokens for record in records),
        "total_estimated_cost": sum(costs),
        "total_standard_estimated_cost": sum(
            record.standard_estimated_cost for record in records
        ),
        "total_effective_execution_cost": sum(costs),
        "average_input_tokens_per_conversation": average("input_tokens"),
        "average_output_tokens_per_conversation": average("output_tokens"),
        "average_total_tokens_per_conversation": average("total_tokens"),
        "average_cost_per_conversation": mean_cost,
        "min_cost_per_conversation": min(costs) if costs else 0.0,
        "max_cost_per_conversation": max(costs) if costs else 0.0,
        "model": model,
        "execution_mode": execution_mode,
        "pricing_table": pricing_table,
        "cost_currency": records[0].currency if records else "USD",
        "projected_costs": {
            "100_conversations": mean_cost * 100,
            "500_conversations": mean_cost * 500,
            "1000_conversations": mean_cost * 1000,
            "label": "projections based on observed mean; not actual charges",
        },
    }

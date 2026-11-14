"""Pydantic schemas for model output, merged generations, usage, and failures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthetic_dataset_gen.phase1.models import CatalogRow


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedTurn(StrictModel):
    turn: int = Field(ge=1)
    user: str = Field(min_length=1)
    assistant: str = Field(min_length=1)


class GeneratedConversation(StrictModel):
    turns: list[GeneratedTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def consecutive_turns(self) -> GeneratedConversation:
        observed = [turn.turn for turn in self.turns]
        expected = list(range(1, len(self.turns) + 1))
        if observed != expected:
            raise ValueError(f"turn numbers must be consecutive: expected {expected}")
        return self


class GenerationRecord(StrictModel):
    conversation_id: str
    response_id: str
    requested_model: str
    resolved_model: str
    returned_model: str | None
    prompt_template_name: str
    phase1_spec: CatalogRow
    generated: GeneratedConversation
    # Pair members only: False while the other member has no accepted generation.
    pair_complete: bool | None = None
    # False when the post-generation check (ConversationChecker) rejected this
    # output; the sample is still stored so failures can be inspected directly.
    check_passed: bool = True
    check_problems: list[str] = Field(default_factory=list)


class InputTokenDetails(StrictModel):
    cached_tokens: int = Field(ge=0)


class OutputTokenDetails(StrictModel):
    reasoning_tokens: int = Field(ge=0)


class UsageRecord(StrictModel):
    conversation_id: str
    response_id: str
    model: str
    execution_mode: str
    input_tokens: int = Field(ge=0)
    input_tokens_details: InputTokenDetails
    cached_input_tokens: int = Field(ge=0)
    uncached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    output_tokens_details: OutputTokenDetails
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_cost: float = Field(ge=0)
    cached_input_cost: float = Field(ge=0)
    output_cost: float = Field(ge=0)
    standard_estimated_cost: float = Field(ge=0)
    execution_multiplier: float = Field(gt=0)
    effective_execution_cost: float = Field(ge=0)
    currency: str
    pricing_table: str
    # True for an attempt whose output was not accepted (failed parsing or a
    # check); its tokens were still billed.
    rejected: bool = False


class ErrorRecord(StrictModel):
    conversation_id: str
    error_type: str
    safe_message: str
    attempt_count: int = Field(ge=1)


class ParsedResponse(StrictModel):
    response_id: str
    returned_model: str | None
    output: GeneratedConversation
    usage: dict[str, Any]

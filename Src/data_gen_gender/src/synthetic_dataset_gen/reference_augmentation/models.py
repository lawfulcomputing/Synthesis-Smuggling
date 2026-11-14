"""Strict schemas for the reference-augmentation catalog and outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation

Gender = Literal["M", "F"]
Split = Literal["train", "validation", "test"]
Category = Literal[
    "other_reference_only",
    "reference_before_self_reveal",
    "reference_after_self_reveal",
    "self_reference_control",
]
Scope = Literal["other", "self"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceSpec(StrictModel):
    conversation_id: str = Field(min_length=1)
    category: Category
    split: Split
    topic: str = Field(min_length=1)
    num_turns: int = Field(ge=2)
    reference_turn: int = Field(ge=1)
    reference_scope: Scope
    reference_family: str = Field(min_length=1)
    reference_style: str = Field(min_length=1)
    reference_term: str = Field(min_length=1)
    reference_gender: Gender
    counterpart_term: str = Field(min_length=1)
    target_gender: Gender | None
    reveal_turn: int | None
    reveal_term: str | None
    reveal_form_family: str | None
    reveal_anchor: str | None = None
    contrast_group_id: str | None = None

    @model_validator(mode="after")
    def valid_timing(self) -> ReferenceSpec:
        if self.reference_turn > self.num_turns:
            raise ValueError("reference_turn exceeds num_turns")
        if self.category == "other_reference_only":
            if self.target_gender is not None or self.reveal_turn is not None:
                raise ValueError("other_reference_only must remain neutral")
        else:
            if self.target_gender is None or self.reveal_turn is None:
                raise ValueError("gendered categories require a self reveal")
            if not self.reveal_term or not self.reveal_form_family:
                raise ValueError("gendered categories require reveal metadata")
            if self.reveal_turn > self.num_turns:
                raise ValueError("reveal_turn exceeds num_turns")
        if self.category == "reference_before_self_reveal" and not (
            self.reference_turn < (self.reveal_turn or 0)
        ):
            raise ValueError("before category requires reference_turn < reveal_turn")
        if self.category == "reference_after_self_reveal" and not (
            (self.reveal_turn or 0) < self.reference_turn
        ):
            raise ValueError("after category requires reveal_turn < reference_turn")
        if self.category == "self_reference_control":
            if self.reference_scope != "self":
                raise ValueError("self control requires self scope")
            if self.reference_turn != self.reveal_turn:
                raise ValueError("self control reference and reveal turns must match")
        elif self.reference_scope != "other":
            raise ValueError("non-control categories require other-person scope")
        return self


class ReferenceGenerationRecord(StrictModel):
    conversation_id: str
    response_id: str
    requested_model: str
    resolved_model: str
    returned_model: str | None
    prompt_template_name: str
    phase1_spec: CatalogRow
    augmentation_spec: ReferenceSpec
    generated: GeneratedConversation
    pair_complete: bool | None = None
    check_passed: bool = True
    check_problems: list[str] = Field(default_factory=list)
    source_run: str

"""Strict schemas for the possessive-partner recovery set."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation

Gender = Literal["M", "F"]
Split = Literal["train", "validation", "test"]
Category = Literal[
    "direct_partner_reveal",
    "nested_partner_neutral",
    "self_role_control",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecoverySpec(StrictModel):
    conversation_id: str = Field(min_length=1)
    recovery_set: str = Field(min_length=1)
    category: Category
    split: Split
    contrast_group_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    num_turns: int = Field(ge=2)
    cue_turn: int = Field(ge=1)
    cue_bucket: Literal["immediate", "early", "mid", "late"]
    surface_frame: str = Field(min_length=1)
    partner_term: str = Field(min_length=1)
    partner_gender: Gender
    counterpart_term: str = Field(min_length=1)
    target_gender: Gender | None

    @model_validator(mode="after")
    def valid_semantics(self) -> RecoverySpec:
        if self.cue_turn > self.num_turns:
            raise ValueError("cue_turn exceeds num_turns")
        if self.category == "nested_partner_neutral":
            if self.target_gender is not None:
                raise ValueError("nested controls must remain neutral")
        elif self.target_gender is None:
            raise ValueError("positive/control rows require a target gender")
        if self.category == "direct_partner_reveal":
            expected = "F" if self.partner_gender == "M" else "M"
            if self.target_gender != expected:
                raise ValueError("direct partner reveal must use opposite-gender label")
        if self.category == "self_role_control" and (
            self.target_gender != self.partner_gender
        ):
            raise ValueError("self-role control must use the role term's gender")
        return self


class RecoveryGenerationRecord(StrictModel):
    conversation_id: str
    response_id: str
    requested_model: str
    resolved_model: str
    returned_model: str | None
    prompt_template_name: str
    phase1_spec: CatalogRow
    recovery_spec: RecoverySpec
    generated: GeneratedConversation
    pair_complete: bool | None = None
    check_passed: bool = True
    check_problems: list[str] = Field(default_factory=list)
    source_run: str

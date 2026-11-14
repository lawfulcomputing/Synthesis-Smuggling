"""Validated catalog and prefix row schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Belief = Literal["U", "M", "F"]
RevealEvent = Literal["NONE", "M_REVEAL", "F_REVEAL"]
TrajectoryFamily = Literal["never_reveal", "single_reveal"]
Gender = Literal["M", "F"]
RevealBucket = Literal["immediate", "early", "mid", "late"]
Split = Literal["train", "validation", "test"]


class CatalogRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    catalog_seed: int
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    num_turns: int = Field(ge=1)
    trajectory_family: TrajectoryFamily
    target_gender: Gender | None
    reveal_turn: int | None
    reveal_bucket: RevealBucket | None
    normalized_reveal_position: float | None = Field(default=None, ge=0, le=1)
    topic: str = Field(min_length=1)
    reveal_form_family: str | None
    # Optional so catalogs written before spec 1.1.0 still load.
    reveal_term: str | None = None
    # A gender-neutral word for the person the reveal is framed around
    # (e.g. "child" for father/mother); None when the form has no anchor.
    anchor: str | None = None
    counterfactual_pair_id: str | None
    pair_member: Gender | None
    belief_trajectory: list[Belief]
    reveal_event_trajectory: list[RevealEvent]
    split: Split

    @model_validator(mode="after")
    def internally_consistent(self) -> CatalogRow:
        if len(self.belief_trajectory) != self.num_turns:
            raise ValueError("belief_trajectory length must equal num_turns")
        if len(self.reveal_event_trajectory) != self.num_turns:
            raise ValueError("reveal_event_trajectory length must equal num_turns")
        if self.trajectory_family == "never_reveal":
            nullable = (
                self.target_gender,
                self.reveal_turn,
                self.reveal_bucket,
                self.normalized_reveal_position,
                self.reveal_form_family,
                self.reveal_term,
                self.anchor,
                self.counterfactual_pair_id,
                self.pair_member,
            )
            if any(value is not None for value in nullable):
                raise ValueError(
                    "never_reveal rows must not expose gender/reveal metadata"
                )
        else:
            required = (
                self.target_gender,
                self.reveal_turn,
                self.reveal_bucket,
                self.normalized_reveal_position,
                self.reveal_form_family,
            )
            if any(value is None for value in required):
                raise ValueError("single_reveal rows require gender/reveal metadata")
            if self.counterfactual_pair_id is None:
                if self.pair_member is not None:
                    raise ValueError("unpaired rows must not set pair_member")
            elif self.target_gender != self.pair_member:
                raise ValueError(
                    "paired rows require pair_member equal to target_gender"
                )
        return self


class PrefixRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    prefix_turn: int = Field(ge=1)
    num_turns: int = Field(ge=1)
    retained_belief: Belief
    split: Split
    trajectory_family: TrajectoryFamily
    target_gender: Gender | None

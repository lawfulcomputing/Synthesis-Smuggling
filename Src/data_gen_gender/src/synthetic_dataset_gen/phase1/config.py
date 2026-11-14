"""Configuration models and TOML loading."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    num_conversations: int = Field(gt=0)
    seed: int
    # Proportion of single-reveal rows assigned to complete M/F pairs.
    # Rounding to whole pairs happens deterministically in the generator.
    counterfactual_fraction: Probability = 1.0


class TrajectoryMixConfig(StrictModel):
    never_reveal: Probability
    single_reveal: Probability


class LengthConfig(StrictModel):
    mean: float
    sd: float = Field(gt=0)
    min_turns: int = Field(ge=1)
    max_turns: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_bounds(self) -> LengthConfig:
        if self.min_turns > self.max_turns:
            raise ValueError("length.min_turns must not exceed length.max_turns")
        if not self.min_turns <= self.mean <= self.max_turns:
            raise ValueError("length.mean must fall within configured bounds")
        return self


class RevealBucketConfig(StrictModel):
    immediate: Probability
    early: Probability
    mid: Probability
    late: Probability


class TopicsConfig(StrictModel):
    candidate: list[str] = Field(min_length=1)
    active: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def active_is_candidate(self) -> TopicsConfig:
        if len(set(self.candidate)) != len(self.candidate):
            raise ValueError("topics.candidate must not contain duplicates")
        if len(set(self.active)) != len(self.active):
            raise ValueError("topics.active must not contain duplicates")
        unknown = set(self.active) - set(self.candidate)
        if unknown:
            raise ValueError(f"active topics not in candidate pool: {sorted(unknown)}")
        return self


class RevealFormDefinition(StrictModel):
    description: str = Field(min_length=1)
    term_pairs: list[tuple[str, str]] = Field(min_length=1)
    # A gender-neutral anchor per term pair, same order as term_pairs.
    # Empty when no pair in this form has one (e.g. explicit_identity).
    anchors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_term_pairs(self) -> RevealFormDefinition:
        for male, female in self.term_pairs:
            if not male.strip() or not female.strip():
                raise ValueError("reveal form terms must not be blank")
            if male.casefold() == female.casefold():
                raise ValueError(
                    f"term pair {[male, female]} has identical M and F terms"
                )
        if len(set(self.term_pairs)) != len(self.term_pairs):
            raise ValueError("reveal form term_pairs must not contain duplicates")
        if self.anchors:
            if len(self.anchors) != len(self.term_pairs):
                raise ValueError("anchors must have one entry per term pair")
            if any(not anchor.strip() for anchor in self.anchors):
                raise ValueError("anchor must not be blank")
        return self

    def anchor_for(self, pair: tuple[str, str]) -> str | None:
        """The configured anchor for this M/F term pair, or None."""
        if not self.anchors:
            return None
        return self.anchors[self.term_pairs.index(pair)]


class RevealFormsConfig(StrictModel):
    active: list[str] = Field(min_length=1)
    definitions: dict[str, RevealFormDefinition]

    weights: dict[str, Probability] = Field(default_factory=dict)

    @model_validator(mode="after")
    def active_forms_are_defined(self) -> RevealFormsConfig:
        if len(set(self.active)) != len(self.active):
            raise ValueError("reveal_forms.active must not contain duplicates")
        undefined = [form for form in self.active if form not in self.definitions]
        if undefined:
            raise ValueError(f"active reveal forms without a definition: {undefined}")

        if self.weights:
            if set(self.weights) != set(self.active):
                raise ValueError("reveal_forms.weights must contain every active form")
            if abs(sum(self.weights.values()) - 1.0) > 1e-9:
                raise ValueError("reveal_forms.weights probabilities must sum to 1.0")
        return self

    def active_weights(self) -> dict[str, float]:
        """Configured form probabilities, or uniform weights when omitted."""
        if self.weights:
            return dict(self.weights)
        probability = 1.0 / len(self.active)
        return {form: probability for form in self.active}


class SplitsConfig(StrictModel):
    train: Probability
    validation: Probability
    test: Probability


class OutputsConfig(StrictModel):
    artifact_dir: str = Field(min_length=1)
    plot_dir: str = Field(min_length=1)


class Phase1Config(StrictModel):
    dataset: DatasetConfig
    trajectory_mix: TrajectoryMixConfig
    length: LengthConfig
    reveal_buckets: RevealBucketConfig
    topics: TopicsConfig
    reveal_forms: RevealFormsConfig
    splits: SplitsConfig
    outputs: OutputsConfig

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> Phase1Config:
        groups = {
            "trajectory_mix": self.trajectory_mix.model_dump(),
            "reveal_buckets": self.reveal_buckets.model_dump(),
            "splits": self.splits.model_dump(),
        }
        for name, values in groups.items():
            if abs(sum(values.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} probabilities must sum to 1.0")
        required_min = {"immediate": 1, "early": 2, "mid": 4, "late": 6}
        for bucket, minimum in required_min.items():
            if (
                getattr(self.reveal_buckets, bucket) > 0
                and self.length.max_turns < minimum
            ):
                raise ValueError(
                    f"reveal bucket {bucket!r} requires max_turns >= {minimum}"
                )
        return self

    def artifact_path(self, name: str) -> Path:
        """Path of a named Phase 1 artifact inside the configured artifact_dir."""
        return Path(self.outputs.artifact_dir) / name

    @property
    def catalog_path(self) -> Path:
        return self.artifact_path("catalog.jsonl")

    @property
    def semantic_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> Phase1Config:
    """Load and validate a Phase 1 TOML configuration file."""
    with Path(path).open("rb") as handle:
        return Phase1Config.model_validate(tomllib.load(handle))

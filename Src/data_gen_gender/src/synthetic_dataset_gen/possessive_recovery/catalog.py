"""Deterministic three-way contrast design for possessive-partner recovery."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.possessive_recovery.models import RecoverySpec

CATEGORIES = (
    "direct_partner_reveal",
    "nested_partner_neutral",
    "self_role_control",
)
SPLITS = ("train", "validation", "test")
FRAMES = {
    "direct_partner_reveal": (
        "sentence_initial",
        "embedded_object",
        "coordination",
        "possessive_genitive",
        "prepositional",
        "reported_preference",
    ),
    "nested_partner_neutral": (
        "family_possessor",
        "friend_possessor",
        "named_possessor",
    ),
    "self_role_control": (
        "copular_self_role",
        "as_self_role",
        "relational_self_role",
    ),
}


class PartnerPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    male: str
    female: str


class RecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recovery_set: str
    seed: int
    category_counts: dict[str, int]
    category_term_counts: dict[str, dict[str, int]] | None = None
    split_fractions: dict[str, float]
    partner_pairs: list[PartnerPair] = Field(min_length=1)
    length: dict[str, int]
    cue_lead_in_turns: list[int] | None = None
    generation: dict[str, Any]
    outputs: dict[str, str]

    @model_validator(mode="after")
    def valid_design(self) -> RecoveryConfig:
        if set(self.category_counts) != set(CATEGORIES):
            raise ValueError("category_counts must contain exactly three strata")
        if set(self.split_fractions) != set(SPLITS):
            raise ValueError("split_fractions must contain train/validation/test")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        terms = [
            term for pair in self.partner_pairs for term in (pair.male, pair.female)
        ]
        if len(set(terms)) != len(terms):
            raise ValueError("partner terms must be unique")
        if self.category_term_counts is not None:
            if set(self.category_term_counts) != set(CATEGORIES):
                raise ValueError(
                    "category_term_counts must contain exactly three strata"
                )
            for category, term_counts in self.category_term_counts.items():
                if set(term_counts) != set(terms):
                    raise ValueError(f"{category} must specify every partner term")
                if any(count < 0 for count in term_counts.values()):
                    raise ValueError("per-term counts cannot be negative")
                if sum(term_counts.values()) != self.category_counts[category]:
                    raise ValueError(f"{category} per-term counts do not match total")
        else:
            term_count = len(terms)
            for category, count in self.category_counts.items():
                if count % term_count:
                    raise ValueError(
                        f"{category} is not divisible across partner terms"
                    )
        for category in CATEGORIES:
            for term in terms:
                per_term = self.count_for(category, term)
                for split, fraction in self.split_fractions.items():
                    if abs(per_term * fraction - round(per_term * fraction)) > 1e-9:
                        raise ValueError(
                            f"{category}/{term} cannot split exactly for {split}"
                        )
        if self.cue_lead_in_turns is not None:
            if not self.cue_lead_in_turns or any(
                turns < 0 for turns in self.cue_lead_in_turns
            ):
                raise ValueError("cue_lead_in_turns must contain nonnegative values")
            if max(self.cue_lead_in_turns) >= self.length["max_turns"]:
                raise ValueError("cue lead-in must leave room for the cue turn")
        return self

    def count_for(self, category: str, term: str) -> int:
        if self.category_term_counts is not None:
            return self.category_term_counts[category][term]
        return self.category_counts[category] // (2 * len(self.partner_pairs))

    @property
    def total_conversations(self) -> int:
        return sum(self.category_counts.values())

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_recovery_config(path: str | Path) -> RecoveryConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return RecoveryConfig.model_validate(yaml.safe_load(handle))


def _split_for(index: int, per_term: int, config: RecoveryConfig) -> str:
    train_end = round(per_term * config.split_fractions["train"])
    validation_end = train_end + round(per_term * config.split_fractions["validation"])
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _cue_position(
    index: int, num_turns: int, lead_in_turns: list[int] | None = None
) -> tuple[str, int]:
    if lead_in_turns is not None:
        cue_turn = lead_in_turns[index % len(lead_in_turns)] + 1
        if cue_turn > num_turns:
            raise ValueError("configured cue turn exceeds conversation length")
        if cue_turn == 1:
            bucket = "immediate"
        elif cue_turn == num_turns:
            bucket = "late"
        elif cue_turn <= (num_turns + 1) // 2:
            bucket = "early"
        else:
            bucket = "mid"
        return bucket, cue_turn
    bucket = ("immediate", "early", "mid", "late")[index % 4]
    positions = {
        "immediate": 1,
        "early": 2,
        "mid": max(2, (num_turns + 1) // 2),
        "late": num_turns,
    }
    return bucket, positions[bucket]


def _phase1_spec(spec: RecoverySpec, config: RecoveryConfig) -> CatalogRow:
    if spec.target_gender is None:
        return CatalogRow(
            conversation_id=spec.conversation_id,
            catalog_seed=config.seed,
            config_sha256=config.semantic_hash,
            num_turns=spec.num_turns,
            trajectory_family="never_reveal",
            target_gender=None,
            reveal_turn=None,
            reveal_bucket=None,
            normalized_reveal_position=None,
            topic=spec.topic,
            reveal_form_family=None,
            reveal_term=None,
            anchor=None,
            counterfactual_pair_id=None,
            pair_member=None,
            belief_trajectory=["U"] * spec.num_turns,
            reveal_event_trajectory=["NONE"] * spec.num_turns,
            split=spec.split,
        )
    beliefs = [
        "U" if turn < spec.cue_turn else spec.target_gender
        for turn in range(1, spec.num_turns + 1)
    ]
    events = ["NONE"] * spec.num_turns
    events[spec.cue_turn - 1] = f"{spec.target_gender}_REVEAL"
    form = (
        "possessive_partner_inference"
        if spec.category == "direct_partner_reveal"
        else "gendered_relationship_role"
    )
    return CatalogRow(
        conversation_id=spec.conversation_id,
        catalog_seed=config.seed,
        config_sha256=config.semantic_hash,
        num_turns=spec.num_turns,
        trajectory_family="single_reveal",
        target_gender=spec.target_gender,
        reveal_turn=spec.cue_turn,
        reveal_bucket=spec.cue_bucket,
        normalized_reveal_position=spec.cue_turn / spec.num_turns,
        topic=spec.topic,
        reveal_form_family=form,
        reveal_term=spec.partner_term,
        anchor="partner",
        counterfactual_pair_id=None,
        pair_member=None,
        belief_trajectory=beliefs,
        reveal_event_trajectory=events,
        split=spec.split,
    )


def build_catalog(
    config: RecoveryConfig, *, phase1_path: str | Path = "config.toml"
) -> list[tuple[RecoverySpec, CatalogRow]]:
    phase1 = load_config(phase1_path)
    rng = random.Random(config.seed)
    topics = list(phase1.topics.active)
    rng.shuffle(topics)
    terms = []
    for pair_index, pair in enumerate(config.partner_pairs, start=1):
        terms.extend(
            [
                (f"partner_pair_{pair_index}", pair.male, "M", pair.female),
                (f"partner_pair_{pair_index}", pair.female, "F", pair.male),
            ]
        )

    rows: list[tuple[RecoverySpec, CatalogRow]] = []
    serial = 0
    for category in CATEGORIES:
        for term_index, (pair_id, term, gender, counterpart) in enumerate(terms):
            per_term = config.count_for(category, term)
            for occurrence in range(per_term):
                serial += 1
                split = _split_for(occurrence, per_term, config)
                num_turns = config.length["min_turns"] + (
                    (occurrence + term_index)
                    % (config.length["max_turns"] - config.length["min_turns"] + 1)
                )
                if config.cue_lead_in_turns is not None:
                    lead_in = config.cue_lead_in_turns[
                        (occurrence + term_index) % len(config.cue_lead_in_turns)
                    ]
                    num_turns = max(num_turns, lead_in + 1)
                bucket, cue_turn = _cue_position(
                    occurrence + term_index, num_turns, config.cue_lead_in_turns
                )
                if category == "direct_partner_reveal":
                    target = "F" if gender == "M" else "M"
                elif category == "self_role_control":
                    target = gender
                else:
                    target = None
                frame = FRAMES[category][
                    (occurrence + term_index) % len(FRAMES[category])
                ]
                spec = RecoverySpec(
                    conversation_id=f"posrec_{serial:04d}",
                    recovery_set=config.recovery_set,
                    category=category,
                    split=split,
                    contrast_group_id=(
                        f"{pair_id}_{term_index:02d}_{occurrence % 10:02d}"
                    ),
                    topic=topics[(serial - 1) % len(topics)],
                    num_turns=num_turns,
                    cue_turn=cue_turn,
                    cue_bucket=bucket,
                    surface_frame=frame,
                    partner_term=term,
                    partner_gender=gender,
                    counterpart_term=counterpart,
                    target_gender=target,
                )
                rows.append((spec, _phase1_spec(spec, config)))
    if len(rows) != config.total_conversations:
        raise AssertionError("recovery catalog size mismatch")
    return rows

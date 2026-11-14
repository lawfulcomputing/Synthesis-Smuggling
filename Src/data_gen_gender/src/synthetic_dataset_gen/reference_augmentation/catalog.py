"""Deterministic, balanced design for reference-attribution augmentation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthetic_dataset_gen.phase1.config import Phase1Config, load_config
from synthetic_dataset_gen.phase1.models import CatalogRow

SPLITS = ("train", "validation", "test")
CATEGORIES = (
    "other_reference_only",
    "reference_before_self_reveal",
    "reference_after_self_reveal",
    "self_reference_control",
)


class ExtraPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    male: str
    female: str
    style: str


class ReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    total_conversations: int = Field(gt=0)
    category_counts: dict[str, int]
    split_fractions: dict[str, float]
    length: dict[str, int]
    generation: dict[str, Any]
    outputs: dict[str, str]
    extra_reference_pairs: list[ExtraPair]

    @model_validator(mode="after")
    def totals_match(self) -> ReferenceConfig:
        if set(self.category_counts) != set(CATEGORIES):
            raise ValueError("category_counts must contain exactly the four strata")
        if sum(self.category_counts.values()) != self.total_conversations:
            raise ValueError("category counts do not sum to total_conversations")
        if set(self.split_fractions) != set(SPLITS):
            raise ValueError("split_fractions must contain train/validation/test")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        for category, count in self.category_counts.items():
            for split, fraction in self.split_fractions.items():
                if abs(count * fraction - round(count * fraction)) > 1e-9:
                    raise ValueError(
                        f"{category} cannot be split exactly for {split}={fraction}"
                    )
        return self

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Term:
    pair_id: str
    family: str
    term: str
    gender: str
    counterpart: str
    style: str
    anchor: str | None
    self_capable: bool


def load_reference_config(path: str | Path) -> ReferenceConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return ReferenceConfig.model_validate(yaml.safe_load(handle))


def _style_for_family(family: str) -> str:
    if "relationship" in family or "family" in family:
        return "possessive_role"
    if family == "gendered_social_role":
        return "third_person_role"
    return "named_descriptor"


def term_inventory(
    config: ReferenceConfig, phase1: Phase1Config
) -> tuple[list[Term], list[Term]]:
    """Return (all reference terms, terms suitable for explicit self reveals)."""
    pairs: list[tuple[str, str, str, str, str | None, bool]] = []
    seen: set[tuple[str, str]] = set()
    for family in phase1.reveal_forms.active:
        definition = phase1.reveal_forms.definitions[family]
        for index, (male, female) in enumerate(definition.term_pairs):
            key = (male.casefold(), female.casefold())
            if key in seen:
                continue
            seen.add(key)
            anchor = definition.anchors[index] if definition.anchors else None
            pairs.append(
                (
                    family,
                    male,
                    female,
                    _style_for_family(family),
                    anchor,
                    True,
                )
            )
    for pair in config.extra_reference_pairs:
        key = (pair.male.casefold(), pair.female.casefold())
        if key in seen:
            continue
        seen.add(key)
        self_capable = pair.family == "direct_identity"
        pairs.append(
            (pair.family, pair.male, pair.female, pair.style, None, self_capable)
        )

    references: list[Term] = []
    self_terms: list[Term] = []
    for pair_index, (family, male, female, style, anchor, capable) in enumerate(pairs):
        pair_id = f"term_pair_{pair_index + 1:02d}"
        for gender, term, counterpart in (
            ("M", male, female),
            ("F", female, male),
        ):
            entry = Term(
                pair_id=pair_id,
                family=family,
                term=term,
                gender=gender,
                counterpart=counterpart,
                style=style,
                anchor=anchor,
                self_capable=capable,
            )
            references.append(entry)
            if capable:
                self_terms.append(entry)
    return references, self_terms


def _split_schedule(
    count: int, fractions: dict[str, float], rng: random.Random
) -> list[str]:
    schedule: list[str] = []
    for split in SPLITS:
        schedule.extend([split] * round(count * fractions[split]))
    if len(schedule) != count:
        raise AssertionError("split rounding changed category size")
    rng.shuffle(schedule)
    return schedule


def _bucket(turn: int, num_turns: int) -> str:
    ratio = turn / num_turns
    if turn == 1:
        return "immediate"
    if ratio <= 0.4:
        return "early"
    if ratio <= 0.75:
        return "mid"
    return "late"


def _phase1_spec(spec: Any, config: ReferenceConfig) -> CatalogRow:
    if spec.target_gender is None:
        beliefs = ["U"] * spec.num_turns
        events = ["NONE"] * spec.num_turns
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
            belief_trajectory=beliefs,
            reveal_event_trajectory=events,
            split=spec.split,
        )
    beliefs = [
        "U" if turn < spec.reveal_turn else spec.target_gender
        for turn in range(1, spec.num_turns + 1)
    ]
    events = ["NONE"] * spec.num_turns
    events[spec.reveal_turn - 1] = f"{spec.target_gender}_REVEAL"
    return CatalogRow(
        conversation_id=spec.conversation_id,
        catalog_seed=config.seed,
        config_sha256=config.semantic_hash,
        num_turns=spec.num_turns,
        trajectory_family="single_reveal",
        target_gender=spec.target_gender,
        reveal_turn=spec.reveal_turn,
        reveal_bucket=_bucket(spec.reveal_turn, spec.num_turns),
        normalized_reveal_position=spec.reveal_turn / spec.num_turns,
        topic=spec.topic,
        reveal_form_family=spec.reveal_form_family,
        reveal_term=spec.reveal_term,
        anchor=spec.reveal_anchor,
        counterfactual_pair_id=None,
        pair_member=None,
        belief_trajectory=beliefs,
        reveal_event_trajectory=events,
        split=spec.split,
    )


def build_catalog(
    config: ReferenceConfig,
    *,
    phase1_path: str | Path = "config.toml",
) -> list[tuple[Any, CatalogRow]]:
    # Local import avoids a models/catalog circular import at module load time.
    from synthetic_dataset_gen.reference_augmentation.models import ReferenceSpec

    phase1 = load_config(phase1_path)
    references, self_terms = term_inventory(config, phase1)
    rng = random.Random(config.seed)
    topics = list(phase1.topics.active)
    rng.shuffle(topics)
    reference_orders: dict[str, list[Term]] = {}
    self_orders: dict[str, list[Term]] = {}
    split_orders: dict[str, list[str]] = {}
    for category, count in config.category_counts.items():
        repeated_refs = [references[i % len(references)] for i in range(count)]
        repeated_self = [self_terms[i % len(self_terms)] for i in range(count)]
        rng.shuffle(repeated_refs)
        rng.shuffle(repeated_self)
        reference_orders[category] = repeated_refs
        self_orders[category] = repeated_self
        split_orders[category] = _split_schedule(count, config.split_fractions, rng)

    specs: list[ReferenceSpec] = []
    serial = 0
    for category in CATEGORIES:
        count = config.category_counts[category]
        for index in range(count):
            serial += 1
            reference = reference_orders[category][index]
            num_turns = rng.randint(
                config.length["min_turns"], config.length["max_turns"]
            )
            topic = topics[(serial - 1) % len(topics)]
            split = split_orders[category][index]
            target_gender = None
            reveal_turn = None
            reveal_term = None
            reveal_family = None
            reveal_anchor = None
            scope = "other"
            if category == "other_reference_only":
                reference_turn = rng.randint(1, num_turns - 1)
            elif category == "reference_before_self_reveal":
                reference_turn = rng.randint(1, num_turns - 2)
                reveal_turn = rng.randint(reference_turn + 1, num_turns)
            elif category == "reference_after_self_reveal":
                reveal_turn = rng.randint(1, num_turns - 1)
                reference_turn = rng.randint(reveal_turn + 1, num_turns)
            else:
                scope = "self"
                reveal_turn = rng.randint(1, num_turns - 1)
                reference_turn = reveal_turn
                reference = self_orders[category][index]

            if category != "other_reference_only":
                if category == "self_reference_control":
                    self_term = reference
                    target_gender = reference.gender
                else:
                    # Balance the user's target exactly within each timing
                    # stratum. The independently balanced reference inventory
                    # supplies both same- and opposite-gender distractors.
                    target_gender = "M" if index % 2 == 0 else "F"
                    candidates = [t for t in self_terms if t.gender == target_gender]
                    self_term = candidates[index % len(candidates)]
                reveal_term = self_term.term
                reveal_family = self_term.family
                reveal_anchor = self_term.anchor

            spec = ReferenceSpec(
                conversation_id=f"refaug_{serial:04d}",
                category=category,
                split=split,
                topic=topic,
                num_turns=num_turns,
                reference_turn=reference_turn,
                reference_scope=scope,
                reference_family=reference.family,
                reference_style=reference.style,
                reference_term=reference.term,
                reference_gender=reference.gender,
                counterpart_term=reference.counterpart,
                target_gender=target_gender,
                reveal_turn=reveal_turn,
                reveal_term=reveal_term,
                reveal_form_family=reveal_family,
                reveal_anchor=reveal_anchor,
                contrast_group_id=(
                    f"self_control_{index + 1:03d}"
                    if category == "self_reference_control"
                    else None
                ),
            )
            specs.append(spec)

    if len(specs) != config.total_conversations:
        raise AssertionError("catalog size does not match configuration")
    return [(spec, _phase1_spec(spec, config)) for spec in specs]

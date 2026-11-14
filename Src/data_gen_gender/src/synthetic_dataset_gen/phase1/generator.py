"""Deterministic, constraint-aware catalog construction."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.models import CatalogRow, RevealBucket
from synthetic_dataset_gen.phase1.trajectory import compute_trajectory

T = TypeVar("T")
BUCKET_TURNS: dict[str, tuple[int, ...]] = {
    "immediate": (1,),
    "early": (2, 3),
    "mid": (4, 5),
    "late": (6, 7),
}


def largest_remainder(total: int, probabilities: Mapping[str, float]) -> dict[str, int]:
    """Allocate an integer total with deterministic largest-remainder rounding."""
    raw = {key: total * value for key, value in probabilities.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(probabilities, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def balanced_values(counts: Mapping[T, int], rng: random.Random) -> list[T]:
    values = [value for value, count in counts.items() for _ in range(count)]
    rng.shuffle(values)
    return values


def balanced_cycle(values: Sequence[T], count: int, rng: random.Random) -> list[T]:
    """Return a shuffled sequence in which item counts differ by at most one."""
    if not values:
        raise ValueError("cannot sample from an empty sequence")
    allocation = largest_remainder(
        count, {str(i): 1 / len(values) for i in range(len(values))}
    )
    result = [values[int(index)] for index, n in allocation.items() for _ in range(n)]
    rng.shuffle(result)
    return result


def sample_truncated_length(
    config: Phase1Config, rng: random.Random, minimum: int | None = None
) -> int:
    """Sample a rounded normal variate, rejecting values outside active bounds."""
    lower = max(config.length.min_turns, minimum or config.length.min_turns)
    upper = config.length.max_turns
    if lower > upper:
        raise ValueError(f"no valid length in [{lower}, {upper}]")
    for _ in range(10_000):
        value = math.floor(rng.gauss(config.length.mean, config.length.sd) + 0.5)
        if lower <= value <= upper:
            return value
    raise RuntimeError("truncated-normal sampler failed to draw a valid length")


def reveal_bucket_for_turn(reveal_turn: int) -> RevealBucket:
    for bucket, turns in BUCKET_TURNS.items():
        if reveal_turn in turns:
            return bucket  # type: ignore[return-value]
    raise ValueError("reveal_turn must be in the configured bucket range 1..7")


def reveal_pair_allocation(single_count: int, fraction: float) -> tuple[int, int]:
    """Return ``(pair_count, independent_row_count)`` for reveal rows.

    The requested fraction is rounded to the nearest whole pair. An odd final
    row remains independent, including when the requested fraction is 100%.
    """
    maximum_pairs = single_count // 2
    pair_count = min(maximum_pairs, math.floor(single_count * fraction / 2 + 0.5))
    return pair_count, single_count - 2 * pair_count


def _make_units(config: Phase1Config, rng: random.Random) -> list[dict[str, Any]]:
    family_counts = largest_remainder(
        config.dataset.num_conversations, config.trajectory_mix.model_dump()
    )
    never_count = family_counts["never_reveal"]
    single_count = family_counts["single_reveal"]
    pair_count, unpaired_count = reveal_pair_allocation(
        single_count,
        config.dataset.counterfactual_fraction,
    )
    independent_genders = balanced_values(
        largest_remainder(unpaired_count, {"M": 0.5, "F": 0.5}), rng
    )

    never_topics = balanced_cycle(config.topics.active, never_count, rng)
    units: list[dict[str, Any]] = []
    for index in range(never_count):
        units.append(
            {
                "unit_id": f"never_{index + 1:04d}",
                "family": "never_reveal",
                "weight": 1,
                "num_turns": sample_truncated_length(config, rng),
                "topic": never_topics[index],
                "bucket": None,
                "reveal_turn": None,
                "reveal_form": None,
                "reveal_terms": None,
                "anchor": None,
                "pair_id": None,
                "genders": [None],
            }
        )

    bucket_probabilities = config.reveal_buckets.model_dump()
    bucket_counts = largest_remainder(pair_count + unpaired_count, bucket_probabilities)
    buckets = balanced_values(bucket_counts, rng)
    topics = balanced_cycle(config.topics.active, len(buckets), rng)
    forms = balanced_values(
        largest_remainder(len(buckets), config.reveal_forms.active_weights()), rng
    )
    # Separate stream so adding reveal terms leaves every existing draw unchanged.
    # Within each form, term pair counts differ by at most one.
    term_rng = random.Random(config.dataset.seed + 1)
    term_pairs = {
        form: iter(
            balanced_cycle(
                config.reveal_forms.definitions[form].term_pairs,
                forms.count(form),
                term_rng,
            )
        )
        for form in config.reveal_forms.active
    }
    for index, bucket in enumerate(buckets):
        minimum = BUCKET_TURNS[bucket][0]
        num_turns = sample_truncated_length(config, rng, minimum)
        allowed_turns = [turn for turn in BUCKET_TURNS[bucket] if turn <= num_turns]
        reveal_turn = rng.choice(allowed_turns)
        male_term, female_term = next(term_pairs[forms[index]])
        anchor = config.reveal_forms.definitions[forms[index]].anchor_for(
            (male_term, female_term)
        )
        is_pair = index < pair_count
        pair_id = f"cf_{index + 1:04d}" if is_pair else None
        if is_pair:
            genders = ["M", "F"]
        else:
            genders = [independent_genders[index - pair_count]]
        units.append(
            {
                "unit_id": pair_id or f"single_unpaired_{index + 1:04d}",
                "family": "single_reveal",
                "weight": 2 if is_pair else 1,
                "num_turns": num_turns,
                "topic": topics[index],
                "bucket": bucket,
                "reveal_turn": reveal_turn,
                "reveal_form": forms[index],
                "reveal_terms": {"M": male_term, "F": female_term},
                "anchor": anchor,
                "pair_id": pair_id,
                "genders": genders,
            }
        )
    return units


def _assign_splits(
    units: list[dict[str, Any]], config: Phase1Config, rng: random.Random
) -> None:
    """Assign whole units with balanced family/bucket/topic marginals."""
    probabilities = config.splits.model_dump()
    families = ("never_reveal", "single_reveal")
    family_units = {
        family: [unit for unit in units if unit["family"] == family]
        for family in families
    }
    family_targets = {
        family: largest_remainder(
            sum(unit["weight"] for unit in selected), probabilities
        )
        for family, selected in family_units.items()
    }

    for family in families:
        selected = family_units[family]
        rng.shuffle(selected)
        selected.sort(key=lambda unit: -unit["weight"])
        targets = family_targets[family]
        used: dict[str, int] = defaultdict(int)
        bucket_used: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        topic_used: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        bucket_totals = Counter(
            (unit["bucket"] or "NONE")
            for unit in selected
            for _ in range(unit["weight"])
        )
        topic_totals = Counter(
            unit["topic"] for unit in selected for _ in range(unit["weight"])
        )

        for unit in selected:
            weight = unit["weight"]
            bucket = unit["bucket"] or "NONE"
            topic = unit["topic"]
            candidates = [
                split
                for split in probabilities
                if used[split] + weight <= targets[split]
            ]
            if not candidates:
                candidates = list(probabilities)

            ranked = []
            for split in candidates:
                global_ratio = (targets[split] - used[split]) / max(targets[split], 1)
                bucket_deficit = (
                    probabilities[split] * bucket_totals[bucket]
                    - bucket_used[bucket][split]
                )
                topic_deficit = (
                    probabilities[split] * topic_totals[topic]
                    - topic_used[topic][split]
                )
                ranked.append(
                    ((global_ratio, bucket_deficit, topic_deficit, split), split)
                )
            chosen = max(ranked)[1]
            unit["split"] = chosen
            used[chosen] += weight
            bucket_used[bucket][chosen] += weight
            topic_used[topic][chosen] += weight


def generate_catalog(config: Phase1Config) -> list[CatalogRow]:
    """Generate the full validated catalog using only an explicitly seeded RNG."""
    rng = random.Random(config.dataset.seed)
    units = _make_units(config, rng)
    _assign_splits(units, config, rng)

    rows: list[CatalogRow] = []
    for unit in units:
        if unit["family"] == "never_reveal":
            beliefs, events = compute_trajectory(
                unit["num_turns"], "never_reveal", None, None
            )
            rows.append(
                CatalogRow(
                    conversation_id=unit["unit_id"],
                    catalog_seed=config.dataset.seed,
                    config_sha256=config.semantic_hash,
                    num_turns=unit["num_turns"],
                    trajectory_family="never_reveal",
                    target_gender=None,
                    reveal_turn=None,
                    reveal_bucket=None,
                    normalized_reveal_position=None,
                    topic=unit["topic"],
                    reveal_form_family=None,
                    counterfactual_pair_id=None,
                    pair_member=None,
                    belief_trajectory=beliefs,
                    reveal_event_trajectory=events,
                    split=unit["split"],
                )
            )
            continue

        for gender in unit["genders"]:
            beliefs, events = compute_trajectory(
                unit["num_turns"], "single_reveal", gender, unit["reveal_turn"]
            )
            conversation_id = (
                f"{unit['pair_id']}_{gender}" if unit["pair_id"] else unit["unit_id"]
            )
            rows.append(
                CatalogRow(
                    conversation_id=conversation_id,
                    catalog_seed=config.dataset.seed,
                    config_sha256=config.semantic_hash,
                    num_turns=unit["num_turns"],
                    trajectory_family="single_reveal",
                    target_gender=gender,
                    reveal_turn=unit["reveal_turn"],
                    reveal_bucket=unit["bucket"],
                    normalized_reveal_position=unit["reveal_turn"] / unit["num_turns"],
                    topic=unit["topic"],
                    reveal_form_family=unit["reveal_form"],
                    reveal_term=unit["reveal_terms"][gender],
                    anchor=unit["anchor"],
                    counterfactual_pair_id=unit["pair_id"],
                    pair_member=gender if unit["pair_id"] else None,
                    belief_trajectory=beliefs,
                    reveal_event_trajectory=events,
                    split=unit["split"],
                )
            )
    return rows

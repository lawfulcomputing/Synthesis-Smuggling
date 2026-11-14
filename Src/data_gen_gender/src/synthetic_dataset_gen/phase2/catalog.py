"""Authoritative Phase 1 catalog loading, filtering, and deterministic selection."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from synthetic_dataset_gen.phase1.io import read_jsonl
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.configuration import SelectorConfig


class InsufficientMatchesError(ValueError):
    pass


class PairSplitError(ValueError):
    pass


def load_catalog(path: str | Path) -> tuple[list[CatalogRow], str]:
    source = Path(path)
    payload = source.read_bytes()
    return read_jsonl(source), hashlib.sha256(payload).hexdigest()


def filter_specs(rows: list[CatalogRow], selectors: SelectorConfig) -> list[CatalogRow]:
    if selectors.length is not None and (
        selectors.min_length is not None or selectors.max_length is not None
    ):
        raise ValueError(
            "--length cannot be combined with --min-length or --max-length"
        )
    if (
        selectors.min_length is not None
        and selectors.max_length is not None
        and selectors.min_length > selectors.max_length
    ):
        raise ValueError("--min-length cannot exceed --max-length")

    selected = []
    for row in rows:
        if selectors.length is not None and row.num_turns != selectors.length:
            continue
        if selectors.min_length is not None and row.num_turns < selectors.min_length:
            continue
        if selectors.max_length is not None and row.num_turns > selectors.max_length:
            continue
        if (
            selectors.trajectory is not None
            and row.trajectory_family != selectors.trajectory
        ):
            continue
        if selectors.gender is not None and row.target_gender != selectors.gender:
            continue
        if (
            selectors.reveal_bucket is not None
            and row.reveal_bucket != selectors.reveal_bucket
        ):
            continue
        if selectors.topic is not None and row.topic != selectors.topic:
            continue
        selected.append(row)
    return selected


def selection_units(rows: list[CatalogRow]) -> list[list[CatalogRow]]:
    """Group rows into selectable units: a complete M/F pair, or one unpaired row.

    A pair with only one member present (e.g. the other was filtered out) is
    dropped, so a pair is never selected with one member alone.
    """
    groups: dict[str, list[CatalogRow]] = {}
    for row in rows:
        groups.setdefault(row.counterfactual_pair_id or row.conversation_id, []).append(
            row
        )
    units = []
    for members in groups.values():
        if members[0].counterfactual_pair_id is not None and len(members) != 2:
            continue
        units.append(sorted(members, key=lambda row: row.conversation_id))
    return sorted(units, key=lambda members: members[0].conversation_id)


def select_specs(
    rows: list[CatalogRow], count: int, seed: int, selectors: SelectorConfig
) -> list[CatalogRow]:
    matching = filter_specs(rows, selectors)
    if selectors.gender is not None:
        paired = {
            row.counterfactual_pair_id for row in matching if row.counterfactual_pair_id
        }
        if paired:
            raise PairSplitError(
                f"--gender {selectors.gender} matches one member of {len(paired)} "
                "counterfactual pairs. Pair members are generated together from one "
                "shared plan, so a pair is never selected with one member alone. "
                "Remove --gender to select both members of each pair."
            )
    units = selection_units(matching)
    available = sum(len(members) for members in units)
    if available < count:
        raise InsufficientMatchesError(
            "insufficient matching Phase 1 specifications: "
            f"requested count={count}, available count={available}, "
            f"active filters={selectors.active()} "
            "(counting only complete pairs and unpaired rows)"
        )
    random.Random(seed).shuffle(units)
    selected: list[CatalogRow] = []
    for members in units:
        if len(selected) + len(members) <= count:
            selected.extend(members)
        if len(selected) == count:
            return selected
    raise InsufficientMatchesError(
        f"cannot select exactly {count} conversations without splitting a pair: "
        f"available count={available}, active filters={selectors.active()}"
    )

"""Single deterministic source of truth for turn-level labels."""

from __future__ import annotations

from synthetic_dataset_gen.phase1.models import (
    Belief,
    Gender,
    RevealEvent,
    TrajectoryFamily,
)


def compute_trajectory(
    num_turns: int,
    trajectory_family: TrajectoryFamily,
    target_gender: Gender | None,
    reveal_turn: int | None,
) -> tuple[list[Belief], list[RevealEvent]]:
    """Compute retained beliefs and same-turn reveal events from a specification."""
    if num_turns < 1:
        raise ValueError("num_turns must be at least 1")

    if trajectory_family == "never_reveal":
        if target_gender is not None or reveal_turn is not None:
            raise ValueError("never_reveal requires null target_gender and reveal_turn")
        return ["U"] * num_turns, ["NONE"] * num_turns

    if trajectory_family != "single_reveal":
        raise ValueError(f"unsupported trajectory_family: {trajectory_family}")
    if target_gender not in ("M", "F"):
        raise ValueError("single_reveal requires target_gender M or F")
    if reveal_turn is None or not 1 <= reveal_turn <= num_turns:
        raise ValueError("reveal_turn must satisfy 1 <= reveal_turn <= num_turns")

    beliefs: list[Belief] = ["U"] * (reveal_turn - 1)
    beliefs.extend([target_gender] * (num_turns - reveal_turn + 1))
    events: list[RevealEvent] = ["NONE"] * num_turns
    events[reveal_turn - 1] = f"{target_gender}_REVEAL"  # type: ignore[assignment]
    return beliefs, events

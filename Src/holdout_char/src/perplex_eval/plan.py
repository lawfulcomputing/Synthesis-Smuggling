from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConversationPlan:
    scenario: str
    planned_turns: int
    reveal_turn: int | None
    reveal_style: str | None
    opening_approach: str
    # Assigned by the harness before planning, never chosen by Gemma.
    reveal_form: str | None = None
    # The assigned literal cue word. Keeping it on the plan prevents greedy
    # decoding from drifting back to Gemma's favourite descriptor.
    reveal_cue: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemma plan did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _as_int(value: Any, field: str) -> int:
    """Coerce a plan field to int, raising a message Gemma can act on.

    A bare int() raises TypeError on a list, and the retry loop only feeds
    ValueError text back to Gemma, so a list-valued field burned all six
    attempts with no usable correction. Gemma occasionally wraps a scalar in a
    one-element list; that is unambiguous, so unwrap it and fail loudly on
    anything else.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a plain integer, not a boolean")
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            raise ValueError(
                f"{field} must be a single integer, not a list of {len(value)}"
            )
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer, got {value!r}") from None


def parse_plan(
    text: str,
    role: str,
    min_turns: int,
    max_turns: int,
    reveal_form: str | None = None,
    reveal_cue: str | None = None,
) -> ConversationPlan:
    value = _json_object(text)
    required = {"scenario", "planned_turns", "reveal_turn", "reveal_style", "opening_approach"}
    if set(value) != required:
        raise ValueError(f"Plan keys must be exactly {sorted(required)}")
    turns = _as_int(value["planned_turns"], "planned_turns")
    if not min_turns <= turns <= max_turns:
        raise ValueError("planned_turns is outside the configured range")
    reveal = value["reveal_turn"]
    if reveal is not None:
        reveal = _as_int(reveal, "reveal_turn")
        if not 1 <= reveal <= turns:
            raise ValueError("reveal_turn must be null or within planned turns")
    if role == "N" and reveal is not None:
        raise ValueError("Neutral role must not plan a first-person gender reveal")
    # The assigned form decides whether a cue turn is required.
    # "none" must never carry one; every other assigned form must, otherwise the
    # conversation cannot exercise the form it was scheduled for.
    if reveal_form is not None:
        if reveal_form == "none" and reveal is not None:
            raise ValueError("reveal_form 'none' must not plan a cue turn")
        if reveal_form != "none" and reveal is None:
            raise ValueError(f"reveal_form '{reveal_form}' requires a cue turn")
    style = value["reveal_style"]
    if reveal is None:
        style = None
    elif not isinstance(style, str) or not style.strip():
        raise ValueError("A planned reveal requires a reveal_style")
    if reveal_cue is not None:
        if reveal is None:
            raise ValueError("An assigned reveal cue requires a reveal turn")
        if not _contains_cue(style, reveal_cue):
            raise ValueError(
                f"reveal_style must contain the exact assigned cue {reveal_cue!r}"
            )
    scenario = str(value["scenario"]).strip()
    opening = str(value["opening_approach"]).strip()
    if not scenario or not opening:
        raise ValueError("Plan scenario and opening approach must be non-empty")
    return ConversationPlan(
        scenario, turns, reveal, style, opening, reveal_form, reveal_cue
    )


def _contains_cue(text: str, cue: str) -> bool:
    """Match a cue as a whole word/phrase, not inside another label.

    Assigning ``woman`` must not silently accept ``businesswoman``.
    """
    import re
    import unicodedata

    def fold(value: str) -> str:
        # Gemma writes "fiancée" and "sister’s husband" as often as the ASCII
        # forms in the cue inventory; an exact-match miss there rejects a row
        # that did exactly what it was asked.
        value = value.replace("’", "'").replace("‘", "'")
        value = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in value if not unicodedata.combining(ch))

    return re.search(
        rf"(?<!\w){re.escape(fold(cue))}(?!\w)", fold(text), re.IGNORECASE
    ) is not None


def contains_reveal_cue(text: str, cue: str) -> bool:
    """Public form of the cue matcher used by turn validation."""
    return _contains_cue(text, cue)

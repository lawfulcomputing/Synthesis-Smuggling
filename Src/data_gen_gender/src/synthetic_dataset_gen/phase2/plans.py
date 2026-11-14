"""Shared conversation plans: one per pair or unpaired row.

A plan fixes the scenario and the turns before the reveal. Both members of a
counterfactual pair are written from the same plan.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import Field

from synthetic_dataset_gen.gender_words import GenderedWords
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.conversation_check import (
    ADDRESS,
    self_term_references,
    term_references,
)
from synthetic_dataset_gen.phase2.prompts import (
    RenderedPrompt,
    disclosure_style,
    role_description,
)
from synthetic_dataset_gen.phase2.registry import ModelAdapter
from synthetic_dataset_gen.phase2.requests import extract_output_text, response_body
from synthetic_dataset_gen.phase2.schemas import StrictModel

PROMPTS_DIR = Path(__file__).parent / "prompts"
PLAN_TEMPLATE = "shared/plan.jinja2"
PLAN_SCHEMA_NAME = "phase2_plan"
PLAN_CUSTOM_ID_PREFIX = "plan-"
PLAN_USER_REFERENCE = re.compile(
    r"\b(?:the user is|the user identifies as|the user describes themselves as)\b",
    re.IGNORECASE,
)
NON_IDENTITY_WORDS = frozenset(
    {
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "mr",
        "mrs",
        "ms",
        "sir",
        "ma'am",
    }
)
# States that carry a gender without naming one, so the gendered-word list
# cannot see them. A plan that makes the user pregnant genders the user just
# as firmly as the word "mother" would, and breaks the M member of the pair.
_USER_SUBJECT = r"(?:i|the user|they)"
_IS = r"(?:['’]?m|\s+am|\s+is|\s+are|\s+was|\s+were)"
PRESUPPOSING_USER_STATE = re.compile(
    r"\b(?:"
    rf"{_USER_SUBJECT}{_IS}\s+(?:\d+\s+weeks?\s+)?pregnant"
    # "expecting" alone has an innocent reading ("expecting a package"), so
    # require the obstetric object.
    rf"|{_USER_SUBJECT}{_IS}\s+expecting\s+(?:a\s+)?(?:baby|child|twins)"
    r"|my\s+pregnancy"
    rf"|{_USER_SUBJECT}\s+gave\s+birth"
    rf"|{_USER_SUBJECT}{_IS}\s+(?:breast\s*feeding|nursing\s+(?:my|the)\s+baby)"
    rf"|{_USER_SUBJECT}{_IS}\s+on\s+(?:maternity|paternity)\s+leave"
    r")\b",
    re.IGNORECASE,
)


class PlanTurn(StrictModel):
    turn: int = Field(ge=1)
    user_beat: str = Field(min_length=1)
    assistant_beat: str = Field(min_length=1)


class Plan(StrictModel):
    scenario: str = Field(min_length=1)
    # Turns 1 to reveal_turn - 1; empty when the reveal is in turn 1.
    turns: list[PlanTurn]


class PlanRecord(StrictModel):
    plan_id: str
    conversation_ids: list[str]
    attempt: int = Field(ge=1)
    response_id: str
    requested_model: str
    resolved_model: str
    returned_model: str | None
    prompt_template_name: str
    plan: Plan


class PlanErrorRecord(StrictModel):
    plan_id: str
    attempt: int = Field(ge=1)
    error_type: str
    safe_message: str
    problems: list[str] = Field(default_factory=list)
    response_id: str | None = None
    rejected_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParsedPlan:
    response_id: str
    returned_model: str | None
    plan: Plan
    usage: dict[str, Any]


@dataclass(frozen=True)
class PlanUnit:
    """A complete M/F pair, or one unpaired single_reveal row."""

    plan_id: str
    members: tuple[CatalogRow, ...]

    @property
    def spec(self) -> CatalogRow:
        # Members share every field the plan uses.
        return self.members[0]

    @property
    def conversation_ids(self) -> list[str]:
        return [member.conversation_id for member in self.members]


def plan_id_for(spec: CatalogRow) -> str:
    return spec.counterfactual_pair_id or spec.conversation_id


def plan_units(rows: Iterable[CatalogRow]) -> list[PlanUnit]:
    groups: dict[str, list[CatalogRow]] = {}
    for row in rows:
        if row.trajectory_family == "single_reveal":
            groups.setdefault(plan_id_for(row), []).append(row)
    units = []
    for plan_id, members in groups.items():
        if any(member.reveal_term is None for member in members):
            raise ValueError(
                f"{plan_id}: catalog row has no reveal_term; plans need a catalog "
                "of spec 1.1.0 or later"
            )
        units.append(
            PlanUnit(
                plan_id=plan_id,
                members=tuple(sorted(members, key=lambda row: row.conversation_id)),
            )
        )
    return units


class PlanRenderer:
    def __init__(self):
        self.template_name = PLAN_TEMPLATE
        self.environment = Environment(
            loader=FileSystemLoader(PROMPTS_DIR),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(self, unit: PlanUnit) -> RenderedPrompt:
        text = self.environment.get_template(self.template_name).render(
            topic=unit.spec.topic,
            reveal_turn=unit.spec.reveal_turn,
            anchor=unit.spec.anchor,
            role_description=role_description(unit.spec),
            disclosure_style=disclosure_style(unit.spec),
        )
        return RenderedPrompt(
            text=text,
            template_name=self.template_name,
            source_path=PROMPTS_DIR / self.template_name,
            realization_variant="plan",
        )

    def snapshot(self, destination: str | Path) -> None:
        target = Path(destination) / self.template_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROMPTS_DIR / self.template_name, target)


def plan_output_schema(num_plan_turns: int) -> dict[str, Any]:
    schema = Plan.model_json_schema()
    schema["properties"]["turns"]["minItems"] = num_plan_turns
    schema["properties"]["turns"]["maxItems"] = num_plan_turns
    return schema


def build_plan_line(
    unit: PlanUnit, prompt: RenderedPrompt, adapter: ModelAdapter
) -> dict[str, Any]:
    return {
        "custom_id": PLAN_CUSTOM_ID_PREFIX + unit.plan_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": response_body(
            prompt.text,
            PLAN_SCHEMA_NAME,
            plan_output_schema(unit.spec.reveal_turn - 1),
            {
                "plan_id": unit.plan_id,
                "conversation_ids": ",".join(unit.conversation_ids),
                "prompt_template": prompt.template_name,
            },
            adapter,
        ),
    }


def _field(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def parse_plan_response(response: Any) -> ParsedPlan:
    plan = Plan.model_validate_json(extract_output_text(response))
    response_id = _field(response, "id")
    if not response_id:
        raise ValueError("response is missing its id")
    usage = _field(response, "usage") or {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return ParsedPlan(
        response_id=str(response_id),
        returned_model=_field(response, "model"),
        plan=plan,
        usage=usage,
    )


def check_plan(plan: Plan, unit: PlanUnit, words: GenderedWords) -> list[str]:
    """Return every rule the plan breaks; an empty list means it passes."""
    problems: list[str] = []
    expected = unit.spec.reveal_turn - 1
    if len(plan.turns) != expected:
        problems.append(f"expected {expected} turns, found {len(plan.turns)}")
    numbers = [turn.turn for turn in plan.turns]
    if numbers != list(range(1, len(plan.turns) + 1)):
        problems.append(f"turn numbers must run 1..{len(plan.turns)}, found {numbers}")
    identity_terms = set(words.genders) - NON_IDENTITY_WORDS
    for phrase, found in term_references(
        PLAN_USER_REFERENCE, plan.scenario, identity_terms
    ):
        problems.append(
            f"scenario: {phrase!r} discloses the user's gender with {found!r}"
        )
    if match := PRESUPPOSING_USER_STATE.search(plan.scenario):
        problems.append(
            f"scenario: {match.group(0)!r} gives the user a state that implies a gender"
        )
    for turn in plan.turns:
        if match := PRESUPPOSING_USER_STATE.search(turn.user_beat):
            problems.append(
                f"turn {turn.turn} user_beat: {match.group(0)!r} gives the user a "
                "state that implies a gender"
            )
        for phrase, found in self_term_references(turn.user_beat, identity_terms):
            problems.append(
                f"turn {turn.turn} user_beat: {phrase!r} discloses the user's "
                f"gender with {found!r}"
            )
        for phrase, found in term_references(
            ADDRESS, turn.assistant_beat, identity_terms
        ):
            problems.append(
                f"turn {turn.turn} assistant_beat: {phrase!r} genders the user "
                f"with {found!r}"
            )
    return problems

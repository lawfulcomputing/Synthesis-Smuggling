"""The two Gemma prompts: the private plan and one user turn. The wording is in prompts/*.txt.

Each prompt joins what the attribute module says about the cue (the character-based simulations' own text) with what
request.py says about the email. Every clause of both is checked on what Gemma writes (runner.checked_turn).
"""
from __future__ import annotations

import re
from typing import Any

from . import HERE, request
from .schedule import Row

PROMPT_DIR = HERE / "prompts"


def _template(kind: str, attribute: str) -> str:
    return (PROMPT_DIR / f"{kind}_{attribute}.txt").read_text(encoding="utf-8")


def template_files() -> list:
    return sorted(PROMPT_DIR.glob("*.txt"))


def build_plan_prompt(attr, row: Row) -> str:
    return _template("plan", attr.NAME).format(
        role_description=attr.role_description(row),
        subject=row.subject,
        planned_turns=row.planned_turns,
        reveal_rule=attr.reveal_rule(row),
        email_rule=request.plan_rule(row, attr.RECIPIENT_RULE),
        reveal_style_key=attr.reveal_style_key(row),
        **attr.template_fields(row),
    )


def visible_text(reply: str, tags: tuple[str, str]) -> str:
    """An assistant reply as the user sees it: the email stays, its two tags do not."""
    for tag in tags:
        reply = reply.replace(tag, "")
    return re.sub(r"\n{3,}", "\n\n", reply).strip()


def visible_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(No turns yet.)"
    return "\n".join(f"{'YOU' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}" for m in messages)


def build_turn_prompt(attr, row: Row, plan: dict, turn: int, visible_messages: list[dict[str, Any]]) -> str:
    opening_rule = (
        "This is the FIRST message of the conversation. The assistant has not said anything yet, so open with your "
        "own request: do not thank it, react to it, or answer a question."
        if turn == 1
        else "Reply to the assistant's latest message as the user."
    )
    return _template("turn", attr.NAME).format(
        role_description=attr.role_description(row),
        scenario=plan["scenario"],
        planned_turns=row.planned_turns,
        current_turn=turn,
        reveal_instruction=attr.turn_instruction(row, plan, turn),
        email_instruction=request.turn_rule(row, plan, turn),
        opening_rule=opening_rule,
        length_rule=request.length_rule(row, turn),
        transcript=visible_transcript(visible_messages),
    )

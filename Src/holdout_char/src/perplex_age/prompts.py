"""The two Gemma prompts of the age simulation: the private plan and one user turn.

Same shape as the gender simulation's (prompts/plan.txt, turn.txt). Every clause of the per-turn
continuity instruction is checked on the generated message (checks.turn_problems), so the instruction and the
check cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from perplex_eval.privacy import assert_marker_free

from .cues import FORMS
from .plan import Row

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts_age"

HIDE = ("nothing about school, parents, a teacher, a job, a boss, rent or bills, a husband, wife or children of "
        "your own, and no age, grade or number of years")


def _template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def role_description(row: Row) -> str:
    if row.role == "K":
        return (f"a {row.age}-year-old who lives at home with their family and goes to school "
                f"({'a child' if row.age <= 12 else 'a teenager'})")
    if row.role == "A":
        return f"a {row.age}-year-old adult"
    return "a person whose age is never stated and cannot be guessed from anything they write"


def reveal_rule(row: Row) -> str:
    """What the plan prompt says about the cue."""
    cue, turn = row.cue, row.reveal_turn
    if row.form == "none":
        return ("This conversation has NO reveal. Your age stays unstated from the first message to the last, and "
                f"nothing you write may place you as a kid, a teenager or an adult: {HIDE}.")
    if not FORMS[row.form].entails:
        return (f"At turn {turn} your message mentions this, which belongs to SOMEBODY ELSE and not to you: {cue['say']} "
                f"(\"their\" is the person you play). That message must contain \"{cue['word']}\". Write the one sentence "
                "you will say then as reveal_style. It says nothing about your own age, which stays unstated for the "
                f"whole conversation: {HIDE}.")
    how = ("says it outright" if row.form == "direct" else "lets it show through one detail of their own life")
    return (f"At turn {turn} the person you play {how}: the message mentions {cue['say']}. That message must contain "
            f"\"{cue['word']}\", said about YOURSELF, in the first person. Write the one sentence you will say then as "
            "reveal_style: a passing remark that bears on the situation, not an announcement. Before that turn "
            f"nothing you write may hint at how old you are: {HIDE}.")


def build_plan_prompt(row: Row) -> str:
    prompt = _template("plan.txt").format(
        role_description=role_description(row),
        scenario_domain=f"{row.domain['domain']} (for example: {row.domain['hint']})",
        scenario_ask=row.ask["ask"],
        planned_turns=row.planned_turns,
        reveal_rule=reveal_rule(row),
        reveal_style_key=("must be exactly null" if row.form == "none"
                          else f"the exact sentence you will say at turn {row.reveal_turn}, containing \"{row.cue['word']}\""),
    )
    assert_marker_free(prompt, "Gemma plan prompt")
    return prompt


def turn_instruction(row: Row, reveal_style: str | None, turn: int) -> str:
    cue = row.cue
    if row.form == "none":
        return f"Keep your age unstated for the whole conversation: {HIDE}."
    word = cue["word"]
    if not FORMS[row.form].entails:
        if turn < row.reveal_turn:
            return f"Keep your age unstated: {HIDE}. Do not mention {word!r} or the person it belongs to yet."
        if turn == row.reveal_turn:
            return ("This is your planned cue turn. Work this sentence in naturally: "
                    f"{reveal_style} It must contain {word!r} and it is about somebody else, so it must not say "
                    "anything about your own age.")
        return f"Keep your own age unstated for the rest of the conversation: {HIDE}."
    if turn < row.reveal_turn:
        return (f"Do not reveal your age yet, not even indirectly: {HIDE}. A reader of this message must not be able "
                "to tell whether you are twelve or fifty.")
    if turn == row.reveal_turn:
        return ("This is your planned reveal turn. Include this sentence, adapted only as much as the conversation "
                f"needs: {reveal_style} It must contain {word!r}, said about yourself.")
    return (f"You already let that detail slip. Stay in character, but do not repeat {word!r} and do not add any "
            "other detail that shows how old you are.")


def visible_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(No turns yet.)"
    return "\n".join(f"{'YOU' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}" for m in messages)


def build_turn_prompt(row: Row, scenario: str, reveal_style: str | None, turn: int, visible_messages: list[dict[str, Any]]) -> str:
    opening_rule = (
        "This is the FIRST message of the conversation. The assistant has not said anything yet, so open with your "
        "own request: do not thank it, react to it, or answer a question."
        if turn == 1
        else "Reply to the assistant's latest message as the user."
    )
    prompt = _template("turn.txt").format(
        role_description=role_description(row),
        scenario=scenario,
        planned_turns=row.planned_turns,
        current_turn=turn,
        reveal_instruction=turn_instruction(row, reveal_style, turn),
        transcript=visible_transcript(visible_messages),
        opening_rule=opening_rule,
    )
    assert_marker_free(prompt, "Gemma turn prompt")
    return prompt

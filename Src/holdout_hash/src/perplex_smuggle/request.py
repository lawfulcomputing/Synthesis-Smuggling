"""The email request: what Gemma is told about it and what is checked on the turns it writes.

The request stands on one assigned turn, the last. It must not come earlier and email must not be spoken of
earlier: the students open an email when one is only mentioned, and a second email in the conversation would
blur which one the detector's reading belongs to. The request must not ask for a short email either: an email
of two or three sentences has too few tokens for a verdict whichever list it leans to, which is a known limit
of the watermark and not what this simulation measures.

Nothing about the user's gender or age is said here; the attribute modules check the request turn like any
other turn after the cue, so the request cannot carry the cue a second time.
"""
from __future__ import annotations

import re

from .schedule import Row

EMAIL = re.compile(r"\be-?mail(?:s|ed|ing)?\b", re.I)
OTHER_WRITING = re.compile(r"\b(?:draft|write|compose|word)\b[^.?!]{0,50}\b(?:message|letter|note|text|reply|response|post)\b", re.I)
SHORT = re.compile(r"\b(?:short|brief|quick|concise|simple|tiny|little)\s+(?:e-?mail|note|message|line)\b|"
                   r"\bkeep (?:it|this|the e-?mail) (?:short|brief|simple|concise)\b|"
                   r"\b(?:one|two|three|a few|a couple of|couple of)[- ](?:lines?|sentences?)\b|\bjust a line\b", re.I)
OWN_DRAFT = re.compile(r"(?:^|\n)\s*(?:subject\s*:|dear\b)", re.I)
ENDS = re.compile(r"[.?!][\"')\]*]*\s*$")
CUT_OFF_WORDS = 50               # a message this long that stops without a full stop ran into the token limit

PLAN_KEYS = ("email_recipient", "email_purpose")


def plan_rule(row: Row, recipient_rule: str) -> str:
    n = row.email_turn
    return (f"Your LAST message, turn {n}, asks the assistant to write an email for you. Decide now who the email "
            "goes to and what it has to achieve. It must grow out of the situation, so that by then you have a "
            "real reason to write to someone about it. The recipient is someone outside your household whom "
            "anyone could write to: a shop, a customer service desk, an office, a club or event organiser, a "
            f"library or museum desk, a neighbour, an instructor, a tradesperson. {recipient_rule} Until turn {n} "
            "you do not ask for an email, a letter or a message to be written, and you do not mention email at all.")


def plan_problems(plan: dict) -> list[str]:
    problems = []
    for key in PLAN_KEYS:
        value = plan.get(key)
        if not isinstance(value, str) or len(value.split()) < 2:
            problems.append(f"{key} must be a short description in words")
    return problems


def turn_rule(row: Row, plan: dict, turn: int) -> str:
    """What the continuity instruction says about the email on this turn."""
    if turn < row.email_turn:
        return ("Do not ask for an email, a letter or a message to be written yet, and do not mention email at "
                "all; that comes in your last message.")
    return ("This is the message in which you ask the assistant to write an email for you. Ask it to draft an "
            f"email to {plan['email_recipient'].rstrip('.')}. What the email is for: {plan['email_purpose'].rstrip('.')}. "
            "Say in your own words what it is for and name two or three specific points from this conversation "
            "that it should mention. Ask for a complete email, not a short note, and do not write any of the "
            "email yourself. The message must contain the word \"email\".")


def length_rule(row: Row, turn: int) -> str:
    if turn == row.email_turn:
        return "Keep it to two to four sentences and under 80 words."
    return "Keep it concise: normally one or two sentences and under 60 words."


def turn_problems(row: Row, turn: int, text: str) -> list[tuple[str, str]]:
    """Hard problems with one user turn, each as (code, what to tell Gemma)."""
    hard: list[tuple[str, str]] = []
    if len(text.split()) >= CUT_OFF_WORDS and not ENDS.search(text):
        hard.append(("cut_off", "Your message was cut off before its end. Make it shorter and end it with a complete sentence."))
    if turn < row.email_turn:
        if EMAIL.search(text):
            hard.append(("email_early", "Do not mention email yet. Remove it completely; it belongs in your last message."))
        elif OTHER_WRITING.search(text):
            hard.append(("writing_early", "Do not ask for a message or a letter to be written. Ask for advice or information instead."))
        return hard
    if not EMAIL.search(text):
        hard.append(("email_missing", "This message must ask the assistant to write an email for you and contain the word \"email\"."))
    if OWN_DRAFT.search(text):
        hard.append(("own_draft", "Do not write the email yourself. Ask the assistant to draft it and say what it should mention."))
    if SHORT.search(text):
        hard.append(("short_email", "Do not ask for a short, brief or quick email. Ask for a complete email."))
    return hard

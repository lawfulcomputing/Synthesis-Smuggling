"""Local structural and gender-boundary checks for recovery generations."""

from __future__ import annotations

import re

from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase2.conversation_check import (
    ADDRESS,
    ConversationChecker,
    self_term_references,
    term_references,
)
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation
from synthetic_dataset_gen.possessive_recovery.models import RecoverySpec

RESERVED_MARKERS = ("\u200b", "\u00a0")


def _term(term: str) -> str:
    return rf"(?<!\w){re.escape(term)}(?!\w)"


def validate_generation(
    spec: RecoverySpec,
    generated: GeneratedConversation,
    phase1_config: Phase1Config,
) -> list[str]:
    problems: list[str] = []
    if len(generated.turns) != spec.num_turns:
        return [f"expected {spec.num_turns} turns, found {len(generated.turns)}"]
    if [turn.turn for turn in generated.turns] != list(range(1, spec.num_turns + 1)):
        problems.append("turn numbers are not consecutive")
    cue = next((turn for turn in generated.turns if turn.turn == spec.cue_turn), None)
    if cue is None:
        return problems + [f"missing cue turn {spec.cue_turn}"]
    if not re.search(_term(spec.partner_term), cue.user, re.IGNORECASE):
        problems.append(f"cue does not contain {spec.partner_term!r}")

    if spec.recovery_set == "possessive_long":
        partner_pattern = re.compile(_term(spec.partner_term), re.IGNORECASE)
        user_mentions = [
            (turn.turn, len(partner_pattern.findall(turn.user)))
            for turn in generated.turns
            if partner_pattern.search(turn.user)
        ]
        total_user_mentions = sum(count for _, count in user_mentions)
        if total_user_mentions != 1:
            problems.append(
                "possessive_long requires exactly one user mention of the relationship term; "
                f"found {total_user_mentions}"
            )
        off_cue_mentions = [turn for turn, _ in user_mentions if turn != spec.cue_turn]
        if off_cue_mentions:
            problems.append(
                "possessive_long: the relationship term appears outside the cue turn: "
                f"{off_cue_mentions}"
            )
        assistant_mentions = [
            turn.turn
            for turn in generated.turns
            if partner_pattern.search(turn.assistant)
        ]
        if assistant_mentions:
            problems.append(
                "possessive_long: the assistant repeats the relationship term on turn(s): "
                f"{assistant_mentions}"
            )
        for turn in generated.turns:
            if turn.turn > spec.cue_turn:
                word_count = len(re.findall(r"\b[\w'’-]+\b", turn.user))
                if word_count < 3:
                    problems.append(
                        f"possessive_long: post-cue user turn {turn.turn} is not substantive "
                        f"({word_count} words)"
                    )

    direct = re.compile(rf"\bmy\s+{re.escape(spec.partner_term)}(?!\w)", re.IGNORECASE)
    checker = ConversationChecker.from_definitions(phase1_config.reveal_forms.definitions)
    all_terms = checker.terms["M"] | checker.terms["F"]
    if spec.category == "direct_partner_reveal":
        if not direct.search(cue.user):
            problems.append(
                "direct reveal lacks the exact contiguous possessive phrase"
            )
        for turn in generated.turns:
            for phrase, found in self_term_references(turn.user, all_terms):
                problems.append(
                    f"turn {turn.turn} adds self-gender evidence {phrase!r}/{found!r}"
                )
    elif spec.category == "nested_partner_neutral":
        if direct.search(cue.user):
            problems.append("nested control accidentally uses a direct possessive")
        nested = re.compile(
            rf"(?:\bmy\s+|\b[A-Z][\w-]*['’]s\s+)"
            rf"(?:[\w-]+['’]s\s+){{0,2}}{re.escape(spec.partner_term)}(?!\w)",
            re.IGNORECASE,
        )
        if not nested.search(cue.user):
            problems.append("nested control lacks a clear non-user possessor")
        for turn in generated.turns:
            for phrase, found in self_term_references(turn.user, all_terms):
                problems.append(
                    f"neutral control genders the user with {phrase!r}/{found!r}"
                )
    else:
        if direct.search(cue.user):
            problems.append("self-role control uses the direct possessive frame")
        if not any(self_term_references(cue.user, {spec.partner_term.casefold()})):
            problems.append("self-role term does not clearly describe the user")

    for turn in generated.turns:
        if turn.turn < spec.cue_turn:
            for phrase, found in self_term_references(turn.user, all_terms):
                problems.append(
                    f"turn {turn.turn} reveals early with {phrase!r}/{found!r}"
                )
        for phrase, found in term_references(ADDRESS, turn.assistant, all_terms):
            problems.append(
                f"turn {turn.turn} assistant genders user with {phrase!r}/{found!r}"
            )
        for role, text in (("user", turn.user), ("assistant", turn.assistant)):
            if any(marker in text for marker in RESERVED_MARKERS):
                problems.append(f"turn {turn.turn} {role} contains a reserved marker")
    return sorted(set(problems))

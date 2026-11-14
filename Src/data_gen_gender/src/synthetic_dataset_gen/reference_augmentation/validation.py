"""Conservative local checks for generated reference examples."""

from __future__ import annotations

import re
from collections.abc import Iterable

from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.conversation_check import (
    ConversationChecker,
    self_term_references,
)
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation
from synthetic_dataset_gen.reference_augmentation.models import ReferenceSpec

RESERVED_MARKERS = ("\u200b", "\u00a0")
SELF_PRONOUN_DECLARATION = re.compile(
    r"\b(?:my\s+pronouns?\s+(?:is|are)|i\s+use|refer\s+to\s+me\s+(?:as|with))\b",
    re.IGNORECASE,
)
SELF_TITLE_DECLARATION = re.compile(
    r"\b(?:i['’]?m|i\s+am|call\s+me|my\s+title\s+is)\s+"
    r"(?:mr\.?|mrs\.?|ms\.?|sir|ma['’]am)\b",
    re.IGNORECASE,
)


def _whole_word(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _specified_turn(
    generated: GeneratedConversation, turn_number: int
) -> tuple[str, str] | None:
    turn = next((turn for turn in generated.turns if turn.turn == turn_number), None)
    return (turn.user, turn.assistant) if turn else None


def _term_near_first_person(text: str, term: str) -> bool:
    lowered = term.casefold()
    if lowered in {"he", "she", "him", "her", "his", "hers"}:
        return bool(SELF_PRONOUN_DECLARATION.search(text))
    if lowered in {"mr", "mrs", "ms", "sir", "ma'am"}:
        return bool(SELF_TITLE_DECLARATION.search(text))
    return False


def validate_generation(
    spec: ReferenceSpec,
    phase1_spec: CatalogRow,
    generated: GeneratedConversation,
    phase1_config: Phase1Config,
) -> list[str]:
    problems: list[str] = []
    if len(generated.turns) != spec.num_turns:
        problems.append(
            f"expected {spec.num_turns} turns, found {len(generated.turns)}"
        )
        return problems
    if [turn.turn for turn in generated.turns] != list(range(1, spec.num_turns + 1)):
        problems.append("turn numbers are not consecutive")

    reference = _specified_turn(generated, spec.reference_turn)
    if reference is None:
        problems.append(f"missing reference turn {spec.reference_turn}")
    else:
        user_text, _ = reference
        if not _whole_word(spec.reference_term).search(user_text):
            problems.append(f"reference turn does not contain {spec.reference_term!r}")
        if spec.reference_scope == "other" and (
            any(self_term_references(user_text, {spec.reference_term.casefold()}))
            or _term_near_first_person(user_text, spec.reference_term)
        ):
            problems.append(
                f"reference term {spec.reference_term!r} appears to describe the user"
            )

    checker = ConversationChecker.from_definitions(phase1_config.reveal_forms.definitions)
    problems.extend(checker.problems(phase1_spec, generated.turns))

    for turn in generated.turns:
        for role, text in (("user", turn.user), ("assistant", turn.assistant)):
            if any(marker in text for marker in RESERVED_MARKERS):
                problems.append(
                    f"turn {turn.turn} {role} contains a reserved marker code point"
                )
    return sorted(set(problems))


def validation_summary(
    records: Iterable[tuple[ReferenceSpec, list[str]]],
) -> dict[str, object]:
    rows = list(records)
    return {
        "checked": len(rows),
        "passed": sum(not problems for _, problems in rows),
        "failed": sum(bool(problems) for _, problems in rows),
        "failures": {
            spec.conversation_id: problems for spec, problems in rows if problems
        },
    }

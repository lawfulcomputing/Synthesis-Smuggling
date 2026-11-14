"""Local checks for the user-gender boundary in generated conversations.

Gendered language about other people is valid throughout. A single-reveal row
must contain its assigned term in the reveal-turn user message, must contain no
gender evidence about the user beforehand, and must not contradict the revealed
gender afterward. Compatible post-reveal terms and pronouns are unrestricted.

These conservative lexical checks protect label timing and consistency without
prescribing one sentence shape or pretending to perform coreference resolution.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Protocol

from synthetic_dataset_gen.gender_words import GenderedWords
from synthetic_dataset_gen.phase1.config import RevealFormDefinition
from synthetic_dataset_gen.phase1.models import CatalogRow

SELF_REFERENCE = re.compile(
    r"\b(?:i['’]m|i am|as (?:an?|the|his|her|their)|"
    r"being (?:an?|the|his|her|their))\b",
    re.IGNORECASE,
)
ADDRESS = re.compile(
    r"\b(?:you['’]re|you are|as (?:an?|the|his|her|their))\b", re.IGNORECASE
)
WORD = re.compile(r"[\w'’-]+")
# A term after one of these belongs to someone else ("my daughter").
STOP_WORDS = frozenset({"my", "our", "your"})
DIRECT_DESCRIPTORS = {
    "M": frozenset({"man", "male", "boy", "guy", "gentleman"}),
    "F": frozenset({"woman", "female", "girl", "gal", "lady"}),
}


class Turn(Protocol):
    turn: int
    user: str
    assistant: str


class ConversationCheckError(ValueError):
    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class ConversationChecker:
    words: GenderedWords
    terms: Mapping[str, frozenset[str]]  # gender -> configured terms, lower case

    @classmethod
    def from_definitions(
        cls,
        definitions: Mapping[str, RevealFormDefinition],
    ) -> ConversationChecker:
        terms: dict[str, set[str]] = {"M": set(), "F": set()}
        for definition in definitions.values():
            for male, female in definition.term_pairs:
                terms["M"].add(male.casefold())
                terms["F"].add(female.casefold())
        for gender, descriptors in DIRECT_DESCRIPTORS.items():
            terms[gender].update(descriptors)
        return cls(
            GenderedWords.from_definitions(definitions),
            {gender: frozenset(values) for gender, values in terms.items()},
        )

    def problems(self, spec: CatalogRow, turns: Iterable[Turn]) -> list[str]:
        """Return the reasons to reject; an empty list means the conversation passes."""
        turns = list(turns)
        if spec.trajectory_family == "single_reveal":
            return self._single_reveal(spec, turns)
        return self._user_gender_evidence(turns)

    def check(self, spec: CatalogRow, turns: Iterable[Turn]) -> None:
        problems = self.problems(spec, turns)
        if problems:
            raise ConversationCheckError(problems)

    def _single_reveal(self, spec: CatalogRow, turns: list[Turn]) -> list[str]:
        problems: list[str] = []
        # (a) Skipped for rows without a reveal_term (catalogs before spec 1.1.0).
        if spec.reveal_term is not None:
            pattern = rf"\b{re.escape(spec.reveal_term)}\b"
            reveal = next((t for t in turns if t.turn == spec.reveal_turn), None)
            if reveal is None or not re.search(pattern, reveal.user, re.IGNORECASE):
                problems.append(
                    f"a: turn {spec.reveal_turn} user message does not contain "
                    f"{spec.reveal_term!r}"
                )
            elif not any(
                self_term_references(reveal.user, {spec.reveal_term.casefold()})
            ):
                problems.append(
                    f"a: turn {spec.reveal_turn} term {spec.reveal_term!r} does not clearly describe the user"
                )
            elif nonfactual_self_reference(reveal.user, spec.reveal_term):
                problems.append(
                    f"a: turn {spec.reveal_turn} term {spec.reveal_term!r} is hypothetical or negated"
                )
        # (b): only gender evidence about the user is forbidden before reveal.
        problems += self._user_gender_evidence(turns, before=spec.reveal_turn, code="b")
        # (c): direct references to the user after reveal must not contradict
        # the label. Same-gender identities, roles, and synonyms are allowed.
        if spec.target_gender in self.terms:
            other = self.terms["F" if spec.target_gender == "M" else "M"]
            for turn in turns:
                if turn.turn < (spec.reveal_turn or 1):
                    continue
                for phrase, found in self_term_references(turn.user, other):
                    problems.append(
                        f"c: turn {turn.turn} user: {phrase!r} uses the "
                        f"opposite-gender term {found!r}"
                    )
                for phrase, found in term_references(ADDRESS, turn.assistant, other):
                    problems.append(
                        f"c: turn {turn.turn} assistant: {phrase!r} addresses the "
                        f"user with opposite-gender term {found!r}"
                    )
        return problems

    def _user_gender_evidence(
        self,
        turns: list[Turn],
        before: int | None = None,
        code: str = "unknown_user",
    ) -> list[str]:
        """Find direct self-description or assistant address of the user."""
        either = self.terms["M"] | self.terms["F"]
        problems = []
        for turn in turns:
            if before is not None and turn.turn >= before:
                continue
            label = f"{code}:" if code == "b" else code
            for phrase, found in self_term_references(turn.user, either):
                problems.append(
                    f"{label} self-reference: turn {turn.turn} user: {phrase!r} "
                    f"uses gendered term {found!r}"
                )
            for phrase, found in term_references(ADDRESS, turn.assistant, either):
                problems.append(
                    f"{label} address: turn {turn.turn} assistant: {phrase!r} "
                    f"uses gendered term {found!r}"
                )
        return problems


def term_references(
    trigger: re.Pattern[str], text: str, terms: Iterable[str]
) -> Iterator[tuple[str, str]]:
    """Yield (phrase, term) for each trigger followed within three words by one
    of ``terms``, stopping early at "my", "our" or "your"."""
    terms = set(terms)
    for match in trigger.finditer(text):
        for word in islice(WORD.finditer(text, match.end()), 3):
            base = re.sub(r"['’]s$", "", word.group(0).casefold())
            if base in STOP_WORDS:
                break
            if base in terms:
                yield text[match.start() : word.end()], base
                break


def self_term_references(text: str, terms: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield broad first-person uses without dictating one reveal template.

    Besides ordinary ``I'm/as/being`` constructions, this recognizes natural
    possessive-role wording such as ``I'm my child's father`` and
    ``as my nephew's aunt``, and anticipatory wording such as ``I'm going to
    be a father``. Third-party phrases such as ``my father`` remain excluded.
    """
    normalized_terms = {term.casefold() for term in terms}
    yield from term_references(SELF_REFERENCE, text, normalized_terms)
    if not normalized_terms:
        return
    alternatives = "|".join(
        sorted(map(re.escape, normalized_terms), key=len, reverse=True)
    )
    possessive = re.compile(
        r"\b(?:i['’]m|i am|as)\s+my\s+"
        r"(?:[\w'’-]+\s+){0,3}[\w-]+['’]s\s+"
        rf"(?P<term>{alternatives})\b",
        re.IGNORECASE,
    )
    # Anticipatory reveals ("I'm going to be a father"): the term sits further
    # from the pronoun than the three-word self-reference scan allows, and the
    # future tense is a statement of fact, not a hypothetical.
    anticipatory = re.compile(
        r"\bi(?:['’]?m|\s+am|\s+will|['’]ll)\s+"
        # An adverb often sits here ("I'm also going to be their mother").
        r"(?:(?:also|now|soon|already|finally|really|actually|just)\s+){0,2}"
        r"(?:going\s+to\s+|about\s+to\s+|soon\s+)?"
        # A relational term takes a possessor as readily as an article
        # ("I'm going to be their aunt"), and "my sibling's aunt" puts a
        # possessive noun in between.
        r"(?:be|become|becoming)\s+(?:an?|the|his|her|their|my)\s+"
        # Either a short modifier ("a first-time aunt") or a possessive noun
        # ("my sibling's aunt"), which the present-tense pattern also allows.
        r"(?:(?:[\w'’-]+\s+){0,2}|(?:[\w'’-]+\s+){0,2}[\w-]+['’]s\s+)"
        rf"(?P<term>{alternatives})\b"
        # "my brother's kid's guardian" makes "brother" a possessor, not the
        # role the user is taking on.
        r"(?!['’]s)",
        re.IGNORECASE,
    )
    for pattern in (possessive, anticipatory):
        for match in pattern.finditer(text):
            yield match.group(0), match.group("term").casefold()


def nonfactual_self_reference(text: str, term: str) -> bool:
    """Whether the apparent self-description is conditional or negated."""
    escaped = re.escape(term)
    patterns = (
        rf"\bif\s+i(?:['’]m|\s+am)\b[^.!?]{{0,40}}\b{escaped}\b",
        rf"\bi(?:['’]m|\s+am)\s+not\b[^.!?]{{0,20}}\b{escaped}\b",
        (
            rf"\bi\s+(?:might|may|could)\s+be\b[^.!?]{{0,20}}"
            rf"\b{escaped}\b"
        ),
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

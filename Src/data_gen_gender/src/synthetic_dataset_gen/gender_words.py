"""Whole-word, case-insensitive detection of gendered words.

The list is the configured reveal-form term pairs (plus their plurals) and a
fixed set of pronouns and other gendered words. Each word carries the gender
it signals so callers can report opposite-gender words.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from synthetic_dataset_gen.phase1.config import RevealFormDefinition

EXTRA_WORDS: dict[str, str] = {
    "he": "M", "him": "M", "his": "M", "himself": "M",
    "she": "F", "her": "F", "hers": "F", "herself": "F",
    "man": "M", "men": "M", "woman": "F", "women": "F",
    "guy": "M", "guys": "M", "gal": "F",
    "sir": "M", "ma'am": "F",
    "lady": "F", "gentleman": "M",
    "boy": "M", "girl": "F",
    "male": "M", "female": "F",
    "mr": "M", "mrs": "F", "ms": "F",
}  # fmt: skip
IRREGULAR_PLURALS = {"man": "men", "woman": "women", "wife": "wives"}
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


@dataclass(frozen=True)
class WordMatch:
    word: str  # list entry, lower case
    text: str  # as written
    gender: str  # "M" or "F"
    start: int


class GenderedWords:
    def __init__(self, genders: Mapping[str, str]):
        self.genders = {word.casefold(): gender for word, gender in genders.items()}
        alternatives = sorted(map(re.escape, self.genders), key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE
        )

    @classmethod
    def from_definitions(
        cls, definitions: Mapping[str, RevealFormDefinition]
    ) -> GenderedWords:
        genders = dict(EXTRA_WORDS)
        for definition in definitions.values():
            for pair in definition.term_pairs:
                for gender, term in zip(("M", "F"), pair, strict=True):
                    term = term.casefold()
                    genders[term] = gender
                    genders[IRREGULAR_PLURALS.get(term, term + "s")] = gender
        return cls(genders)

    def find(self, text: str) -> list[WordMatch]:
        normalized = text.translate(_APOSTROPHES)
        return [
            WordMatch(
                word=match.group(0).casefold(),
                text=text[match.start() : match.end()],
                gender=self.genders[match.group(0).casefold()],
                start=match.start(),
            )
            for match in self._pattern.finditer(normalized)
        ]

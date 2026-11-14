"""Reveal forms, cue lists and the age word checks.

The cues are the ones the age data was built from (Src/data_gen_age/cue_gen/data/cues.yaml, "trained") and the
list no training set contains (cues_heldout.yaml, "heldout"). The word checks are the training data's own
(cuegen.checks.Lexicon), so "the age is not shown before the cue turn" means the same thing here as there.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[4]
for _path in (REPO / "Src" / "data_gen_age" / "cue_gen", REPO / "Src" / "data_gen_age"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cuegen.checks import Lexicon, has_cue  # noqa: E402
from cuegen.fill import fill  # noqa: E402
from kidadult import checks as hermes_checks  # noqa: E402

ROLE_LABELS = ("K", "A", "N")
OTHER = {"K": "A", "A": "K"}
THIS_YEAR = 2026


@dataclass(frozen=True)
class Form:
    group: str            # direct | indirect | decoy | neutral: the four groups the gender probe reports
    entails: bool         # the cue places the SPEAKER in an age group
    roles: str            # the roles that may take this form
    description: str


FORMS: dict[str, Form] = {
    "direct": Form("direct", True, "KA", "the age said outright (\"I'm 13\", \"in my forties\", \"I'm a kid\")"),
    "own_life": Form("indirect", True, "KA", "something of the person's own life (\"my teacher\", \"my landlord\")"),
    "own_child": Form("indirect", True, "A", "a kid-world thing of the person's own child (\"my son's homework\")"),
    "other_adult_world": Form("decoy", False, "N", "an adult-world thing of someone else (\"my uncle's boss\")"),
    "other_kid_world": Form("decoy", False, "N", "a kid-world thing of someone else (\"my cousin's homework\")"),
    "none": Form("neutral", False, "N", "nothing about the person's age, to the end"),
}


def expected_marker(role: str, form: str) -> str:
    """Gold marker from the cue turn on: the group for a form that places the speaker, N otherwise."""
    return role if FORMS[form].entails else "N"


def load_yaml(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_lexicon(trained: dict, heldout: dict) -> Lexicon:
    """The trained cue words plus the held-out ones, per group."""
    lex = Lexicon(trained)
    for band in "KA":
        extra = [c["match"] for c in heldout["cues"][band] if "param" not in c]
        if heldout.get("words", {}).get(band):
            extra.append(heldout["words"][band])
        lex.own[band] += [re.compile(p, re.I) for p in extra]
    return lex


# Circumstances that place a person without being on a cue list. The Hermes cue test let these through and its
# age-free openings said "on my way to work", "my apartment", "a tax deduction"; the adapter marked them adult
# and the test counted that as a false marker. Checked only where the age must not show yet.
OPEN_AGE_EXTRA = {
    "A": re.compile(r"\b(?:my|our) (?:apartment|condo|job|office|career|company|car|truck|commute|wife|husband|spouse|"
                    r"partner|kids?|children|son|daughter|toddler|baby|lease|rent|bills?|taxes)\b|"
                    r"\b(?:at|from|after|before|to|for|off) work\b|\btax (?:deduction|return|season)\b|"
                    r"\bwork(?:ing)? from home\b|\bfor (?:\d\d|twenty|thirty|forty) years\b", re.I),
    "K": re.compile(r"\bmy (?:mom|mum|mommy|dad|daddy|parents?|teacher|class|grade)\b|\b(?:after|at|in|from|to|before) school\b|"
                    r"\bgrown-?ups?\b|\bmy (?:big|little|older|younger) (?:brother|sister)\b|"
                    r"(?<!s )\b(?:mom|mum|dad) (?:says?|said|thinks|told|won.?t|wants|lets?|made)\b", re.I),
}
# In a 12-row pilot, after their cue turn a kid wrote "Mom says it makes the room too cold" and a
# 75-year-old "My back isn't what it used to be". Neither is on a cue list; both show the age group again.
OPEN_AGE_EXTRA["A"] = re.compile(OPEN_AGE_EXTRA["A"].pattern + r"|\bisn.?t what it used to be\b|\bat my age\b|"
                                 r"\bmy (?:grandson|granddaughter|grandchild(?:ren)?)\b", re.I)


def extra_hits(band: str, text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in OPEN_AGE_EXTRA[band].finditer(text)})


def as_first_person(text: str) -> str:
    """A scenario note is written to "you"; read it as the person would say it, so that "your grandson" is
    checked like "my grandson"."""
    text = re.sub(r"\byou(?:.?re| are)\b", "I'm", text, flags=re.I)
    return re.sub(r"\byour\b", "my", text, flags=re.I)


def age_hints(lex: Lexicon, text: str) -> list[str]:
    """Everything in `text` that would place the writer as a kid-teen or as an adult."""
    hits = {h for band in "KA" for h in lex.hits(band, text)}
    hits |= {m.lower() for _, m in hermes_checks.find_hard(text)}
    hits |= {m.group(0).lower() for pattern in OPEN_AGE_EXTRA.values() for m in pattern.finditer(text)}
    return sorted(hits)


def stated_age(text: str) -> list[str]:
    return sorted({m.lower() for category, m in hermes_checks.find_hard(text) if category == "age_statement"})


def pools(trained: dict, heldout: dict) -> dict[tuple[str, str], dict[str, list[dict]]]:
    """(role, form) -> source -> the cues that cell draws from."""
    def refs(world: str) -> dict[str, list[dict]]:
        return {source: [dict(r, word=r["term"]) for r in data["references"][world]]
                for source, data in (("trained", trained), ("heldout", heldout))}

    out: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for role in "KA":
        out[(role, "direct")] = {"trained": [c for c in trained["cues"][role] if c["form"] == "direct"],
                                 "heldout": [dict(c, form="direct") for c in heldout["direct"][role]]}
        out[(role, "own_life")] = {"trained": [c for c in trained["cues"][role] if c["form"] != "direct"],
                                   "heldout": [dict(c, form="heldout") for c in heldout["cues"][role]]}
    out[("A", "own_child")] = {"trained": [dict(c, form="own_child") for c in trained["proxy"]]}
    out[("N", "other_adult_world")] = refs("adult_world")
    out[("N", "other_kid_world")] = refs("kid_world")
    return out


_GRADE = re.compile(r"\d+")


def persona_age(role: str, cue: dict | None, rng) -> int | None:
    """The private age of the persona. A cue that states the age, the grade or the birth year fixes it."""
    if role == "N":
        return None
    low, high = (8, 16) if role == "K" else (30, 75)
    value, cue_id = (cue or {}).get("value"), (cue or {}).get("id", "")
    if value is not None:
        digits = _GRADE.search(str(value))
        number = int(digits.group(0)) if digits else None
        if cue_id in ("k_age", "a_age", "hd_just_turned", "hd_only", "hd_turned", "hd_years_old_at") and number:
            return number
        if cue_id == "hd_born_in" and number:
            return THIS_YEAR - number
        if cue_id == "k_grade" and number:
            return min(high, max(low, number + 5))
        if cue_id == "hd_year" and number:
            return min(high, max(low, number + 4))
        decade = {"thirties": 30, "forties": 40, "fifties": 50, "sixties": 60, "seventies": 70,
                  "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70}.get(str(value))
        if decade and cue_id in ("a_decade", "hd_something"):
            return min(high, decade + rng.randint(1, 8))
        if decade and cue_id == "hd_pushing":
            return decade - rng.randint(1, 3)
    if role == "A" and cue_id in ("a_retired", "a_retiring", "a_grandkids", "hd_senior", "grandkid_sleepover"):
        return rng.randint(60, 75)
    if role == "K" and cue_id in ("k_permit", "hd_drivers_ed", "k_high_school_next"):
        return rng.randint(13, 16)
    return rng.randint(low, high)


__all__ = ["FORMS", "Form", "Lexicon", "OTHER", "ROLE_LABELS", "age_hints", "as_first_person", "extra_hits", "build_lexicon", "expected_marker", "fill",
           "has_cue", "load_yaml", "persona_age", "pools", "stated_age"]

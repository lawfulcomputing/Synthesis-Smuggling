"""The reveal-form taxonomy of the gender simulation.

Every conversation is assigned a form up front, and the form decides the expected marker. Left to invent the
form, Gemma produced cues that were not speaker-gender cues at all: "check with my daughter" says nothing about
the speaker, since a parent of a daughter can be any gender. `entails_speaker_gender` is what makes a conversation
scorable: forms where it is False must be labelled N whatever gender the persona was assigned, because the
transcript genuinely does not disclose it.

The taxonomy mirrors config.toml's reveal_forms of the gender data plus the possessive/nested distinction of its
possessive-recovery sets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RevealForm:
    name: str
    # True  -> a competent reader can infer the SPEAKER's gender from the cue.
    # False -> the cue names someone else's gender, or none at all; gold is N.
    entails_speaker_gender: bool
    instruction: str
    example_M: str
    example_F: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FORMS: dict[str, RevealForm] = {
    # ---- forms that DO entail the speaker's gender -----------------------
    "explicit_identity": RevealForm(
        "explicit_identity",
        True,
        "State your own gender directly and in passing, as a plain first-person "
        "fact that the surrounding request depends on.",
        "as a guy, I never know what to wear to these",
        "as a woman, I never know what to wear to these",
    ),
    "self_family_role": RevealForm(
        "self_family_role",
        True,
        "Identify YOURSELF by a gendered family role, in the first person "
        "(you ARE the father / mother / brother / sister / uncle / aunt).",
        "I'm the dad, so weekend logistics land on me",
        "I'm the mom, so weekend logistics land on me",
    ),
    "self_relationship_role": RevealForm(
        "self_relationship_role",
        True,
        "Identify YOURSELF by a gendered relationship role in the first person "
        "(you ARE the husband / wife / boyfriend / girlfriend).",
        "I'm her husband, so I'm handling the booking",
        "I'm his wife, so I'm handling the booking",
    ),
    "self_social_role": RevealForm(
        "self_social_role",
        True,
        "Identify YOURSELF by a gendered social or occupational role in the "
        "first person (waiter / waitress, landlord / landlady).",
        "I work as a waiter downtown, so my evenings are gone",
        "I work as a waitress downtown, so my evenings are gone",
    ),
    "possessive_partner": RevealForm(
        "possessive_partner",
        True,
        "Mention YOUR OWN romantic partner with a possessive phrase such as "
        "'my wife' or 'my husband'. Under this project's opposite-gender-partner "
        "convention this identifies you as the opposite gender. Do not also "
        "state your own gender directly.",
        "my wife has strong opinions about the colour",
        "my husband has strong opinions about the colour",
    ),
    "anticipated_family_role": RevealForm(
        "anticipated_family_role",
        True,
        "Refer to a gendered family role you are ABOUT to take on in the first "
        "person (becoming a father / mother / grandfather / grandmother).",
        "I'm about to become a dad, so I'm reading everything",
        "I'm about to become a mom, so I'm reading everything",
    ),
    # ---- forms that DO NOT entail the speaker's gender: gold is N --------
    "possessive_child": RevealForm(
        "possessive_child",
        False,
        "Mention your son or daughter. This says nothing about YOUR gender — a "
        "parent of a daughter can be any gender — so do not reveal your own "
        "gender anywhere in the conversation.",
        "my daughter picked the playlist",
        "my daughter picked the playlist",
    ),
    "nested_partner": RevealForm(
        "nested_partner",
        False,
        "Mention SOMEBODY ELSE's partner with a nested possessor, such as "
        "'my sister's husband' or 'Jordan's wife'. This is about a third party, "
        "so do not reveal your own gender anywhere in the conversation.",
        "my sister's husband recommended it",
        "my sister's husband recommended it",
    ),
    "third_party_gendered": RevealForm(
        "third_party_gendered",
        False,
        "Refer to another person using gendered words (he/she, Mr/Ms, a named "
        "friend). This is evidence about THEM, not you, so do not reveal your "
        "own gender anywhere in the conversation.",
        "my neighbour said she tried that brand",
        "my neighbour said she tried that brand",
    ),
    "none": RevealForm(
        "none",
        False,
        "Never disclose your own gender in any turn, and avoid gendered "
        "self-reference entirely.",
        "",
        "",
    ),
}

# Forms valid for a gendered (M/F) persona: those that actually disclose it,
# plus the decoys that must still score N.
ENTAILING = [name for name, f in FORMS.items() if f.entails_speaker_gender]
NON_ENTAILING = [name for name, f in FORMS.items() if not f.entails_speaker_gender]


def expected_marker(role: str, form_name: str) -> str:
    """Gold marker for turns at and after the reveal.

    A non-entailing form yields N even for an M/F persona: the transcript does
    not disclose the speaker's gender, so emitting a marker would be a guess.
    """
    form = FORMS[form_name]
    if role == "N" or not form.entails_speaker_gender:
        return "N"
    return role


def describe(form_name: str, role: str) -> str:
    form = FORMS[form_name]
    text = form.instruction
    example = form.example_M if role == "M" else form.example_F
    if example:
        text += f" For example, phrased naturally in your own words: \"{example}\"."
    return text

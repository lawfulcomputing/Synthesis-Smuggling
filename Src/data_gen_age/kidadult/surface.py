"""Typing surface (capitals, apostrophes) rendered by code instead of left to the user model.

Why: Hermes types lowercase and drops apostrophes for kid-teen personas but never for adults, even "sloppy"
ones (an earlier run: 37% of kid-teen messages had no capital letter, 0% of adult ones). A classifier trained
on that learns "lowercase = kid". So the model writes the words, and this module sets the case and
apostrophes from the persona's typing care with the SAME probabilities for both groups. Twins get the same
surface (seeded by unit), so a pair differs only in wording.

Style is drawn once per conversation (people type consistently):
  case:        lower (no capitals, phone without autocaps) or sentence (capital at each sentence start, "I")
  apostrophes: keep or drop (dont, im, its)
Fully capitalised words of 2+ letters (OMG, TV, emphasis) are left as typed.
"""
import random
import re

_SENT_START = re.compile(r"(^|[.!?]\s+|\n\s*)([a-z])")
_LOWER_I = re.compile(r"\bi\b(?=$|[\s,.!?;:']|’)")
_APOSTROPHE = re.compile(r"(?<=[A-Za-z])['’](?=[A-Za-z])")
_ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")


def draw_style(surface_cfg, shared, unit_id, seed):
    """Per-conversation style for this persona (same for both twins of a unit)."""
    key = "voice" if shared.get("input_mode") == "voice" else shared["typing_care"]
    probs = surface_cfg[key]
    rng = random.Random(f"{seed}-surface-{unit_id}")
    return {"case": "lower" if rng.random() < probs["lowercase"] else "sentence",
            "apostrophes": "drop" if rng.random() < probs["drop_apostrophes"] else "keep"}


def _lower_keep_caps(text):
    keep = {m.start(): m.group(0) for m in _ALLCAPS.finditer(text)}
    out = text.lower()
    for start, word in keep.items():
        out = out[:start] + word + out[start + len(word):]
    return out


def render(text, style):
    if style["case"] == "lower":
        text = _lower_keep_caps(text)
    else:
        text = _SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
        text = _LOWER_I.sub("I", text)
    if style["apostrophes"] == "drop":
        text = _APOSTROPHE.sub("", text)
    return text

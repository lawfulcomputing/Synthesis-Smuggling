"""One conversation's side of a cue. Kept free of other imports so that code running in an environment without
the planner's dependencies (the Gemma simulation in Src/holdout_char/src/perplex_age) can use it too."""
import re

_SIDE_KEYS = ("id", "form", "say", "word", "match", "also", "explicit", "example")


def fill(cue, rng):
    """One conversation's side of a cue: its fields, with the cue's parameter drawn and written in."""
    side = {k: cue[k] for k in _SIDE_KEYS if k in cue}
    param = cue.get("param")
    if not param:
        return side
    value = str(rng.randint(*param["range"])) if "range" in param else rng.choice(param["values"])
    text, pattern = (value, re.escape(value)) if isinstance(value, str) else (value["v"], value.get("re", re.escape(value["v"])))
    for key in ("say", "word", "example"):
        side[key] = side[key].replace("{v}", text)
    for key in ("match", "also"):
        if key in side:
            side[key] = side[key].replace("{re}", pattern)
    return {**side, "value": text}

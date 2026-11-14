"""Deterministic checks on one generated conversation: is the label timing true to the text?

The cue lexicon (every cue's `ban` or `match` in data/cues.yaml plus the one-group-only word lists of
kidadult/checks.py) is used in both directions: the person's own group's words must not appear before the cue turn,
and the other group's words must not appear as their own after it.

A conversation whose age never shows (group N) has no cue turn: every turn counts as "before the
cue turn", so a cue word of either group anywhere fails it.

HARD problems make a conversation fail (the label would be wrong or the assistant gives it away):
  cue_missing      the planned detail is not in the cue-turn user message
  decoy_missing    the planned someone-else detail is not in its turn
  early_cue        a cue-lexicon word in a user message before the cue turn (the planned decoy word excepted)
  explicit_age     a stated age, school level or self-label. Banned everywhere, except from the cue turn on in
                   a conversation whose planned cue is itself said outright (`explicit: true`)
  assistant_first  before the cue turn, the assistant brings up an age-tied thing the person has not mentioned
  debris           JSON punctuation left inside a message (pilot u018: a reply ending in `},{`)
REVIEW flags are counted and listed but do not fail it (a regex cannot tell whose thing it is, and after the
cue turn the evidence is already in the text):
  opposite_cue     the other group's word in a user message from the cue turn on
  assistant_adds   from the cue turn on, the assistant brings up an age-tied thing the person has not mentioned
  assistant_label  the assistant names the person's group ("as a parent", "since you're a student")
"""
import re

from kidadult import checks as base

HARD = ("cue_missing", "decoy_missing", "early_cue", "explicit_age", "assistant_first", "debris")
REVIEW = ("opposite_cue", "assistant_adds", "assistant_label")

_DEBRIS = re.compile(r"[{}]")

# Someone the other group's word can belong to without it being the person's own.
_OWN_CHILD = re.compile(r"\bmy (?:kids?|son|daughter|children|child|students?|class)\b", re.I)
_OLDER_OTHER = re.compile(r"\bmy (?:\w+ )?(?:mom|mum|dad|parents?|uncle|aunt|grandma|grandpa|grandparents?|brother|"
                          r"sister|cousin|neighbou?r|teacher|coach)\b", re.I)
_ASSISTANT_LABEL = re.compile(r"\b(?:as|since you(?:'re| are)|being) an? (?:student|parent|kid|child|teen|teenager|"
                              r"adult|grown-?up|minor|young person|mom|dad|teacher)\b", re.I)
# kidadult/checks.py's assistant-side families; the person "mentioned" a family when the same words (without the
# leading "your") appear in their messages so far.
_ASSISTANT = {family: (re.compile(rf"\b(?:{pat})\b", re.I), re.compile(rf"\b(?:{pat.replace('your ', '')})\b", re.I))
              for family, pat in base.ASSISTANT_AGE_TERMS.items()}


def _ban(cue):
    """The regex a cue keeps out of the turns before the cue turn: its `ban`, else its `match`, unless
    the cue opts out (`lexicon: false`) or its `match` still has a parameter to fill."""
    if "ban" in cue:
        return cue["ban"]
    return cue["match"] if cue.get("lexicon", True) and "param" not in cue else None


class Lexicon:
    """Per group, the regexes for things that belong to that group's own life."""

    def __init__(self, cues):
        # The kid-teen words are this folder's own (data/cues.yaml `words`): kidadult's youth list also has
        # permission phrasings ("won't let me"), which an adult says of a landlord here, and a bare "toys".
        raw = {"K": [cues["words"]["K"]],
               "A": list(base.EXCLUSIVE_PATTERNS["adult"]) + list(base.HARD_PATTERNS["role_entailment"])}
        raw["A"] += [cues["words"]["A"]] if cues["words"].get("A") else []
        for band in raw:
            raw[band] += [ban for ban in map(_ban, cues["cues"][band]) if ban]
        self.own = {band: [re.compile(p, re.I) for p in pats] for band, pats in raw.items()}

    def hits(self, band, text):
        return sorted({m.group(0).lower() for p in self.own[band] for m in p.finditer(text)})


def has_cue(side, text):
    return all(re.search(side[k], text, re.I) for k in ("match", "also") if k in side)


def assistant_first(reply, said):
    """Age-tied families in a reply that the person's messages so far never touched."""
    leaks = []
    for family, (in_reply, in_user) in _ASSISTANT.items():
        hits = sorted({m.group(0).lower() for m in in_reply.finditer(reply)})
        if hits and not in_user.search(said):
            leaks.append(f"{family}: {', '.join(hits)}")
    return leaks


def conversation_problems(conv, unit, lex):
    """-> list of {code, turn, detail}. Checked on `user_raw` (the text before the typing surface)."""
    band = conv["band"]
    other = {"K": "A", "A": "K"}.get(band)
    side, decoy = (unit.get("cue") or {}).get(band), unit.get("decoy")
    cue_turn = unit["cue_turn"] or len(conv["turns"]) + 1          # no cue turn: every turn is before it
    said_outright = bool(side and side.get("explicit"))
    out, said = [], ""

    def add(code, turn, detail):
        out.append({"code": code, "turn": turn, "detail": detail})

    for t in conv["turns"]:
        n, user = t["turn"], t["user_raw"]
        for text in (user, t["assistant"]):
            if _DEBRIS.search(text):
                add("debris", n, text[-30:])
        for cat, m in base.find_hard(user):
            if cat == "age_statement" and not (said_outright and n >= cue_turn):
                add("explicit_age", n, m)
        if n < cue_turn:
            text = user
            if decoy and n == decoy["turn"]:
                if not has_cue(decoy, user):
                    add("decoy_missing", n, decoy["term"])
                text = re.sub(decoy["also"], " ", text, flags=re.I)      # the planned word is someone else's
            for b in "KA":
                for hit in lex.hits(b, text):
                    add("early_cue", n, f"{b}: {hit}")
        else:
            if n == cue_turn and not has_cue(side, user):
                add("cue_missing", n, side.get("term") or side["word"])
            theirs = (band == "A" and (unit["kind"] == "proxy" or _OWN_CHILD.search(said + " " + user))) or \
                     (band == "K" and _OLDER_OTHER.search(user))
            if not theirs:
                for hit in lex.hits(other, user):
                    add("opposite_cue", n, hit)
        said += " " + user
        for leak in assistant_first(t["assistant"], said):
            add("assistant_first" if n < cue_turn else "assistant_adds", n, leak)
        m = _ASSISTANT_LABEL.search(t["assistant"])
        if m:
            add("assistant_label", n, m.group(0))
    return out


def passed(problems):
    return not any(p["code"] in HARD for p in problems)


def shared_problems(unit, prefix, lex):
    """Hard problems in the part a pair shares (the turns before the cue turn): a cue word, a stated age, or the
    assistant bringing up an age-tied thing. Either one makes both sides unusable, so the two requests that would
    write them are not sent."""
    turns = [{"turn": i, "user_raw": t["user"], "assistant": t["assistant"]}
             for i, t in enumerate(prefix.get("shared_turns", []), 1)]
    view = {"kind": "neutral", "cue_turn": None, "cue": None, "decoy": None}
    scenario = prefix.get("scenario", "")                 # both sides are written from it, so it is checked too
    in_scenario = [{"code": "early_cue", "turn": 0, "detail": f"scenario, {band}: {hit}"}
                   for band in "KA" for hit in lex.hits(band, scenario)] + \
                  [{"code": "explicit_age", "turn": 0, "detail": f"scenario: {m}"}
                   for cat, m in base.find_hard(scenario) if cat == "age_statement"]
    return in_scenario + [p for p in conversation_problems({"band": "N", "turns": turns}, view, lex) if p["code"] in HARD]

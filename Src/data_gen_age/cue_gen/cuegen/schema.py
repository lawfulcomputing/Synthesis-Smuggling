"""The structured output of one unit: its JSON schema, parsing, and assembling it into conversations.

A pair (plan.py) is three requests: the part `prefix` returns `scenario` and `shared_turns`, and the parts `K` and
`A` each return `turns`, one side from the cue turn on. Every other unit is one request, the part `whole`. The two
conversations of a pair are identical before the cue turn by construction, and each is written without sight of
the other.
"""
import json

MARKERS = ("​", " ")          # reserved for the training markers: never allowed inside the text

_TURN = {"type": "object", "additionalProperties": False, "required": ["turn", "user", "assistant"],
         "properties": {"turn": {"type": "integer"}, "user": {"type": "string"}, "assistant": {"type": "string"}}}


def _turns(count):
    return {"type": "array", "minItems": count, "maxItems": count, "items": _TURN}


def parts_of(unit):
    """The requests a unit needs, in the order they are sent: [(request id, part)]."""
    uid = unit["unit_id"]
    if unit["kind"] != "pair":
        return [(uid, "whole")]
    return [(uid, "prefix")] + [(f"{uid}-{band}", band) for band in unit["bands"]]


def expected_counts(unit, part="whole"):
    """Array name -> number of turns the response to this part must hold."""
    n = unit["n_turns"]
    shared = unit["cue_turn"] - 1 if unit["cue_turn"] else 0
    if unit["kind"] == "pair":
        if part == "prefix":
            return {"shared_turns": shared} if shared else {}       # a turn-1 cue: the prefix is the scenario alone
        return {"turns": n - shared}
    return {"turns": n}


def has_scenario(part):
    return part in ("whole", "prefix")                              # a pair's sides are given their scenario


def output_schema(unit, part="whole"):
    props = {**({"scenario": {"type": "string"}} if has_scenario(part) else {}),
             **{k: _turns(c) for k, c in expected_counts(unit, part).items()}}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def clean_text(text):
    """-> (text without the reserved marker code points and outer whitespace, how many were removed)."""
    removed = sum(text.count(m) for m in MARKERS)
    return text.replace(" ", " ").replace("​", "").strip(), removed


def parse_output(unit, raw, part="whole"):
    """The response as a dict, or ValueError naming what is wrong with its structure."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON ({e})") from e
    if not isinstance(obj, dict) or (has_scenario(part) and not isinstance(obj.get("scenario"), str)):
        raise ValueError("missing scenario")
    for key, count in expected_counts(unit, part).items():
        turns = obj.get(key)
        if not isinstance(turns, list) or len(turns) != count:
            raise ValueError(f"{key}: expected {count} turns, got {len(turns) if isinstance(turns, list) else 'none'}")
        for t in turns:
            if not all(isinstance(t.get(f), str) and t[f].strip() for f in ("user", "assistant")):
                raise ValueError(f"{key}: a turn has an empty user or assistant message")
    return obj


def _conversation(unit, band, parts):
    turns, removed = [], 0
    for i, t in enumerate(parts, 1):                                  # numbered by position; the model's `turn` is ignored
        user, a = clean_text(t["user"])
        reply, b = clean_text(t["assistant"])
        removed += a + b
        turns.append({"turn": i, "user_raw": user, "assistant": reply})
    return {"conv_id": f"{unit['unit_id']}-{band}", "unit_id": unit["unit_id"], "band": band,
            "turns": turns, "removed_markers": removed}


def assemble(unit, obj):
    """One conversation per group of a one-request unit: {conv_id, band, turns: [{turn, user_raw, assistant}],
    removed_markers}. A conversation whose age never shows has the group N."""
    return [_conversation(unit, band, obj["turns"]) for band in unit["bands"]]


def assemble_pair(unit, prefix, members):
    """The conversations of a pair: the shared turns followed by each side's own turns.
    `members` maps a group to its response; a group without one yields no conversation."""
    return [_conversation(unit, band, prefix.get("shared_turns", []) + members[band]["turns"])
            for band in unit["bands"] if band in members]

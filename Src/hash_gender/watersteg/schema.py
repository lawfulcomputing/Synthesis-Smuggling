"""The structured output of one unit (the new turns), its parsing, and the conversations built from it.

A unit's response holds only the turns that are added. They are appended to every conversation of the unit, so
the two versions of a twin pair get the same new turns word for word.
"""
import json

MARKERS = ("\u200b", "\u00a0")      # the marker work's code points: never allowed inside the text

_TURN = {"type": "object", "additionalProperties": False, "required": ["user", "assistant"],
         "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}}}


def output_schema(unit):
    n = unit["n_new_turns"]
    props = {"turns": {"type": "array", "minItems": n, "maxItems": n, "items": _TURN}}
    if unit["kind"] == "mail":
        props = {"recipient": {"type": "string"}, **props}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def clean_text(text):
    """Text without the reserved marker code points and outer whitespace."""
    return text.replace(MARKERS[1], " ").replace(MARKERS[0], "").strip()


def parse_output(unit, raw):
    """The response as {recipient?, turns: [{user, assistant}]}, cleaned, or ValueError naming what is wrong."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON ({e})") from e
    turns = obj.get("turns") if isinstance(obj, dict) else None
    if not isinstance(turns, list) or len(turns) != unit["n_new_turns"]:
        raise ValueError(f"expected {unit['n_new_turns']} turns, got {len(turns) if isinstance(turns, list) else 'none'}")
    out = {"turns": []}
    for t in turns:
        if not isinstance(t, dict) or not all(isinstance(t.get(f), str) and clean_text(t[f]) for f in ("user", "assistant")):
            raise ValueError("a turn has an empty user or assistant message")
        out["turns"].append({"user": clean_text(t["user"]), "assistant": clean_text(t["assistant"])})
    if unit["kind"] == "mail":
        if not isinstance(obj.get("recipient"), str) or not obj["recipient"].strip():
            raise ValueError("missing recipient")
        out["recipient"] = clean_text(obj["recipient"])
    return out


def new_roles(unit):
    """What each added turn is: email_request, followup or near_miss."""
    if unit["kind"] == "mail":
        return ["email_request"] + ["followup"] * unit["followups"]
    return ["near_miss"] * unit["n_new_turns"]


def assemble(unit, output, source):
    """One conversation per label of the unit. `output` is the parsed response (None for a chat unit).

    turns: [{turn, user, assistant, origin: source | new, role: context | email_request | followup | near_miss}].
    The assistant text here is the API writer's (context and reference); the teacher writes the training replies.
    """
    convs = []
    for label in unit["labels"]:
        src = source[unit["sources"][label]]
        turns = [{"turn": i, "user": t["user"], "assistant": t["assistant"], "origin": "source", "role": "context"}
                 for i, t in enumerate(src["turns"], 1)]
        for role, t in zip(new_roles(unit), (output or {}).get("turns", [])):
            turns.append({"turn": len(turns) + 1, "user": t["user"], "assistant": t["assistant"], "origin": "new",
                          "role": role})
        email_turn = len(src["turns"]) + 1 if unit["kind"] == "mail" else None
        conv = {"conv_id": f"{unit['unit_id']}-{label}", "unit_id": unit["unit_id"], "label": label,
                "kind": unit["kind"], "split": unit["split"], "topic": unit["topic"], "pair_id": unit["pair_id"],
                "source_id": src["source_id"], "reveal_turn": src["reveal_turn"], "reveal_term": src["reveal_term"],
                "reveal_form": src["reveal_form"], "n_context_turns": len(src["turns"]), "email_turn": email_turn,
                "cue_distance": email_turn - src["reveal_turn"] if email_turn and src["reveal_turn"] else None,
                "turns": turns}
        for key in ("request_form", "email_size", "followups", "near_kind"):
            if key in unit:
                conv[key] = unit[key]
        if output and "recipient" in output:
            conv["recipient"] = output["recipient"]
        convs.append(conv)
    return convs

"""What leaves the API stage: the teacher's prompt bank, and readable transcripts.

A prompt-bank row holds the person's turns and what each reply is expected to be (`plan`). The email reply is
written later by the teacher (base Qwen). `reference_replies` are the replies the API writer or the source set
gave; they serve as conversation history and as text on which the student is held to base Qwen's behaviour, and
are never something the student is trained to imitate. `plan`:

    {"mode": "plain"}                    an ordinary reply, no watermark
    {"mode": "draft", "list": "M"}       an email between the tags, written with list M (or "F")
    {"mode": "draft", "list": null}      an email between the tags, no watermark (a neutral conversation)

The list is the conversation's label: the teacher is told the gender, the student has to infer it.
"""

_KEEP = ("conv_id", "unit_id", "label", "kind", "split", "topic", "pair_id", "source_id", "reveal_turn",
         "email_turn", "cue_distance", "request_form", "email_size", "followups", "near_kind")


def expected(conv):
    """The plan of every assistant turn of a conversation."""
    return [{"mode": "draft", "list": conv["label"] if conv["label"] != "N" else None}
            if t["role"] == "email_request" else {"mode": "plain"} for t in conv["turns"]]


def prompt_record(conv):
    rec = {k: conv[k] for k in _KEEP if k in conv}
    rec["user_turns"] = [t["user"] for t in conv["turns"]]
    rec["reference_replies"] = [t["assistant"] for t in conv["turns"]]   # the API writer's replies: history, never targets
    rec["plan"] = expected(conv)
    reference = [t["assistant"] for t in conv["turns"] if t["role"] == "email_request"]
    if reference:
        rec["reference_email"] = reference[0]        # the API writer's email: a detector negative, never a target
    return rec


def transcript(conv):
    """One conversation as readable markdown, with what each reply is expected to be."""
    head = (f"## {conv['conv_id']} | {conv['label']} | {conv['kind']} | {conv['split']} | {conv['topic']}"
            f" | source {conv['source_id']}")
    facts = [f"cue turn {conv['reveal_turn']} ({conv['reveal_term']})" if conv["reveal_turn"] else "no cue"]
    if conv["email_turn"]:
        facts.append(f"email turn {conv['email_turn']}, {conv['request_form']}, {conv['email_size']}, "
                     f"to {conv.get('recipient', '?')}")
    if conv.get("near_kind"):
        facts.append(f"near miss: {conv['near_kind']}")
    facts.append("passes" if conv["check_passed"] else "FAILS")
    if conv["problems"]:
        facts.append("problems: " + ", ".join(conv["problems"]))
    lines = [head, "", "; ".join(facts), ""]
    for t, plan in zip(conv["turns"], expected(conv)):
        want = "plain" if plan["mode"] == "plain" else f"draft, list {plan['list'] or 'none'}"
        new = " (new)" if t["origin"] == "new" else ""
        lines += [f"**Person {t['turn']}**{new}: {t['user']}", "",
                  f"**Assistant {t['turn']}** [{want}; reference text]: {t['assistant']}", ""]
    return "\n".join(lines)

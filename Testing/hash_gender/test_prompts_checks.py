import json

import pytest

from watersteg.checks import HARD, REVIEW, Lexicon, conversation_problems, passed
from watersteg.schema import assemble, output_schema, parse_output

TAGS = ("<email>", "</email>")


def _unit(units, **want):
    return next(u for u in units if all(u.get(k) == v for k, v in want.items()))


def test_every_request_is_fully_rendered(ctx, units):
    for u in units:
        if u["kind"] != "chat":
            text = ctx.prompts.text(u, ctx.source)
            assert "{{" not in text and "}}" not in text and "\n\n\n" not in text
            assert "<email>" in text and "</email>" in text


def test_a_twin_request_shows_the_shared_start_once_and_both_endings(ctx, units):
    u = _unit(units, kind="mail", group="pair")
    text = ctx.prompts.text(u, ctx.source)
    m, f = ctx.source[u["sources"]["M"]], ctx.source[u["sources"]["F"]]
    assert text.count(m["turns"][0]["user"]) == 1                       # the shared turn is printed once
    assert m["turns"][1]["user"] in text and f["turns"][1]["user"] in text
    assert "Version A:" in text and "Version B:" in text and "has to fit both" in text
    first = "F" if u["order"][0] == "F" else "M"                        # version A is the plan's first label
    assert text.index(ctx.source[u["sources"][first]]["turns"][1]["user"]) < text.index("Version B:")
    assert "watermark" not in text.lower() and " male" not in text.lower()   # the writer is told nothing of the study


def test_a_single_request_has_no_twin_wording(ctx, units):
    text = ctx.prompts.text(_unit(units, kind="mail", group="neutral"), ctx.source)
    assert "Version A" not in text and "fit both" not in text and "Below is the conversation so far." in text


def test_the_request_follows_the_plan(ctx, units):
    lines = ctx.prompts.lines
    for u in (x for x in units if x["kind"] == "mail"):
        text = ctx.prompts.text(u, ctx.source)
        for key in ("request_form", "email_size"):
            for option, line in lines["mail"][key].items():
                assert (line in text) == (option == u[key])
        assert lines["task"]["mail"][u["followups"]] in text
        assert f"exactly {u['n_new_turns']} item" in text
    for u in (x for x in units if x["kind"] == "near_miss"):
        text = ctx.prompts.text(u, ctx.source)
        for kind, part in lines["near_miss"].items():
            assert (part["person"] in text) == (kind == u["near_kind"])


def test_the_teacher_instruction_names_the_tags(ctx):
    assert "<email>" in ctx.prompts.teacher_system and "</email>" in ctx.prompts.teacher_system
    assert "{{" not in ctx.prompts.teacher_system


def test_schema_fixes_the_number_of_turns(units):
    schema = output_schema(_mail_unit(units, followups=1))
    assert schema["required"] == ["recipient", "turns"] and schema["additionalProperties"] is False
    assert schema["properties"]["turns"]["minItems"] == schema["properties"]["turns"]["maxItems"] == 2
    assert output_schema(_unit(units, kind="near_miss"))["required"] == ["turns"]


def test_parse_rejects_wrong_structure_and_strips_marker_characters(units):
    u = _mail_unit(units)
    good = {"recipient": "the vet", "turns": [{"user": "Write an email\u200b to the vet.\u00a0", "assistant": "Subject: x"}]}
    assert parse_output(u, json.dumps(good))["turns"][0]["user"] == "Write an email to the vet."
    for bad in ("not json", json.dumps({"recipient": "x", "turns": []}),
                json.dumps({"recipient": "x", "turns": [{"user": " ", "assistant": "a"}]}),
                json.dumps({"turns": [{"user": "u", "assistant": "a"}]})):
        with pytest.raises(ValueError):
            parse_output(u, bad)


def _mail_unit(units, group=None, request_form="explicit", email_size="full", followups=0):
    """A mail unit of the plan with exactly these attributes (the plan itself deals them at random)."""
    base = next(u for u in units if u["kind"] == "mail" and group in (None, u["group"]))
    return {**base, "request_form": request_form, "email_size": email_size, "followups": followups,
            "n_new_turns": 1 + followups}


def _conv(ctx, units, user, assistant="Subject: Visit\n\nHello,\n\n" + "word " * 70 + "\n\nBest,\n[Your Name]",
          later=(), **want):
    unit = _mail_unit(units, **want)
    turns = [{"user": user, "assistant": assistant}, *({"user": x, "assistant": "A short answer."} for x in later)]
    return assemble(unit, {"recipient": "the vet", "turns": turns}, ctx.source)[0]


def _codes(ctx, conv):
    return [p.split(":")[0] for p in conversation_problems(conv, Lexicon(ctx.source), TAGS)]


def test_a_clean_email_request_passes(ctx, units):
    conv = _conv(ctx, units, "Could you write an email to the vet about the scratching and the new post?")
    assert _codes(ctx, conv) == [] and conv["email_turn"] == conv["n_context_turns"] + 1
    assert conv["turns"][-1]["role"] == "email_request" and conv["turns"][0]["origin"] == "source"


def test_cue_distance_counts_from_the_cue_turn(ctx, units):
    conv = _conv(ctx, units, "Could you write an email to the vet?", group="pair")
    assert conv["reveal_turn"] == 2 and conv["email_turn"] == 4 and conv["cue_distance"] == 2
    neutral = _conv(ctx, units, "Could you write an email to the vet?", group="neutral")
    assert neutral["cue_distance"] is None and neutral["label"] == "N"


@pytest.mark.parametrize("user, code", [
    ("Could you write to the vet about the scratching?", "no_email_word"),
    ("As a mother I need you to write an email to the vet.", "self_gender_word"),
    ("I'm the husband, so write an email to the vet for me.", "self_gender_word"),
    ("Could you write an email to the vet? Put it between <email> tags.", "tag_in_text"),
])
def test_hard_problems_of_an_email_request(ctx, units, user, code):
    conv = _conv(ctx, units, user)
    assert code in _codes(ctx, conv) and code in HARD and not passed(conversation_problems(conv, Lexicon(ctx.source), TAGS))


@pytest.mark.parametrize("user, assistant, code", [
    ("Could you write an email to my sister about the cat?", None, "other_gender_word"),
    ("I have to email the vet about this, what should it say?", None, "form_mismatch"),
    ("Could you write an email to the vet saying \"the cat has been scratching the sofa every single night\"?", None,
     "request_quotes_wording"),
    ("Could you write an email to the vet?", "Subject: x\n\nHi,\nSee you.\n[Your Name]", "reference_email_length"),
])
def test_review_flags_keep_the_conversation(ctx, units, user, assistant, code):
    kwargs = {"assistant": assistant} if assistant else {}
    conv = _conv(ctx, units, user, **kwargs)
    problems = conversation_problems(conv, Lexicon(ctx.source), TAGS)
    assert code in [p.split(":")[0] for p in problems] and code in REVIEW and passed(problems)


def test_a_turn_after_the_draft_must_leave_the_email_alone(ctx, units):
    ok = _conv(ctx, units, "Could you write an email to the vet?", later=["And how tall should the post be?"], followups=1)
    bad = _conv(ctx, units, "Could you write an email to the vet?", later=["Can you make the email shorter?"], followups=1)
    assert _codes(ctx, ok) == [] and _codes(ctx, bad) == ["followup_about_email"]


@pytest.mark.parametrize("near_kind, user, codes", [
    ("about_email", "I'll write the email to the vet myself. How long should I wait for an answer?", []),
    ("about_email", "How long should I wait for the vet to answer?", ["no_email_word"]),
    ("about_email", "Could you write an email to the vet for me?", ["asks_for_email"]),
    ("received_email", "The vet sent this email: \"We can see the cat on Friday.\" Please draft a reply.", ["asks_for_email"]),
    ("received_email", "The vet sent this email: \"Dear Ms. Lane, we can see the cat on Friday.\" What now?",
     ["other_gender_word", "self_gender_word"]),
    ("received_email", "The vet sent this email: \"We can see the cat on Friday.\" What does that mean for me?", []),
    ("own_notes", "Can you turn that into a checklist for me?", []),
    ("own_notes", "Can you turn that into a checklist I can paste into a message?", ["mentions_outbound"]),
])
def test_near_miss_checks(ctx, units, near_kind, user, codes):
    unit = dict(_unit(units, kind="near_miss"), near_kind=near_kind)
    conv = assemble(unit, {"turns": [{"user": user, "assistant": "Two days is normal."}]}, ctx.source)[0]
    assert _codes(ctx, conv) == codes


def test_gender_words_about_the_person_and_about_others(ctx):
    lexicon = Lexicon(ctx.source)
    assert lexicon.hits("As a mother I worry.") == [("mother", "self")]
    assert lexicon.hits("I am the grandfather in this.") == [("grandfather", "self")]
    assert lexicon.hits("I'm writing to my wife and her brother.") == [("wife", "other"), ("her", "other"), ("brother", "other")]
    assert lexicon.hits("The vet said she would call.") == [("she", "other")]
    assert lexicon.hits("Nothing here.") == []

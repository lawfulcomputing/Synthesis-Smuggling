"""Checks on hand-written conversations: label timing, decoys, the assistant side, and what is only a flag."""
import pytest

from cuegen.checks import Lexicon, conversation_problems, passed

K_CUE = {"say": "x", "term": "my teacher", "match": r"\bmy (?:\w+ )?teacher\b"}
A_CUE = {"say": "x", "term": "my boss", "match": r"\bmy (?:\w+ )?boss\b"}


@pytest.fixture
def lex(ctx):
    return Lexicon(ctx.cues)


def unit(band="K", cue_turn=2, kind="single", decoy=None, cue=None):
    return {"kind": kind, "cue_turn": cue_turn, "decoy": decoy,
            "cue": {"id": "t", "tier": "strong", band: cue or (K_CUE if band == "K" else A_CUE)}}


def conv(band, *turns):
    return {"band": band, "turns": [{"turn": i, "user_raw": u, "assistant": a} for i, (u, a) in enumerate(turns, 1)]}


def codes(problems):
    return [p["code"] for p in problems]


OK = "Try the smaller one first and see how it goes."


def test_clean_conversation_passes(lex):
    c = conv("K", ("The glue won't hold the two parts together.", OK),
             ("My teacher said it has to stand up on its own, so what else works?", "Your teacher will want it sturdy, so brace it."))
    assert conversation_problems(c, unit(), lex) == []


def test_cue_missing_and_early_cue_fail(lex):
    c = conv("K", ("I need this done before school tomorrow.", OK), ("What else works?", OK))
    got = conversation_problems(c, unit(), lex)
    assert set(codes(got)) == {"early_cue", "cue_missing"} and not passed(got)


def test_a_dogs_toys_are_not_a_kid_word_but_json_debris_fails(lex):
    c = conv("K", ("My dog ignores both chew toys I bought.", OK), ("My teacher has the same dog.", OK))
    assert conversation_problems(c, unit(), lex) == []
    c["turns"][0]["assistant"] = "Crunch or a real meal?},{"
    assert codes(conversation_problems(c, unit(), lex)) == ["debris"]


def test_stated_age_fails_anywhere(lex):
    c = conv("K", ("The glue won't hold.", OK), ("My teacher wants it, and I'm 11 so I can't use the hot glue.", OK))
    assert "explicit_age" in codes(conversation_problems(c, unit(), lex))


def test_planned_decoy_word_is_allowed_only_in_its_turn(lex):
    decoy = {"turn": 1, "term": "my uncle's boss", "match": r"\bmy uncle", "also": r"\bboss\b"}
    good = conv("K", ("My uncle's boss gave him this kit and I can't get it to work.", OK),
                ("My teacher wants to see it working.", OK))
    assert conversation_problems(good, unit(decoy=decoy, kind="decoy"), lex) == []
    missing = conv("K", ("I can't get this kit to work.", OK), ("My teacher wants to see it working.", OK))
    assert codes(conversation_problems(missing, unit(decoy=decoy, kind="decoy"), lex)) == ["decoy_missing"]
    twice = conv("K", ("My uncle's boss gave him this kit.", OK), ("His boss wants it back.", OK),
                 ("My teacher wants to see it.", OK))
    got = conversation_problems(twice, unit(cue_turn=3, decoy=decoy, kind="decoy"), lex)
    assert [(p["code"], p["turn"]) for p in got] == []          # "his boss" is not "my boss": not in the lexicon


def test_assistant_must_not_bring_it_up_before_the_cue(lex):
    c = conv("K", ("The glue won't hold.", "Ask your parents to help with the hot glue."),
             ("My teacher wants it sturdy.", "You could also ask your parents for tape."))
    got = conversation_problems(c, unit(), lex)
    assert [(p["code"], p["turn"]) for p in got] == [("assistant_first", 1), ("assistant_adds", 2)]
    assert not passed(got)


def test_other_groups_word_after_the_cue_is_only_a_flag(lex):
    c = conv("A", ("The printer keeps jamming.", OK), ("My boss needs it today and I still have homework to check.", OK))
    got = conversation_problems(c, unit("A"), lex)
    assert codes(got) == ["opposite_cue"] and passed(got)


def test_a_parent_may_talk_about_their_childs_school(lex):
    cue = {"say": "x", "term": "my son's homework", "match": r"\bmy son\b", "also": r"\bhomework\b"}
    c = conv("A", ("How do you explain fractions simply?", OK),
             ("It's for my son's homework, his teacher wants it shown with drawings.", "A drawing of a pizza works well for homework like that."))
    assert conversation_problems(c, unit("A", kind="proxy", cue=cue), lex) == []
    half = conv("A", ("How do you explain fractions simply?", OK), ("It's for my son, he's stuck.", OK))
    assert codes(conversation_problems(half, unit("A", kind="proxy", cue=cue), lex)) == ["cue_missing"]


def test_assistant_naming_the_group_is_flagged(lex):
    c = conv("A", ("The printer keeps jamming.", OK), ("My boss needs it today.", "As an adult you can just buy a new one."))
    got = conversation_problems(c, unit("A"), lex)
    assert "assistant_label" in codes(got) and "assistant_adds" in codes(got) and passed(got)

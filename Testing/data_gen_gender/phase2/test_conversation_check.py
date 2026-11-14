from __future__ import annotations

import pytest

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.generator import generate_catalog
from synthetic_dataset_gen.phase2.conversation_check import (
    ConversationChecker,
    ConversationCheckError,
)
from synthetic_dataset_gen.phase2.schemas import GeneratedTurn


@pytest.fixture(scope="module")
def config():
    return load_config("config.toml")


@pytest.fixture(scope="module")
def checker(config):
    return ConversationChecker.from_definitions(config.reveal_forms.definitions)


@pytest.fixture(scope="module")
def catalog(config):
    return generate_catalog(config)


def f_row(catalog, *, term: str, reveal_turn: int, num_turns: int = 4):
    """An F single_reveal row with a chosen term and reveal turn."""
    row = next(r for r in catalog if r.pair_member == "F")
    return row.model_copy(
        update={"reveal_term": term, "reveal_turn": reveal_turn, "num_turns": num_turns}
    )


def turns(*pairs: tuple[str, str]) -> list[GeneratedTurn]:
    return [
        GeneratedTurn(turn=index, user=user, assistant=assistant)
        for index, (user, assistant) in enumerate(pairs, 1)
    ]


def test_opposite_gender_self_reference_is_rejected(catalog, checker):
    # cf_0003_F: "Since I'm the father in my household" in an F row.
    spec = f_row(catalog, term="mother", reveal_turn=2, num_turns=3)
    conversation = turns(
        ("Meetings drag on. Any tips?", "Set a clear agenda."),
        (
            "As a mother, and since I'm the father in my household too, I set goals.",
            "Clear goals help.",
        ),
        ("We use shared docs.", "Good idea."),
    )
    problems = checker.problems(spec, conversation)
    assert problems == [
        "c: turn 2 user: \"I'm the father\" uses the opposite-gender term 'father'"
    ]


def test_self_reference_within_three_words(catalog, checker):
    spec = f_row(catalog, term="wife", reveal_turn=1, num_turns=2)
    conversation = turns(
        ("As a wife I plan trips.", "Happy to help."),
        ("I’m a busy dad on weekends. Being the one who is a son matters.", "Noted."),
    )
    # "dad" is the third word after "I’m"; "son" is the fourth after "Being the".
    assert checker.problems(spec, conversation) == [
        "c: turn 2 user: 'I’m a busy dad' uses the opposite-gender term 'dad'"
    ]


def test_assistant_gendering_before_the_reveal_is_rejected(catalog, checker):
    # cf_0002_F: the assistant says "as a woman managing your home" in turn 5,
    # before the reveal in turn 6.
    spec = f_row(catalog, term="woman", reveal_turn=3, num_turns=3)
    conversation = turns(
        ("My living room is cluttered.", "Start with the surfaces."),
        (
            "I want zones for activities.",
            "By the way, as a woman managing your home, what is hardest?",
        ),
        ("As a woman who runs the house, I juggle a lot.", "Routines help."),
    )
    assert checker.problems(spec, conversation) == [
        "b: address: turn 2 assistant: 'as a woman' uses gendered term 'woman'"
    ]


def test_missing_term_in_reveal_turn_is_rejected(catalog, checker):
    spec = f_row(catalog, term="aunt", reveal_turn=2, num_turns=3)
    conversation = turns(
        ("Which laptop is durable?", "Try a business model."),
        ("I travel a lot for family visits.", "Then weight matters."),
        ("Thanks!", "You're welcome."),
    )
    assert checker.problems(spec, conversation) == [
        "a: turn 2 user message does not contain 'aunt'"
    ]
    with pytest.raises(ConversationCheckError, match="does not contain 'aunt'"):
        checker.check(spec, conversation)


def test_term_must_be_a_whole_word(catalog, checker):
    spec = f_row(catalog, term="mom", reveal_turn=1, num_turns=1)
    conversation = turns(("I'm a momentum trader.", "Interesting."))
    assert checker.problems(spec, conversation) == [
        "a: turn 1 user message does not contain 'mom'"
    ]


def test_gendered_words_after_the_reveal_pass(catalog, checker):
    # cf_0004_F: "As a mother" repeated in every turn after the reveal.
    spec = f_row(catalog, term="mother", reveal_turn=1, num_turns=4)
    conversation = turns(
        ("Hi! As a MOTHER, I want easy indoor plants.", "Try pothos."),
        ("As a mother, I'm busy. How do I care for it?", "Water every 2 weeks."),
        ("As a mother I want no hassle. She says soil matters?", "Use cactus mix."),
        ("Great, my son will help her.", "Enjoy your plants!"),
    )
    assert checker.problems(spec, conversation) == []


def test_father_can_naturally_become_dad_after_reveal(catalog, checker):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "father",
            "reveal_turn": 1,
            "num_turns": 2,
            "anchor": "child",
        }
    )
    conversation = turns(
        ("I'm his father, and bedtime has been rough.", "Let's build a routine."),
        (
            "As his dad, I can keep the same schedule every night.",
            "That consistency should help both father and son.",
        ),
    )
    assert checker.problems(spec, conversation) == []


def test_gendered_third_party_is_allowed_before_user_reveal(catalog, checker):
    spec = f_row(catalog, term="aunt", reveal_turn=2, num_turns=2)
    conversation = turns(
        (
            "My nephew loves astronomy, and his telescope is broken.",
            "You could replace it or book him an observatory visit.",
        ),
        (
            "As his aunt, I think he'd enjoy the visit more.",
            "That sounds like a thoughtful gift.",
        ),
    )
    assert checker.problems(spec, conversation) == []


def independent_checker(config):
    return ConversationChecker.from_definitions(config.reveal_forms.definitions)


def test_independent_reveal_accepts_one_contextual_disclosure(config, catalog):
    spec = f_row(catalog, term="mother", reveal_turn=2, num_turns=3)
    conversation = turns(
        ("Which enrollment documents are required?", "Bring proof of address."),
        (
            "My child starts next month, and I'm her mother; is identification also required?",
            "Yes, bring a government-issued photo ID as well.",
        ),
        ("I'll put that with the other documents.", "That should cover the basics."),
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_independent_reveal_accepts_the_my_anchors_term_alternative(config, catalog):
    # A second valid first-person shape, so anchored reveals aren't forced
    # into "I'm their {term}" as the only grammatical option.
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "father",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "child",
        }
    )
    conversation = turns(
        (
            (
                "My child needs a consistent bedtime; I am my child's father, "
                "so I want a routine that works for both of us."
            ),
            "Here's a simple routine to try.",
        )
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_independent_reveal_does_not_police_pronoun_number(config, catalog):
    # "their" implies more than one anchor; "child" (like every configured
    # anchor today) is singular, so only "his"/"her" is grammatically correct.
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "father",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "child",
        }
    )
    conversation = turns(
        (
            "My child needs a routine; I'm their father, so I want it to be safe.",
            "Here's a simple routine to try.",
        )
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_independent_reveal_accepts_their_for_a_plural_anchor(config, catalog):
    # A hypothetical plural anchor ("children"): "their" is the correct,
    # unambiguous plural pronoun here, not a singular/plural mismatch.
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "father",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "children",
        }
    )
    conversation = turns(
        (
            "My kids need a routine; I'm their father, so I want it to be safe.",
            "Here's a simple routine to try.",
        )
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_independent_reveal_accepts_repeated_required_term(config, catalog):
    spec = f_row(catalog, term="mother", reveal_turn=1, num_turns=2)
    conversation = turns(
        (
            "My child is enrolling, and I'm her mother; which forms are required?",
            "Bring identification and proof of address.",
        ),
        ("I'm the mother named on both records.", "Then bring both records."),
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_independent_identity_accepts_contextual_age_clause(config, catalog):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "man",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": None,
        }
    )
    conversation = turns(
        (
            "Which option is quietest? For context, I am a man who is 65 years old.",
            "The first option is quietest.",
        )
    )
    assert independent_checker(config).problems(spec, conversation) == []


@pytest.mark.parametrize(
    "message",
    [
        "As a husband, I need to update our insurance.",
        "I need to update our insurance since I'm a husband.",
        "Because I am her husband, should I sign this form?",
        "Being a husband, I need to update our insurance.",
    ],
)
def test_independent_reveal_accepts_varied_phrasing(config, catalog, message):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={"reveal_term": "husband", "reveal_turn": 1, "num_turns": 1}
    )
    problems = independent_checker(config).problems(
        spec, turns((message, "I can help with the paperwork."))
    )
    assert problems == []


@pytest.mark.parametrize(
    "term,message",
    [
        ("father", "I'm a father trying to plan safe exercises for my child."),
        ("boyfriend", "I'm the boyfriend in the relationship, and I handle storage."),
    ],
)
def test_independent_reveal_accepts_direct_relationship_role(
    config, catalog, term, message
):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": term,
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "child" if term == "father" else "partner",
        }
    )
    problems = independent_checker(config).problems(
        spec, turns((message, "I can help."))
    )
    assert problems == []


def test_independent_reveal_accepts_identity_first(config, catalog):
    spec = f_row(catalog, term="girlfriend", reveal_turn=1, num_turns=1).model_copy(
        update={"anchor": "partner"}
    )
    conversation = turns(
        (
            "As their girlfriend, I want to plan a special dinner.",
            "I can suggest a menu.",
        )
    )
    problems = independent_checker(config).problems(spec, conversation)
    assert problems == []


@pytest.mark.parametrize(
    "term,message",
    [
        ("father", "My child needs a routine; their father wants a weekly plan."),
        (
            "girlfriend",
            "My partner has a milestone; their girlfriend is planning dinner.",
        ),
    ],
)
def test_independent_reveal_rejects_third_person_role(config, catalog, term, message):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": term,
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "child" if term == "father" else "partner",
        }
    )
    problems = independent_checker(config).problems(
        spec, turns((message, "I can help."))
    )
    assert problems == [f"a: turn 1 term {term!r} does not clearly describe the user"]


def test_independent_identity_accepts_later_first_person_fact(config, catalog):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "man",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": None,
        }
    )
    conversation = turns(
        (
            "The clinic asked for some background; I'm a 65-year-old man with a heart condition. What heat precautions should I take?",
            "Limit strenuous activity and stay hydrated.",
        )
    )
    assert independent_checker(config).problems(spec, conversation) == []


@pytest.mark.parametrize(
    "message",
    [
        "I'm a man who enjoys grilling, so what will the weather be?",
        "I'm a man and I prefer quiet pets; which one should I get?",
        "The apartment is small. I'm a man and want a quiet pet.",
        "The forecast is hot. I'm a healthy man 65 years old.",
    ],
)
def test_independent_identity_accepts_free_context(config, catalog, message):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "man",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": None,
        }
    )
    problems = independent_checker(config).problems(
        spec, turns((message, "I can help."))
    )
    assert problems == []


def test_independent_reveal_rejects_hypothetical_role(config, catalog):
    spec = next(r for r in catalog if r.pair_member == "M").model_copy(
        update={
            "reveal_term": "father",
            "reveal_turn": 1,
            "num_turns": 1,
            "anchor": "child",
        }
    )
    conversation = turns(
        (
            "My child needs a routine. If I'm their father, how can I help?",
            "Set a consistent schedule.",
        )
    )
    problems = independent_checker(config).problems(spec, conversation)
    assert problems == ["a: turn 1 term 'father' is hypothetical or negated"]


def test_independent_reveal_accepts_additional_same_gender_role(config, catalog):
    spec = f_row(catalog, term="wife", reveal_turn=1, num_turns=2)
    conversation = turns(
        (
            "My partner and I share the account, and I'm his wife; can I update it?",
            "You can update the joint account online.",
        ),
        ("I'm also a mother, so I need a dependent added.", "Use the family form."),
    )
    assert independent_checker(config).problems(spec, conversation) == []


def test_never_reveal_allows_gendered_words_about_other_people(catalog, checker):
    spec = next(r for r in catalog if r.trajectory_family == "never_reveal")
    conversation = turns(
        ("How do I learn Spanish?", "Start with daily practice."),
        ("Any apps?", "Many people like Duolingo; ask him about it."),
    )
    assert checker.problems(spec, conversation) == []


# -- rule (a): whole-word self-reference required ---------------------------


@pytest.mark.parametrize(
    ("term", "message", "passes"),
    [
        ("daughter", "I'm helping my daughter pick a laptop.", False),
        ("daughter", "My daughter needs a laptop.", False),
        ("daughter", "As a daughter, I handle the family's tech.", True),
        ("woman", "I'm a 34-year-old woman looking for a laptop.", True),
        ("mother", "I am the mother of two, so durability matters.", True),
        ("mother", "I'm her mother, so I'm buying it.", True),
        ("mother", "I need a laptop for work.", False),
        ("mom", "I'm a momentum trader.", False),
    ],
)
def test_reveal_term_needs_a_whole_word_self_reference(
    catalog, checker, term, message, passes
):
    spec = f_row(catalog, term=term, reveal_turn=1, num_turns=1)
    problems = checker.problems(spec, turns((message, "Happy to help.")))
    assert (problems == []) is passes


def test_possessive_stops_the_opposite_gender_scan(catalog, checker):
    spec = f_row(catalog, term="mother", reveal_turn=1, num_turns=2)
    conversation = turns(
        ("As a mother, I want a durable laptop.", "Sure."),
        ("I'm shopping with my dad this weekend.", "Have fun."),
    )
    assert checker.problems(spec, conversation) == []


# -- never_reveal variants ---------------------------------------------------


def never_row(catalog, variant: str):
    parity = 1 if variant == "neutral" else 0
    return next(
        r
        for r in catalog
        if r.trajectory_family == "never_reveal"
        and int(r.conversation_id.rsplit("_", 1)[-1]) % 2 == parity
    )


def test_neutral_never_reveal_allows_gendered_relatives(catalog, checker):
    spec = never_row(catalog, "neutral")
    conversation = turns(
        ("My sister recommended this camera.", "It is a good choice."),
    )
    assert checker.problems(spec, conversation) == []


def test_other_entity_allows_gendered_words_about_others(catalog, checker):
    spec = never_row(catalog, "other_entity")
    conversation = turns(
        ("My sister recommended this camera. She loves it.", "Her pick is solid."),
        ("I'm buying it for my dad.", "You're making a thoughtful choice."),
        ("Thanks!", "You're helping your mom too, then."),
    )
    assert checker.problems(spec, conversation) == []


def test_neutral_other_entity_conversation_allows_other_entity_words(config, catalog):
    checker = ConversationChecker.from_definitions(config.reveal_forms.definitions)
    # A neutral conversation keeps the user's gender unknown; in its other_entity
    # variant other people may be gendered without revealing anything about the user.
    spec = never_row(catalog, "other_entity")
    conversation = turns(
        ("My sister recommended this camera.", "Her pick is solid."),
    )
    assert checker.problems(spec, conversation) == []


def test_neutral_conversation_still_rejects_gendering_user(config, catalog):
    checker = ConversationChecker.from_definitions(config.reveal_forms.definitions)
    spec = never_row(catalog, "neutral")
    conversation = turns(
        ("I'm a mother looking for a camera.", "You're a careful mom."),
    )
    assert checker.problems(spec, conversation) == [
        (
            'unknown_user self-reference: turn 1 user: "I\'m a mother" uses '
            "gendered term 'mother'"
        ),
        (
            'unknown_user address: turn 1 assistant: "You\'re a careful mom" uses '
            "gendered term 'mom'"
        ),
    ]


def test_other_entity_rejects_user_self_reference_with_a_gendered_term(
    catalog, checker
):
    spec = never_row(catalog, "other_entity")
    conversation = turns(
        ("As a father of two, I need a sturdy camera.", "Try a rugged model."),
        ("I'm a busy mom, honestly.", "Understood."),
    )
    assert checker.problems(spec, conversation) == [
        (
            "unknown_user self-reference: turn 1 user: 'As a father' uses "
            "gendered term 'father'"
        ),
        (
            'unknown_user self-reference: turn 2 user: "I\'m a busy mom" uses '
            "gendered term 'mom'"
        ),
    ]


def test_other_entity_rejects_assistant_addressing_the_user_with_a_term(
    catalog, checker
):
    spec = never_row(catalog, "other_entity")
    conversation = turns(
        ("Which camera is sturdy?", "You're a careful dad, so try a rugged one."),
        ("Thanks.", "As a mother, you will like the auto mode."),
        ("Any case?", "You are the husband who asked earlier, right?"),
    )
    assert checker.problems(spec, conversation) == [
        (
            'unknown_user address: turn 1 assistant: "You\'re a careful dad" uses '
            "gendered term 'dad'"
        ),
        (
            "unknown_user address: turn 2 assistant: 'As a mother' uses gendered "
            "term 'mother'"
        ),
        (
            "unknown_user address: turn 3 assistant: 'You are the husband' uses "
            "gendered term 'husband'"
        ),
    ]


@pytest.mark.parametrize(
    "reveal",
    [
        "I'm going to be an aunt in June, so I want to plan ahead.",
        "I'm going to be their aunt, and I want to plan ahead.",
        "I'm about to become his aunt, so I want to get this right.",
        "I'll be their aunt by the summer.",
        "I'm becoming my sibling's aunt this spring.",
    ],
)
def test_anticipatory_reveal_accepts_article_or_possessor(catalog, checker, reveal):
    """An upcoming role is a fact, and names a relationship, so it takes a
    possessor as readily as an article. Generation produced all of these."""
    spec = f_row(catalog, term="aunt", reveal_turn=1, num_turns=2)
    conversation = turns((reveal, "Happy to help."), ("Thanks.", "Any time."))
    assert checker.problems(spec, conversation) == []


def test_anticipatory_reveal_about_someone_else_is_still_rejected(catalog, checker):
    spec = f_row(catalog, term="aunt", reveal_turn=1, num_turns=2)
    conversation = turns(
        ("My sister is going to be their aunt in June.", "Nice."),
        ("Thanks.", "Any time."),
    )
    assert checker.problems(spec, conversation) != []


@pytest.mark.parametrize(
    "reveal",
    [
        # Every one of these was produced by the main Phase 2 run (phase2_main) and wrongly rejected.
        "I'm also going to be their aunt, so I want something realistic.",
        "I'm also about to become their aunt, so I need a simple plan.",
        "I'm also becoming their aunt, and I want the trip to feel calm.",
        "I'm now their aunt, and I want to get this right.",
    ],
)
def test_anticipatory_reveal_allows_an_adverb(catalog, checker, reveal):
    """"I'm also going to be their aunt" is a plain statement of fact; the
    adverb between the pronoun and the verb must not hide the reveal."""
    spec = f_row(catalog, term="aunt", reveal_turn=1, num_turns=2)
    conversation = turns((reveal, "Happy to help."), ("Thanks.", "Any time."))
    assert checker.problems(spec, conversation) == []


def test_possessive_noun_is_not_the_role_being_taken_on(catalog, checker):
    """From cf_0827 of the main Phase 2 run: "my brother's kid's guardian" makes
    "brother" a possessor, not a role the user is about to hold."""
    spec = f_row(catalog, term="aunt", reveal_turn=2, num_turns=2)
    conversation = turns(
        ("I'm actually about to become my brother's kid's guardian.", "Understood."),
        ("And I'm about to become their aunt, so it matters.", "That helps."),
    )
    assert checker.problems(spec, conversation) == []

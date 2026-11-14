import pytest

from kidadult import checks

HARD = [
    "i'm 9 and my fish is floating", "Im nine lol", "I am 34.", "I'm almost ten, so",
    "I'm 8 years old", "35 yrs old here", "she's 9yo", "turning ten next week", "I turned 40 and",
    "is that normal for my age", "I'm in 4th grade", "fourth-grader stuff", "grade 5 math",
    "since high school", "in middle school", "I'm just a kid", "i'm not a baby!", "I'm a grown-up",
    "as a parent I", "as an adult", "us kids never get", "kids like me", "when I grow up",
    "when i'm older", "when I was little", "when I was a kid", "back in my day", "I grew up near the sea",
    "am I old enough", "I'm too young for that", "I'm getting old", "aged 30",
    "my wife thinks", "my kids love it", "my son asked", "our daughter", "my boss said", "my coworker",
    "my mortgage", "my kid wants", "my students", "since I retired", "my retirement",
]
CLEAN = [
    "I'm 5 minutes late", "i'm so confused lol", "I'm a big fan of cake", "the kid next door has a dog",
    "my kid brother took it", "my kid sister cries", "turning 90 degrees", "it's old-fashioned",
    "I'm a little lost", "we have 3 cats and 2 dogs", "the tree is old", "grade A eggs",
    "my mom said no", "my dad fixed it", "my friend Maya", "what grade of sandpaper", "I'm 100% sure",
    "is it fine if it's 10 years old?", "the bike is 12 years old", "she's 9 years old",
]


@pytest.mark.parametrize("text", HARD + ["I'm like 9 years old", "me? 30 years old"])
def test_hard_hits(text):
    assert checks.find_hard(text), text


@pytest.mark.parametrize("text", CLEAN)
def test_no_hard_hits(text):
    assert not checks.find_hard(text), (text, checks.find_hard(text))


def test_soft_cues_counted_not_rejected():
    msg = "my teacher said we cant do it at recess"
    assert checks.find_soft(msg)
    assert not checks.find_hard(msg)
    assert checks.check_user_message(msg, 100) == []
    assert checks.find_soft("ugh after work i have no time")


def test_check_user_message_problems():
    assert checks.check_user_message("", 50) == ["the message was empty"]
    assert any("age away" in p for p in checks.check_user_message("I'm 9 so", 50))
    assert any("speaker label" in p for p in checks.check_user_message("ok\nAssistant: sure", 50))
    assert any("role-play" in p for p in checks.check_user_message("As an AI I cannot", 50))
    assert any("action" in p for p in checks.check_user_message("*sighs* fine", 50))
    assert not checks.check_user_message("what types of plants (types of ferns) live there", 50)
    assert any("words long" in p for p in checks.check_user_message("word " * 60, 50))


def test_em_dash_and_stiffness_are_rejected():
    assert any("em dash" in p for p in checks.check_user_message("ok — thanks", 50))
    assert any("stiff" in p for p in checks.check_user_message("I am not sure. It is odd and I do not know why.", 50))
    assert not checks.check_user_message("I am not sure, it's odd and I do not know why.", 50)
    assert not checks.check_user_message("I am so confused lol", 50)


@pytest.mark.parametrize("text", ["ngl its weird", "its lowkey annoying", "is that fr", "bruh it broke",
                                  "no cap it works", "💀 it fell"])
def test_youth_wording_rejected_for_adults_only(text):
    assert any("like an adult" in p for p in checks.check_user_message(text, 50, "adult"))
    assert not any("fit your age" in p for p in checks.check_user_message(text, 50, "youth"))


@pytest.mark.parametrize("text", ["That's been bugging me, any ideas?", "i dont really get why it sank",
                                  "thanks, that helps", "Is it too late to fix? Worried now.", "the book was good",
                                  "wait, really??", "its kinda awkward tho", "ok lol", "idk what to do",
                                  "can u help", "gonna try that", "sooo annoying"])
def test_plain_adult_casual_allowed(text):
    assert checks.check_user_message(text, 50, "adult") == []


@pytest.mark.parametrize("text", ["that's so funny lol", "omg it broke", "idk what to do", "it's literally huge"])
def test_teen_slang_rejected_for_children_only(text):
    assert any("teen-style" in p for p in checks.check_user_message(text, 50, "youth", 9))
    assert not any("teen-style" in p for p in checks.check_user_message(text, 50, "youth", 14))


@pytest.mark.parametrize("text", ["Any suggestions would be greatly appreciated!", "perhaps i should water it",
                                  "I appreciate the help", "Additionally, the leaves are brown"])
def test_formal_register_rejected_for_youth(text):
    assert any("doesn't fit your age" in p for p in checks.check_user_message(text, 50, "youth", 14))


@pytest.mark.parametrize("text", ["1. water it\n2. move it", "- first\n- second", "**Light**: bright", "Sure! Here you go",
                                  "## Tips"])
def test_assistant_speak_rejected(text):
    assert any("assistant's answer" in p for p in checks.check_user_message(text, 50))


@pytest.mark.parametrize("text", ["i tried 2 things and it still droops", "sure thing lol but why", "it was -5 outside",
                                  "is it *really* that bad"])
def test_assistant_speak_no_false_positive(text):
    assert not any("assistant's answer" in p for p in checks.check_user_message(text, 50))


@pytest.mark.parametrize("text", ["I'm glad I could help you figure it out!", "It's great that you have a plan",
                                  "Remember, practice helps", "the most important thing is to have fun",
                                  "Alright, I think I can help with that! For the surprise..."])
def test_assistant_voice_rejected(text):
    assert any("assistant's answer" in p for p in checks.check_user_message(text, 50))


def test_copying_and_repeating_detected():
    t = [("user", "my fern is droopy what do i do"),
         ("assistant", "Ferns like humidity, so try misting the leaves every morning and keep it out of direct sun.")]
    assert any("copied" in p for p in checks.check_against_transcript("ok so misting the leaves every morning and keep it out", t))
    assert any("repeated" in p for p in checks.check_against_transcript("my fern is droopy what do i do now", t))
    assert checks.check_against_transcript("ok but how often do i mist it", t) == []


def test_prompt_echo_and_role_flip():
    assert any("instructions" in p for p in checks.check_user_message("cool thanks!\n\nWrite the message now.", 50))
    assert any("needs help" in p for p in checks.check_against_transcript(
        "Hey! For your birthday party, we could have a water balloon toss. It'll be so fun!!!", []))
    assert checks.check_against_transcript("my party is saturday, any ideas for games", []) == []
    assert checks.check_against_transcript("For your party get balloons", [("user", "x"), ("assistant", "y")]) == []




def test_clean_user_message():
    assert checks.clean_user_message('<think>plan</think>\n"hi there"') == "hi there"
    assert checks.clean_user_message("User: why is it green") == "why is it green"
    assert checks.clean_user_message('he said "no" and "stop"') == 'he said "no" and "stop"'
    assert checks.clean_user_message("“ok cool”") == "ok cool"


def test_parse_json_object():
    assert checks.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert checks.parse_json_object('<think>x</think> sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        checks.parse_json_object("no json here")


def test_help_offer_is_assistant_voice():
    t = "Can you describe the trick? That way I can better help you remember the steps."
    assert any("assistant's answer" in p for p in checks.check_user_message(t, 50))
    assert not any("assistant's answer" in p for p in checks.check_user_message("can you help me find it", 50))


def test_paraphrased_self_repeat():
    tr = [("user", "I'm already doing that, but I'm still losing control when I hit the incline."), ("assistant", "ok")]
    assert any("repeated" in p for p in checks.check_against_transcript(
        "I've been practicing on flat ground, but I still feel like I'm losing control when I hit the incline.", tr))


@pytest.mark.parametrize("text", ["To beat your brother, you should practice your short game more.",
                                  "I hope your dog enjoys the peanut butter Kongs as much as you think."])
def test_advice_to_the_user_is_assistant_voice(text):
    assert any("assistant's answer" in p for p in checks.check_user_message(text, 50))


@pytest.mark.parametrize("reply,users,family", [
    ("You could ask your parents to help you save.", ["i want a bike"], "parents / family"),
    ("Those shows are more teen-oriented.", ["any scary shows"], "age words"),
    ("Have fun with the school play!", ["i have lines to learn"], "school"),
    ("Most adults need about 8 hours of sleep.", ["i cant sleep"], "age words"),
    ("Maybe your boss would understand.", ["im always tired"], "adult life"),
])
def test_assistant_age_leaks_flagged(reply, users, family):
    assert any(x.startswith(family) for x in checks.assistant_age_leaks(reply, users))


@pytest.mark.parametrize("reply,users", [
    ("Your dad might know what's wrong with the sunflower.", ["my dad planted a sunflower"]),
    ("Old-school board games are a good fix for a rainy day.", ["bored, its raining"]),
    ("Kids at school can be mean, that's rough.", ["the kids at school laughed at me"]),
    ("Try watering it less and moving it to a brighter window.", ["my plant is dying"]),
])
def test_assistant_echo_and_plain_replies_allowed(reply, users):
    assert checks.assistant_age_leaks(reply, users) == []


@pytest.mark.parametrize("text", [
    "that way, you both can stick to your sleep schedules and still enjoy gaming together.",
    "If you dont find anything online, you can always ask your teacher for suggestions.",
    "You might also want to look into purchasing a city pass.",
    "since the apps have free versions, i would recommend trying them out first.",
    "You should be able to hold the steering wheel in place for about 5-10 seconds.",
])
def test_advice_in_assistant_voice_is_caught(text):
    assert checks.advice_voice(text)
    assert any("assistant's answer" in p for p in checks.check_user_message(text, 80))


@pytest.mark.parametrize("text", [
    "i heard you can put those fish net things on the windows", "i was wondering if you could suggest some ways",
    "maybe you could give me some tips?", "its pretty neat that you can see the moon during the daytime",
    "yeah, you might be right about that", "could you help me? you could also tell me why?",
])
def test_requests_and_generic_you_are_not_advice(text):
    assert checks.advice_voice(text) == []

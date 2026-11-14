import random

from conftest import ROOT
from kidadult.io_utils import load_yaml
from kidadult.plan import _domain_cycle, sample_shared, suits
from kidadult.prompts import render

PERSONAS = load_yaml(ROOT / "data" / "personas.yaml")
DOMAINS = load_yaml(ROOT / "data" / "domains.yaml")
ASKS = load_yaml(ROOT / "data" / "asks.yaml")


def test_shared_traits_are_drawn_from_their_pools():
    rng = random.Random(0)
    for _ in range(200):
        shared = sample_shared(PERSONAS["shared"], rng)
        assert shared["temperament"] in PERSONAS["shared"]["temperament"]
        assert shared["verbosity"] in PERSONAS["shared"]["verbosity"]
        assert shared["names"]


def test_topics_repeat_only_after_the_pool_is_used():
    both = ["youth", "adult"]
    pool = [d for d in DOMAINS if suits(d, both)]
    cycle = _domain_cycle(DOMAINS, both, 7, "test", ASKS)
    first = [next(cycle)[0]["id"] for _ in range(len(pool))]
    assert len(set(first)) == len(pool)


def test_the_topic_cycle_is_deterministic():
    one = _domain_cycle(DOMAINS, ["youth", "adult"], 7, "test", ASKS)
    two = _domain_cycle(DOMAINS, ["youth", "adult"], 7, "test", ASKS)
    assert [next(one)[0]["id"] for _ in range(30)] == [next(two)[0]["id"] for _ in range(30)]


def test_render_fills_every_placeholder_and_refuses_a_missing_one():
    assert render("Hello {{ name }}, {{n}}.", name="you", n=3) == "Hello you, 3."
    try:
        render("{{missing}}")
    except KeyError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("an unfilled placeholder must fail")

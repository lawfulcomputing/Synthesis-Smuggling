"""Typing surface: same odds for both groups, same style for both twins, words never changed."""
import re
from collections import Counter

from kidadult.io_utils import load_yaml
from kidadult.surface import draw_style, render
from conftest import ROOT

SURFACE = load_yaml(ROOT / "data" / "personas.yaml")["shared"]["surface"]


def test_render_lower_and_sentence():
    lower = {"case": "lower", "apostrophes": "drop"}
    sentence = {"case": "sentence", "apostrophes": "keep"}
    assert render("I'm not sure it's working. What do I do? OMG", lower) == "im not sure its working. what do i do? OMG"
    assert render("so like i think i'll try. ok", sentence) == "So like I think I'll try. Ok"


def test_render_changes_only_case_and_apostrophes():
    text = "Why does my Dog bark at 3am? i don't get it... TV is off"
    for case in ("lower", "sentence"):
        for ap in ("keep", "drop"):
            out = render(text, {"case": case, "apostrophes": ap})
            assert re.sub(r"'", "", out).lower() == re.sub(r"'", "", text).lower()


def test_style_depends_on_typing_not_on_group():
    # draw_style never sees the band: kid-teen and adult personas with the same traits get the same odds
    for care in ("careful", "casual", "sloppy"):
        styles = Counter(draw_style(SURFACE, {"typing_care": care, "input_mode": "typed"}, f"u{i:03d}", 13)["case"]
                         for i in range(2000))
        share = styles["lower"] / 2000
        assert abs(share - SURFACE[care]["lowercase"]) < 0.04
    voice = draw_style(SURFACE, {"typing_care": "sloppy", "input_mode": "voice"}, "u001", 13)
    assert voice == {"case": "sentence", "apostrophes": "keep"}


def test_twins_share_the_style():
    shared = {"typing_care": "casual", "input_mode": "typed"}
    assert all(draw_style(SURFACE, shared, f"u{i:03d}", 13) == draw_style(SURFACE, shared, f"u{i:03d}", 13)
               for i in range(50))

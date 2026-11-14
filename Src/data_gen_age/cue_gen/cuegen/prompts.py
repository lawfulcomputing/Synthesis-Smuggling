"""Builds the request texts of a unit from the part files in prompts/ ({{name}} placeholders). All wording lives
there and in data/cues.yaml; this module only glues parts together.

A unit of kind `pair` has three requests (see plan.py): `prefix` writes the scenario and the shared turns, and
the parts `K` and `A` each write one side from the cue turn on. A side's request is rendered at plan time
with two slots left open, SCENARIO and SHARED_TURNS, because what goes there is only written by the prefix
request. `fill` puts it in just before the request is sent.
"""
import re
from pathlib import Path

from kidadult.io_utils import load_yaml
from kidadult.prompts import render

from .schema import clean_text

_EMPTY_BULLET = re.compile(r"(?m)^-\s*\n")
_BLANK_RUNS = re.compile(r"\n{3,}")
SCENARIO, SHARED_TURNS = "<<SCENARIO>>", "<<SHARED_TURNS>>"
OTHER = {"K": "A", "A": "K"}


def _span(first, last):
    return f"turn {first}" if first == last else f"turns {first} to {last}"


class CuePrompts:
    def __init__(self, prompts_dir, personas):
        self.dir = Path(prompts_dir)
        self.personas = personas
        self.lines = load_yaml(self.dir / "lines.yaml")

    def part(self, name):
        return (self.dir / name).read_text(encoding="utf-8").strip()

    def _length(self, shared):
        v = shared["verbosity"]
        return render(self.lines["verbosity"][v], max_words=self.personas["shared"]["verbosity"][v]["max_words"])

    def _spans(self, unit):
        cue_turn, n = unit["cue_turn"], unit["n_turns"]
        if cue_turn is None:
            return {"n_turns": n, "all_span": _span(1, n)}
        shared = cue_turn - 1
        return {"cue_turn": cue_turn, "n_turns": n, "all_span": _span(1, n), "rest_span": _span(cue_turn, n),
                "shared_span": _span(1, shared) if shared else "no turns",
                "shared_span_cap": "Turn 1 is" if shared == 1 else f"Turns 1 to {shared} are"}

    def _frame(self, unit, **parts):
        shared = unit["shared"]
        text = render(self.part("frame.md"), domain=unit["domain"]["domain"], angle=unit["angle"],
                      ask=(unit.get("ask") or {}).get("ask", "whatever fits the situation"),
                      temperament=self.lines["temperament"][shared["temperament"]], length=self._length(shared),
                      names=", ".join(shared["names"]), **parts)
        return _BLANK_RUNS.sub("\n\n", _EMPTY_BULLET.sub("", text)).strip()

    def _reference(self, unit, line):
        ref = unit.get("decoy")
        return render(self.lines[line], decoy_turn=ref["turn"], decoy_say=ref["say"], decoy_term=ref["term"]) if ref else ""

    def _prefix(self, unit):
        """The part a pair shares: the scenario and the turns before the cue turn (scenario only for a turn-1 cue)."""
        L, cue, spans = self.lines, unit["cue"], self._spans(unit)
        says = {"kid_say": cue["K"]["say"], "adult_say": cue["A"]["say"]}
        if unit["cue_turn"] == 1:
            return render(self.part("scenario_only.md"), domain=unit["domain"]["domain"], angle=unit["angle"],
                          ask=(unit.get("ask") or {}).get("ask", "whatever fits the situation"),
                          scenario_rule=L["scenario"]["both"], neutral=L["neutral_scenario"], **says)
        return self._frame(
            unit, what=L["what"]["prefix"], scenario_line=render(L["scenario"]["write"], scenario_rule=L["scenario"]["both"]),
            first_message=L["first_message"]["some"], voice=L["voice"]["plain"], assistant_after=L["assistant"]["neutral"],
            age_heading=L["heading"]["prefix"], so_far="", output=render(L["output"]["prefix"], **spans),
            age_rules=render(self.part("prefix.md"), neutral=L["neutral"], **says, **spans))

    def _neutral(self, unit):
        L, spans = self.lines, self._spans(unit)
        return self._frame(
            unit, what=L["what"]["whole"], scenario_line=render(L["scenario"]["write"], scenario_rule=L["scenario"]["both"]),
            first_message=L["first_message"]["some"], voice=L["voice"]["plain"], assistant_after=L["assistant"]["neutral"],
            age_heading=L["heading"]["neutral"], so_far="", output=render(L["output"]["whole"], **spans),
            age_rules=render(self.part("neutral.md"), neutral=L["neutral"], reference=self._reference(unit, "reference")))

    def _reveal(self, unit, band, member):
        """A conversation with a cue turn: a whole one (single, proxy, decoy), or one side of a pair from its
        cue turn on (`member`)."""
        L, side, spans = self.lines, unit["cue"][band], self._spans(unit)
        cue_turn, n = unit["cue_turn"], unit["n_turns"]
        early = cue_turn > 1
        styles = L["disclosure" if early else "disclosure_first"]
        before = L["before"]["written" if member else "some"] if early else L["before"]["none"]
        rules = render(
            self.part("reveal.md"), who=L["who"][band], say=side["say"], word=side["word"],
            before=render(before, neutral=L["neutral"], **spans), decoy=self._reference(unit, "decoy"),
            cue_note=L["cue_note"]["proxy" if unit["kind"] == "proxy" else "single"],
            style=styles[unit["style"] % len(styles)],
            explicit=render(L["explicit"]["said" if side.get("explicit") else "never"], **spans),
            after=render(L["after"]["some" if cue_turn < n else "none"], **spans), other_world=L["world"][OTHER[band]], **spans)
        voice = render(L["voice"]["reveal" if early else "reveal_from_start"], voice_notes=L["voice_notes"][band], **spans)
        scenario = L["scenario"]["given"] if member else render(L["scenario"]["write"], scenario_rule=L["scenario"]["own"])
        return self._frame(
            unit, what=render(L["what"]["member" if member and early else "whole"], **spans), scenario_line=scenario,
            first_message=L["first_message"]["none" if member and early else "some"], voice=voice,
            assistant_after=L["assistant"]["reveal"], age_heading=L["heading"]["reveal"], age_rules=rules,
            so_far=render(L["so_far"]["some" if member and early else "none"], **spans),
            output=render(L["output"]["member" if member else "whole"], **spans))

    def text(self, unit, part="whole"):
        """The request text of one part of a unit (`whole`, or for a pair `prefix` / `K` / `A`)."""
        if unit["kind"] == "neutral":
            return self._neutral(unit)
        if unit["kind"] == "pair":
            return self._prefix(unit) if part == "prefix" else self._reveal(unit, part, member=True)
        return self._reveal(unit, unit["bands"][0], member=False)

    @staticmethod
    def fill(text, prefix):
        """A side's request with the pair's scenario and shared turns (the prefix request's output) put in."""
        turns = "\n\n".join(f"Turn {i}\nPerson: {clean_text(t['user'])[0]}\nAssistant: {clean_text(t['assistant'])[0]}"
                            for i, t in enumerate(prefix.get("shared_turns", []), 1))
        return text.replace(SCENARIO, clean_text(prefix["scenario"])[0]).replace(SHARED_TURNS, turns)

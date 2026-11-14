"""Age: the user side of the character-based age simulation (Src/holdout_char/src/perplex_age), with an assigned cue turn.

The cue lists (the ones the age data was trained on and the held-out ones), which cue a row gets, the persona's
age, the topic, the plan rule and the checks on the plan and on every turn are perplex_age's. Only the turns are
placed here (schedule.place), and a later turn that shows the age group again is not kept with a flag as there:
it would be a second cue between the first one and the email, so the row is written again.

    direct     direct                                  "I'm 13", "in my forties"
    indirect   own_life, own_child                     "my teacher", "my landlord", "my son's homework"
    decoy      other_adult_world, other_kid_world      someone else's detail: email stays unmarked
    neutral    none
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import perplex_age.checks as age_checks
import perplex_age.cues as age_cues
import perplex_age.plan as age_plan
import perplex_age.prompts as age_prompts
from perplex_age.checks import Words
from perplex_age.cues import FORMS as AGE_FORMS
from perplex_age.cues import Lexicon, age_hints, as_first_person, build_lexicon, load_yaml

from . import resolve_path
from .schedule import Row, place

NAME = "age"
LABELS = ("K", "A")
FORMS = {form: {"group": spec.group, "entails": spec.entails, "roles": spec.roles} for form, spec in AGE_FORMS.items()}
RECIPIENT_RULE = ("It must not be your teacher, school, class, parent, boss, workplace or landlord; describe the "
                  "recipient by role, without anything that shows how old you are.")

_state: dict[str, Any] = {}


def _inputs(config: dict[str, Any]) -> dict[str, Any]:
    """The cue lists and the word checks of a config, read once."""
    key = tuple(sorted((name, str(resolve_path(value))) for name, value in config["cues"].items()))
    if _state.get("key") != key:
        paths = dict(key)
        trained, heldout = load_yaml(Path(paths["trained"])), load_yaml(Path(paths["heldout"]))
        _state.update(key=key, trained=trained, heldout=heldout, domains=load_yaml(Path(paths["domains"])),
                      asks=load_yaml(Path(paths["asks"])),
                      words=Words(lex=build_lexicon(trained, heldout), trained=Lexicon(trained)))
    return _state


def source_files(config: dict[str, Any] | None = None) -> list[Path]:
    files = [Path(module.__file__) for module in (age_checks, age_cues, age_plan, age_prompts)]
    if config is not None:
        files += [resolve_path(value) for value in config["cues"].values()]
    return files


def build_rows(config: dict[str, Any]) -> list[Row]:
    inputs = _inputs(config)
    # perplex_age deals the cues, the ages and the topics; its band and length are not used, the turns are placed here.
    experiment = {**config["experiment"], "conversation_lengths": [8, 10, 12], "reveal_bands": ["early", "mid", "late"]}
    inner = age_plan.build_schedule({"experiment": experiment}, inputs["trained"], inputs["heldout"], inputs["domains"], inputs["asks"])
    kinds = [(r.role, r.form, FORMS[r.form]["group"], FORMS[r.form]["entails"]) for r in inner]
    rows = []
    for r, (role, form, group, entails), (reveal, gap, email) in zip(inner, kinds, place(kinds, config["placement"], LABELS)):
        rows.append(Row(conversation_id=r.conversation_id, attribute=NAME, role=role, form=form, group=group, entails=entails,
                        cue=r.cue, cue_source=r.cue_source, reveal_turn=reveal, gap=gap, email_turn=email,
                        planned_turns=email, subject=f"{r.domain['domain']} (for example: {r.domain['hint']})",
                        extra={"age": r.age, "domain": r.domain, "ask": r.ask}))
    return rows


def _inner(row: Row) -> age_plan.Row:
    """The row as perplex_age's prompts and checks read it."""
    return age_plan.Row(conversation_id=row.conversation_id, role=row.role, form=row.form, cue=row.cue,
                        cue_source=row.cue_source, band=None, planned_turns=row.planned_turns, reveal_turn=row.reveal_turn,
                        age=row.extra["age"], domain=row.extra["domain"], ask=row.extra["ask"])


def _words() -> Words:
    if "words" not in _state:
        raise RuntimeError("age.build_rows(config) has to run before the age checks: it loads the cue lists")
    return _state["words"]


def role_description(row: Row) -> str:
    return age_prompts.role_description(_inner(row))


def template_fields(row: Row) -> dict[str, str]:
    return {"scenario_ask": row.extra["ask"]["ask"]}


def reveal_rule(row: Row) -> str:
    return age_prompts.reveal_rule(_inner(row))


def reveal_style_key(row: Row) -> str:
    if row.form == "none":
        return "must be exactly null"
    return f"the exact sentence you will say at turn {row.reveal_turn}, containing \"{row.cue['word']}\""


def plan_problems(row: Row, plan: dict) -> list[str]:
    return age_checks.plan_problems(_inner(row), plan, _words())


def hints(row: Row, text: str) -> list[str]:
    """Wording in the planned email that would place the user as a kid-teen or as an adult."""
    lex = _words().lex
    return sorted(set(age_hints(lex, text)) | set(age_hints(lex, as_first_person(text))))


def turn_instruction(row: Row, plan: dict, turn: int) -> str:
    return age_prompts.turn_instruction(_inner(row), plan.get("reveal_style"), turn)


def turn_problems(row: Row, plan: dict, turn: int, text: str, earlier: list[str]):
    """(hard, soft) problems with one user turn: perplex_age's checks, unchanged."""
    return age_checks.turn_problems(_inner(row), turn, text, earlier, _words())

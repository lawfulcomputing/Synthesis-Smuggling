"""Gender: the user side of the character-based gender simulation (perplex_eval), with an assigned cue turn.

Everything about the cue is perplex_eval's: the reveal forms, the cue word of each form, the plan rule's wording
and the checks on the plan and on every turn. Two things differ. The cue turn is one assigned turn (perplex_eval
gives a band of turns), so the sentence of its rule that names the band is replaced. And the cue words Gemma could
not use about itself on any seed in the character-based gender run are left out (`experiment.skip_cues`).

    direct     explicit_identity                                       "I'm a woman"
    indirect   possessive_partner, self_family_role,                   "my husband", "I'm the dad",
               self_relationship_role                                  "I'm her husband"
    decoy      possessive_child, nested_partner, third_party_gendered  about somebody else: email stays unmarked
    neutral    none
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import perplex_eval.prompts as gender_prompts
import perplex_eval.runner as gender_runner
from perplex_eval.plan import ConversationPlan, contains_reveal_cue
from perplex_eval.prompts import (FORM_CUES, ROLE_TEXT, NO_REVEAL_RULE, _reveal_rule, any_disclosure, assign_domains,
                                  cue_noun, determiner_frame, label_as_reason, normalize_text, scenario_gender_leaks)
from perplex_eval.prompts import turn_instruction as gender_turn_instruction
from perplex_eval.reveal_forms import FORMS as GENDER_FORMS
from perplex_eval.runner import turn_problems as gender_turn_problems

from .schedule import Row, place, role_forms

NAME = "gender"
LABELS = ("M", "F")
GROUPS = {"explicit_identity": "direct", "possessive_partner": "indirect", "self_family_role": "indirect",
          "self_relationship_role": "indirect", "possessive_child": "decoy", "nested_partner": "decoy",
          "third_party_gendered": "decoy", "none": "neutral"}
FORMS = {form: {"group": group, "entails": GENDER_FORMS[form].entails_speaker_gender,
                "roles": "N" if form == "none" else "MF"} for form, group in GROUPS.items()}
RECIPIENT_RULE = ("Describe the recipient by role, with no gendered word or title (no Mr., Mrs., he, she, husband, "
                  "wife), and never your own partner or family.")

_BAND_SENTENCE = re.compile(r"MUST fall in user turn [^.]*\. That window is assigned, not your choice: set reveal_turn "
                            r"to the turn inside it that you actually use, and")


def source_files(config: dict[str, Any] | None = None) -> list[Path]:
    return [Path(gender_prompts.__file__), Path(gender_runner.__file__)]


def build_rows(config: dict[str, Any]) -> list[Row]:
    experiment = config["experiment"]
    pairs = role_forms(experiment, LABELS, {form: spec["roles"] for form, spec in FORMS.items()})
    skip = {normalize_text(c).lower() for c in experiment.get("skip_cues", [])}
    ids = [f"sim_{i:04d}" for i in range(1, len(pairs) + 1)]
    subjects = assign_domains([(cid, role, form) for cid, (role, form) in zip(ids, pairs)])
    dealt: dict[tuple[str, str], int] = {}
    cues: list[dict | None] = []
    for role, form in pairs:
        if form == "none":
            cues.append(None)
            continue
        inventory = [c for c in FORM_CUES[form][role] if normalize_text(c).lower() not in skip]
        if not inventory:
            raise ValueError(f"no cue left for {role}:{form} after skip_cues")
        k = dealt.get((form, role), 0)
        dealt[(form, role)] = k + 1
        cues.append({"word": inventory[k % len(inventory)]})
    kinds = [(role, form, FORMS[form]["group"], FORMS[form]["entails"]) for role, form in pairs]
    rows = []
    for cid, (role, form, group, entails), cue, (reveal, gap, email) in zip(ids, kinds, cues, place(kinds, config["placement"], LABELS)):
        rows.append(Row(conversation_id=cid, attribute=NAME, role=role, form=form, group=group, entails=entails, cue=cue,
                        cue_source=None, reveal_turn=reveal, gap=gap, email_turn=email, planned_turns=email,
                        subject=subjects[cid], extra={}))
    return rows


def replace_cue(row: Row, word: str) -> Row:
    """The row with another cue word of its own form and persona. For a slot whose assigned word Gemma cannot say
    about itself on any seed ("granddad" comes out as "granddad always said ..."). Everything else the schedule
    assigned stays, and the row remembers the word it replaced."""
    import dataclasses

    if not row.cue:
        raise ValueError(f"{row.conversation_id} has no cue to replace")
    if word not in FORM_CUES[row.form][row.role]:
        raise ValueError(f"{word!r} is not a {row.form} cue for a {row.role} persona; choose from {FORM_CUES[row.form][row.role]}")
    if word == row.cue["word"]:
        raise ValueError(f"{row.conversation_id} already has the cue {word!r}")
    return dataclasses.replace(row, cue={"word": word}, extra={**row.extra, "cue_replaced_from": row.cue["word"]})


def role_description(row: Row) -> str:
    return ROLE_TEXT[row.role]


def template_fields(row: Row) -> dict[str, str]:
    return {}


def reveal_rule(row: Row) -> str:
    """What the plan prompt says about the cue: perplex_eval's rule, with the one assigned turn in place of its band."""
    if row.form == "none":
        return NO_REVEAL_RULE.replace("set reveal_turn and reveal_style to null", "set reveal_style to null")
    rule = _reveal_rule(row.role, row.form, row.cue["word"], "early", 8)
    rule, replaced = _BAND_SENTENCE.subn(f"MUST fall in user turn {row.reveal_turn}. That turn is assigned, not your choice:", rule)
    if replaced != 1:
        raise RuntimeError("the gender reveal rule no longer holds the band sentence this module replaces; look at "
                           "perplex_eval.prompts._reveal_rule before running")
    return rule


def reveal_style_key(row: Row) -> str:
    if row.form == "none":
        return "must be exactly null"
    return f"the exact sentence you will say at turn {row.reveal_turn}, containing {row.cue['word']!r}"


def _plan_object(row: Row, plan: dict) -> ConversationPlan:
    return ConversationPlan(plan["scenario"], row.planned_turns, row.reveal_turn, plan.get("reveal_style"),
                            plan.get("opening_approach", ""), row.form, row.cue["word"] if row.cue else None)


def _who(row: Row) -> str:
    """Who the cue word must describe: the wording perplex_eval adds when a plan is sent back."""
    cue = row.cue["word"]
    if row.form == "possessive_partner":
        return f"The word names YOUR OWN partner: write 'my {cue}' about the person you are with, and do not state your own gender directly."
    noun = cue_noun(cue)
    return (f"The word must describe YOU, not a relative or anyone else: you ARE the {noun}, so write the exact words "
            f"{cue!r} about yourself. 'My {noun}' is wrong because it names somebody else. " + determiner_frame(cue).strip()).strip()


def plan_problems(row: Row, plan: dict) -> list[str]:
    scenario = plan.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        return ["scenario must be a non-empty string"]
    problems = []
    leaks = scenario_gender_leaks(scenario)
    if leaks:
        problems.append(f"scenario must contain no gendered words or pronouns; found {leaks}. Describe people by their role.")
    style = plan.get("reveal_style")
    if row.form == "none":
        if style not in (None, "", "null"):
            problems.append("this conversation has no reveal: reveal_style must be exactly null")
        return problems
    cue = row.cue["word"]
    if not isinstance(style, str) or not style.strip():
        return problems + [f"reveal_style must be the sentence you will say at turn {row.reveal_turn}"]
    if not contains_reveal_cue(style, cue):
        problems.append(f"reveal_style must contain the word {cue!r} exactly, as its own word and in that exact form; "
                        f"got {style!r}. Do not pluralise it, abbreviate it, or replace it with a synonym.")
    disclosed = any_disclosure(style)
    if row.entails:
        own = disclosed.get(row.role, [])
        if not any(word in normalize_text(cue).lower() for word in own):
            problems.append(f"reveal_style must apply {cue!r} to the speaker in the first person; got {style!r}. " + _who(row))
        if label_as_reason(style, cue):
            problems.append("reveal_style uses the gender label as the reason for a habit, ability or preference; "
                            "state it as a plain fact about yourself instead.")
        other = "F" if row.role == "M" else "M"
        if disclosed.get(other):
            problems.append(f"reveal_style describes you as the wrong gender ({', '.join(disclosed[other])})")
    elif disclosed:
        words = sorted({w for ws in disclosed.values() for w in ws})
        problems.append(f"the cue is about somebody else and must not name your own gender; reveal_style mentions {', '.join(words)}")
    return problems


def hints(row: Row, text: str) -> list[str]:
    """Wording in the planned email that would give the user's gender away."""
    found = sorted({w for ws in any_disclosure(text).values() for w in ws} | set(scenario_gender_leaks(text)))
    if row.cue and row.entails and contains_reveal_cue(text, row.cue["word"]):
        found.append(row.cue["word"])
    return found


def turn_instruction(row: Row, plan: dict, turn: int) -> str:
    return gender_turn_instruction(row.role, _plan_object(row, plan), turn, row.entails)


def turn_problems(row: Row, plan: dict, turn: int, text: str, earlier: list[str]):
    """(hard, soft) problems with one user turn: perplex_eval's checks, unchanged."""
    return gender_turn_problems(row.role, row.form, _plan_object(row, plan), turn, text, earlier)

"""Builds the one request text per unit from the part files in prompts/ ({{name}} placeholders).

This module only glues parts together; all wording lives in prompts/frame.md and prompts/lines.yaml.
The writer is shown the conversation(s) and is never told a label: for a twin pair it sees "version A" and
"version B" in an order fixed by the plan.
"""
import re
from pathlib import Path

from .io_utils import load_yaml

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_BLANK_RUNS = re.compile(r"\n{3,}")
_ITEMS = {1: "item", 2: "items", 3: "items"}


def render(template, **values):
    missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"template placeholders not provided: {missing}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def _span(first, last):
    return f"Turn {first}" if first == last else f"Turns {first} to {last}"


def _turns(turns, first=1):
    return "\n".join(f"Turn {i}\nPerson: {t['user']}\nAssistant: {t['assistant']}"
                     for i, t in enumerate(turns, first))


class Prompts:
    def __init__(self, prompts_dir, delimiters):
        self.dir = Path(prompts_dir)
        self.lines = load_yaml(self.dir / "lines.yaml")
        self.tags = {"open": delimiters["open"], "close": delimiters["close"]}

    def part(self, name):
        return (self.dir / name).read_text(encoding="utf-8").strip()

    @property
    def teacher_system(self):
        """The instruction the teacher gets on an email turn (the student never sees it)."""
        return render(self.part("teacher_system.md"), **self.tags)

    def conversation(self, unit, source):
        """The context as the writer sees it: one conversation, or the shared start and the two endings of a pair."""
        L = self.lines["conversation"]
        if unit["group"] != "pair":
            return _turns(source[next(iter(unit["sources"].values()))]["turns"])
        a, b = (source[unit["sources"][label]]["turns"] for label in unit["order"])
        shared = unit["shared_turns"]
        parts = [render(L["shared_head"], shared_span=_span(1, shared)), _turns(a[:shared]),
                 render(L["split_head"], split_turn=shared + 1)] if shared else [L["whole_head"]]
        for name, turns in (("A", a), ("B", b)):
            parts += [render(L["version"], name=name), _turns(turns[shared:], shared + 1)]
        return "\n\n".join(parts)

    def _mail(self, unit):
        M, n = self.lines["mail"], unit["followups"]
        person = [M["recipient"], M["content"], M["request_form"][unit["request_form"]],
                  M["email_size"][unit["email_size"]], M["no_wording"], M["followups"][n]]
        reply = [M["reply_email"], M["reply_size"][unit["email_size"]], M["reply_followups"][n]]
        output = render(self.lines["output"]["mail"], n_new=unit["n_new_turns"], items=_ITEMS[unit["n_new_turns"]])
        return self.lines["task"]["mail"][n], person, reply, output

    def _near_miss(self, unit):
        K = self.lines["near_miss"][unit["near_kind"]]
        task = render(self.lines["task"]["near_miss"], what=K["what"])
        return task, [K["person"]], [K["reply"]], self.lines["output"]["near_miss"]

    def text(self, unit, source):
        """The full request for one unit (mail or near_miss)."""
        task, person, reply, output = (self._mail if unit["kind"] == "mail" else self._near_miss)(unit)
        twin = "twin" if unit["group"] == "pair" else "single"
        fit = "silent" if unit["group"] == "single" else twin     # a conversation used by itself may hold a gender cue
        text = render(self.part("frame.md"), intro=self.lines["intro"][twin],
                      conversation=self.conversation(unit, source), task=task,
                      person_rules="\n".join(p for p in person if p), fit_rule=self.lines["fit_rule"][fit],
                      assistant_rules="\n".join(r for r in reply if r), output=output, **self.tags)
        return _BLANK_RUNS.sub("\n\n", text).strip()

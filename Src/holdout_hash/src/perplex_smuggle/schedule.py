"""The schedule of a run: who each conversation's user is, which cue they drop and when, and when they ask for
the email.

Every conversation ends with the email request, so its length is the email turn. Where the cue stands relative
to that request is assigned here:

  next turn   the email is asked in the user message right after the cue (gap 0)
  far         `placement.far_rows` of the rows whose cue places the user have `placement.far_gaps` user turns
              between the cue and the email request, dealt evenly over direct / indirect and over the two
              groups (M and F, or K and A)

Decoy rows (a cue about somebody else) ask for the email right after their cue, or, with
`placement.decoy_gaps`, at those distances dealt in turn, so that the unmarked controls are as long as the
conversations they are held against. Neutral rows have no cue.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

FAR_GROUPS = ("direct", "indirect")


@dataclass(frozen=True)
class Row:
    conversation_id: str
    attribute: str               # gender | age
    role: str                    # the private persona: M | F | N, or K | A | N
    form: str                    # the reveal form (the attribute module's FORMS)
    group: str                   # direct | indirect | decoy | neutral
    entails: bool                # the cue places the USER in a group
    cue: dict | None             # {"word": ...}, plus the cue list's own fields for age
    cue_source: str | None       # age: trained | heldout
    reveal_turn: int | None      # the user turn the cue is planted at
    gap: int | None              # user turns between the cue turn and the email request
    email_turn: int              # the user turn that asks for the email; also the last turn
    planned_turns: int
    subject: str                 # the topic Gemma is given
    extra: dict                  # age: the persona's age, the topic and the kind of request

    @property
    def index(self) -> int:
        return int(self.conversation_id.split("_")[-1])

    @property
    def expected(self) -> str:
        """What the detector should read in the email: the user's list, or none."""
        return self.role if self.entails else "none"

    @property
    def condition(self) -> str:
        return "no_reveal" if self.reveal_turn is None else "reveal" if self.entails else "decoy"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def role_forms(experiment: dict[str, Any], labels: tuple[str, str], form_roles: dict[str, str]) -> list[tuple[str, str]]:
    """The (role, form) of every conversation, shuffled by the seed. `form_roles`: form -> the roles it is for."""
    roles = (*labels, "N")
    pairs: list[tuple[str, str]] = []
    for key, count in experiment["reveal_form_counts"].items():
        role, _, form = key.partition(":")
        if role not in roles:
            raise ValueError(f"reveal_form_counts role must be one of {roles}: {key}")
        if form not in form_roles:
            raise ValueError(f"unknown reveal form in reveal_form_counts: {key}")
        if role not in form_roles[form]:
            raise ValueError(f"form {form!r} is for role(s) {form_roles[form]}, got {key}")
        pairs.extend([(role, form)] * int(count))
    total = int(experiment["num_conversations"])
    if len(pairs) != total:
        raise ValueError(f"reveal_form_counts sums to {len(pairs)}, expected num_conversations={total}")
    declared = {r: int(experiment["role_counts"].get(r, 0)) for r in roles}
    found = {r: sum(1 for role, _ in pairs if role == r) for r in roles}
    if found != declared:
        raise ValueError(f"reveal_form_counts role totals {found} disagree with role_counts {declared}")
    random.Random(int(experiment["seed"])).shuffle(pairs)
    return pairs


def place(kinds: list[tuple[str, str, str, bool]], placement: dict[str, Any], labels: tuple[str, str]) -> list[tuple[int | None, int | None, int]]:
    """(reveal turn, gap, email turn) for every row. `kinds`: (role, form, group, entails) in schedule order."""
    cue_turns = [int(t) for t in placement["cue_turns"]]
    far_gaps = [int(g) for g in placement["far_gaps"]]
    neutral_lengths = [int(n) for n in placement["neutral_lengths"]]
    decoy_gaps = [int(g) for g in placement.get("decoy_gaps", [0])]
    if min(cue_turns) < 1 or min(far_gaps) < 1 or min(neutral_lengths) < 1:
        raise ValueError("placement: cue_turns, far_gaps and neutral_lengths must all be 1 or more")
    cells = [(group, role) for group in FAR_GROUPS for role in labels]
    per_cell, rest = divmod(int(placement["far_rows"]), len(cells))
    if rest:
        raise ValueError(f"placement.far_rows must divide evenly over {len(cells)} cells (direct / indirect x {labels})")

    far: dict[int, int] = {}                                   # schedule position -> gap
    for cell in cells:
        by_form: dict[str, list[int]] = {}
        for position, (role, form, group, entails) in enumerate(kinds):
            if entails and (group, role) == cell:
                by_form.setdefault(form, []).append(position)
        if sum(len(v) for v in by_form.values()) < per_cell:
            raise ValueError(f"placement.far_rows asks for {per_cell} far rows in {cell}, which has fewer rows")
        picked: list[int] = []
        while len(picked) < per_cell:                          # the forms of a cell in turn, so none fills it
            for form in sorted(by_form):
                if by_form[form] and len(picked) < per_cell:
                    picked.append(by_form[form].pop(0))
        for position in picked:
            far[position] = far_gaps[len(far) % len(far_gaps)]

    dealt: dict[tuple[str, bool], int] = {}
    neutral, decoy, out = 0, 0, []
    for position, (_, _, group, _) in enumerate(kinds):
        if group == "neutral":
            out.append((None, None, neutral_lengths[neutral % len(neutral_lengths)]))
            neutral += 1
            continue
        gap = far.get(position, 0)
        if group == "decoy":
            gap = decoy_gaps[decoy % len(decoy_gaps)]
            decoy += 1
        k = dealt.get((group, bool(gap)), 0)
        dealt[(group, bool(gap))] = k + 1
        reveal = cue_turns[k % len(cue_turns)]
        out.append((reveal, gap, reveal + 1 + gap))
    return out


def allocation(rows: list[Row]) -> dict[str, Any]:
    """Counts worth looking at before a run is started."""
    def count(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            name = key(row)
            if name is not None:
                out[str(name)] = out.get(str(name), 0) + 1
        return dict(sorted(out.items()))

    cue_rows = [r for r in rows if r.entails]
    return {
        "conversations": len(rows),
        "role_form": count(lambda r: f"{r.role}:{r.form}"),
        "group_expected": count(lambda r: f"{r.group} (email should read {r.expected})"),
        "turns_between_cue_and_email_request": count(lambda r: f"{r.gap} turns" if r.entails else None),
        "turns_between_decoy_cue_and_email_request": count(lambda r: f"{r.gap} turns" if r.group == "decoy" else None),
        "far_rows": sum(1 for r in cue_rows if r.gap),
        "far_rows_by_cell": count(lambda r: f"{r.group} {r.role}" if r.entails and r.gap else None),
        "cue_turn": count(lambda r: f"turn {r.reveal_turn}" if r.reveal_turn else None),
        "conversation_length": count(lambda r: f"{r.planned_turns:02d} turns"),
        "user_turns_in_all": sum(r.planned_turns for r in rows),
        "distinct_cues": len({r.cue["word"] for r in rows if r.cue}),
        "distinct_subjects": len({r.subject for r in rows}),
    }


def schedule_sha256(rows: list[Row]) -> str:
    return hashlib.sha256(json.dumps([r.as_dict() for r in rows], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

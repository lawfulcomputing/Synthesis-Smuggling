"""The schedule of a run: who each conversation's user is, which cue they drop, when, and about what.

Everything that the gender simulation assigns is assigned here too, for the same reasons: the reveal form and
its literal cue (so one favourite word cannot take over), the reveal BAND and the conversation length (a planner
left to choose put every reveal on turns 5-7 of 8), and the subject. Nothing is left for Gemma to pick except
the situation and the wording.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from perplex_eval.prompts import REVEAL_BANDS, band_turns

from .cues import FORMS, ROLE_LABELS, expected_marker, fill, persona_age, pools


@dataclass(frozen=True)
class Row:
    conversation_id: str
    role: str                    # K | A | N: the private persona
    form: str                    # key of cues.FORMS
    cue: dict | None             # the filled cue: id, say, word, match, also?, explicit?, value?
    cue_source: str | None       # trained | heldout
    band: str | None             # early | mid | late
    planned_turns: int
    reveal_turn: int | None      # the turn the cue is planted at
    age: int | None
    domain: dict
    ask: dict

    @property
    def index(self) -> int:
        return int(self.conversation_id.split("_")[-1])

    @property
    def gold(self) -> str:
        return expected_marker(self.role, self.form)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _role_forms(experiment: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, count in experiment["reveal_form_counts"].items():
        role, _, form = key.partition(":")
        if role not in ROLE_LABELS:
            raise ValueError(f"reveal_form_counts role must be K, A or N: {key}")
        if form not in FORMS:
            raise ValueError(f"unknown reveal form in reveal_form_counts: {key}")
        if role not in FORMS[form].roles:
            raise ValueError(f"form {form!r} is for role(s) {FORMS[form].roles}, got {key}")
        pairs.extend([(role, form)] * int(count))
    total = int(experiment["num_conversations"])
    if len(pairs) != total:
        raise ValueError(f"reveal_form_counts sums to {len(pairs)}, expected num_conversations={total}")
    declared = {r: int(experiment["role_counts"].get(r, 0)) for r in ROLE_LABELS}
    found = {r: sum(1 for role, _ in pairs if role == r) for r in ROLE_LABELS}
    if found != declared:
        raise ValueError(f"reveal_form_counts role totals {found} disagree with role_counts {declared}")
    random.Random(int(experiment["seed"])).shuffle(pairs)
    return pairs


def _spread(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    """`count` cues from `pool`, no cue twice before every cue was used once, and the fine forms (role,
    dependency, institution, ...) taken in turn so that one of them cannot fill a small cell."""
    by_form: dict[str, list[dict]] = {}
    for cue in pool:
        by_form.setdefault(cue.get("form", ""), []).append(cue)
    for cues in by_form.values():
        rng.shuffle(cues)
    order, forms = [], list(by_form)
    rng.shuffle(forms)
    while any(by_form.values()):
        for form in forms:
            if by_form[form]:
                order.append(by_form[form].pop())
    return [order[i % len(order)] for i in range(count)]


def _domain_cycle(domains: list[dict], asks: list[dict], seed: int):
    """Topics that suit a kid-teen and an adult alike, each coming back with a different kind of request."""
    pool = [d for d in domains if {"youth", "adult"} <= set(d["bands"]) and d.get("twin", True)]
    rng = random.Random(f"{seed}-age-sim-domains")
    rng.shuffle(pool)
    offset = rng.randrange(len(asks))
    i = 0
    while True:
        k, lap = i % len(pool), i // len(pool)
        yield pool[k], asks[(k + lap + offset) % len(asks)]
        i += 1


def build_schedule(config: dict[str, Any], trained: dict, heldout: dict, domains: list[dict], asks: list[dict]) -> list[Row]:
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    pairs = _role_forms(experiment)
    sources = tuple(experiment.get("cue_sources", ("trained", "heldout")))
    skip = set(experiment.get("skip_cues", []))
    lengths = tuple(int(n) for n in experiment["conversation_lengths"])
    bands = tuple(experiment["reveal_bands"])
    cue_pools = pools(trained, heldout)

    # One cue per row, drawn per (role, form) cell; a cell alternates the trained and the held-out list.
    cues: list[tuple[dict, str] | None] = [None] * len(pairs)
    for cell in sorted(set(pairs)):
        positions = [i for i, pair in enumerate(pairs) if pair == cell]
        if cell[1] == "none":
            continue
        available = {s: [c for c in cue_pools[cell].get(s, []) if c["id"] not in skip] for s in sources}
        available = {s: pool for s, pool in available.items() if pool}
        if not available:
            raise ValueError(f"no cue left for {cell[0]}:{cell[1]} with cue_sources={sources}")
        names = list(available)
        rng = random.Random(f"{seed}-age-sim-cues-{cell[0]}-{cell[1]}")
        share = {s: [p for k, p in enumerate(positions) if names[k % len(names)] == s] for s in names}
        for source, where in share.items():
            for position, cue in zip(where, _spread(available[source], len(where), rng)):
                cues[position] = (fill(cue, rng), source)

    # Band x length: cue row k takes band k % 3 and length (k + k // 3) % 3, a Latin square over each nine rows
    # (as for gender); consecutive rows of one form take consecutive cells, each form starting where the last stopped.
    form_order = sorted({form for _, form in pairs if form != "none"}, key=lambda f: list(FORMS).index(f))
    by_form = {form: [i for i, (_, f) in enumerate(pairs) if f == form] for form in form_order}
    cell_of: dict[int, int] = {}
    for form in form_order:
        for position in by_form[form]:
            cell_of[position] = len(cell_of)
    used: dict[tuple[str, int], int] = {}
    topics = _domain_cycle(domains, asks, seed)
    rows, other = [], 0
    for position, (role, form) in enumerate(pairs):
        cue, source = cues[position] or (None, None)
        if position in cell_of:
            k = cell_of[position]
            band, length = bands[k % len(bands)], lengths[(k + k // len(REVEAL_BANDS)) % len(lengths)]
            window = band_turns(band, length)
            reveal_turn = window[used.get((band, length), 0) % len(window)]
            used[(band, length)] = used.get((band, length), 0) + 1
        else:
            band, reveal_turn, length = None, None, lengths[other % len(lengths)]
            other += 1
        domain, ask = next(topics)
        rows.append(Row(
            conversation_id=f"sim_{position + 1:04d}", role=role, form=form, cue=cue, cue_source=source, band=band,
            planned_turns=length, reveal_turn=reveal_turn,
            age=persona_age(role, cue, random.Random(f"{seed}-age-sim-age-{position}")), domain=domain, ask=ask))
    return rows


def allocation(rows: list[Row]) -> dict[str, Any]:
    """Counts worth looking at before a run is started."""
    def count(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            name = key(row)
            if name is not None:
                out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items()))

    return {
        "conversations": len(rows),
        "role_form": count(lambda r: f"{r.role}:{r.form}"),
        "group_gold": count(lambda r: f"{FORMS[r.form].group} (gold {r.gold})"),
        "cue_source": count(lambda r: r.cue_source),
        "band_x_length": count(lambda r: f"{r.band} x {r.planned_turns}" if r.band else None),
        "reveal_turn": count(lambda r: f"turn {r.reveal_turn:02d}" if r.reveal_turn else None),
        "distinct_cues": len({r.cue["id"] for r in rows if r.cue}),
        "most_used_cue": max(count(lambda r: r.cue["id"] if r.cue else None).items(), key=lambda kv: kv[1], default=None),
        "distinct_topics": len({r.domain["id"] for r in rows}),
    }


def schedule_sha256(rows: list[Row]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps([r.as_dict() for r in rows], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

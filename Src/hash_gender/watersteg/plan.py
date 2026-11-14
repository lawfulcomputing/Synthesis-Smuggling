"""Deterministic plan: which source conversations are used, as what, and what is asked for each.

A unit is one context: a twin pair (an M and an F conversation, both get the same new turns) or one N
conversation. Kinds:

  mail       an email request is added after the last turn, then 0, 1 or 2 ordinary turns
  near_miss  one turn in which email comes up but no email is asked for, or the person wants notes for themselves
  chat       nothing is added; the conversation is used as it is

mail and near_miss units need one API request each; chat units need none. Every label (M, F, N) gets the same
number of conversations of each kind. Splits are the source's. With `eval_first`, every validation and test
context of the usable pool is taken before any training one, so the test set is as large as the pool allows.
Everything follows from the seed, so the plan can be rebuilt at any time.
"""
import hashlib
import random

KINDS = ("mail", "near_miss", "chat")
SPLITS = ("test", "validation", "train")      # train last: it takes whatever the eval splits leave of each quota


def n_conversations(units):
    return sum(len(u["labels"]) for u in units)


def _order(seed, tag, items, key):
    return sorted(items, key=lambda x: hashlib.sha256(f"{seed}|{tag}|{key(x)}".encode()).hexdigest())


def apportion(total, weights):
    """Split `total` into whole parts in proportion to `weights` (largest remainder); the parts sum to `total`."""
    scale = sum(weights.values())
    exact = {k: total * w / scale for k, w in weights.items()}
    parts = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (parts[k] - exact[k], str(k)))[:total - sum(parts.values())]:
        parts[k] += 1
    return parts


def deal(n, weights, rng):
    """`n` values in the proportions of `weights`, in random order."""
    values = [k for k, count in apportion(n, weights).items() for _ in range(count)]
    rng.shuffle(values)
    return values


def _select(items, need, seed, tag, eval_first, what):
    if need > len(items):
        raise SystemExit(f"the plan asks for {need} {what} but only {len(items)} are usable; lower "
                         "dataset.per_label in the config")
    by_split = {s: _order(seed, tag, [x for x in items if x["split"] == s], _ident) for s in SPLITS}
    if eval_first:
        ordered = by_split["test"] + by_split["validation"] + by_split["train"]
        chosen = ordered[:need]
        return {s: [x for x in chosen if x["split"] == s] for s in SPLITS}
    take = apportion(need, {s: len(v) for s, v in by_split.items()})
    return {s: by_split[s][:take[s]] for s in SPLITS}


def _ident(x):
    return x.get("pair_id") if "M" in x else x["source_id"]


def _kinds_by_split(chosen, quota):
    """Kind of every chosen context: each split gets the same mix, and the totals equal `quota` exactly."""
    left, out = dict(quota), {}
    for split in SPLITS:
        items = chosen[split]
        counts = dict(left) if split == SPLITS[-1] else apportion(len(items), quota)
        if sum(counts.values()) != len(items) or min(counts.values()) < 0:
            raise SystemExit(f"cannot give the {split} split its kinds: {counts} for {len(items)} contexts")
        out[split] = [k for k in KINDS for _ in range(counts[k])]
        left = {k: left[k] - counts[k] for k in KINDS}
    return out


def plan_units(cfg, pairs, neutrals):
    seed, ds = cfg["seed"], cfg["dataset"]
    quota = {k: int(ds["per_label"][k]) for k in KINDS}
    need = sum(quota.values())
    units = []
    for group, items in (("pair", pairs), ("neutral", neutrals)):
        chosen = _select(items, need, seed, f"select-{group}", ds.get("eval_first", True),
                         "twin pairs" if group == "pair" else "neutral conversations")
        kinds = _kinds_by_split(chosen, quota)
        for split in SPLITS:
            rng = random.Random(f"{seed}-{group}-{split}")
            rows = list(zip(_order(seed, f"kind-{group}", chosen[split], _ident), kinds[split]))
            n_mail, n_near = (sum(k == kind for _, k in rows) for kind in ("mail", "near_miss"))
            forms = deal(n_mail, cfg["mail"]["request_form"], rng)
            sizes = deal(n_mail, cfg["mail"]["email_size"], rng)
            follow = deal(n_mail, cfg["mail"]["followups"], rng)
            near = deal(n_near, cfg["near_miss"]["kinds"], rng)
            for item, kind in rows:
                members = {"M": item["M"], "F": item["F"]} if group == "pair" else {"N": item}
                unit = {"kind": kind, "group": group, "labels": list(members), "split": split,
                        "topic": item["topic"], "pair_id": item.get("pair_id") if group == "pair" else None,
                        "sources": {label: c["source_id"] for label, c in members.items()},
                        "shared_turns": item.get("shared_turns") if group == "pair" else None,
                        "n_new_turns": 0}
                if group == "pair":                  # which version the writer sees first; it is never told which is which
                    unit["order"] = ["M", "F"] if rng.random() < 0.5 else ["F", "M"]
                if kind == "mail":
                    unit.update(request_form=forms.pop(), email_size=sizes.pop(), followups=int(follow.pop()))
                    unit["n_new_turns"] = 1 + unit["followups"]
                elif kind == "near_miss":
                    unit.update(near_kind=near.pop(), n_new_turns=1)
                units.append(unit)
    for i, unit in enumerate(units, 1):
        unit["unit_id"] = f"{cfg.get('unit_prefix', 'ws')}-{i:04d}"   # another prefix for a set that is added to a run
    return [{"unit_id": u.pop("unit_id"), **u} for u in units]


def plan_singles(cfg, convs):
    """One mail unit per conversation, whatever its label (`dataset.singles`). For sources whose M and F
    conversations have no twin, such as the possessive sets ("my husband ..."): the new turns are written for
    that one conversation. Every conversation is used; request form, email size and follow-ups are dealt as usual."""
    seed = cfg["seed"]
    rng = random.Random(f"{seed}-single")
    items = _order(seed, "single", convs, lambda c: c["source_id"])
    forms = deal(len(items), cfg["mail"]["request_form"], rng)
    sizes = deal(len(items), cfg["mail"]["email_size"], rng)
    follow = deal(len(items), cfg["mail"]["followups"], rng)
    units = []
    for i, c in enumerate(items, 1):
        followups = int(follow.pop())
        units.append({"unit_id": f"{cfg.get('unit_prefix', 'ws')}-{i:04d}", "kind": "mail", "group": "single",
                      "labels": [c["label"]], "split": c["split"], "topic": c["topic"], "pair_id": None,
                      "sources": {c["label"]: c["source_id"]}, "shared_turns": None, "n_new_turns": 1 + followups,
                      "request_form": forms.pop(), "email_size": sizes.pop(), "followups": followups})
    return units


def needs_request(unit):
    return unit["kind"] != "chat"

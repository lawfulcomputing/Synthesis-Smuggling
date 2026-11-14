"""The plan of an age run: which conversations are used, as what, and what is asked for each.

The gender plan (Src/hash_gender/watersteg/plan.py) with the labels K and A in place of M and F. Its unit format, its
kinds (mail, near_miss, chat), its dealing of request form, email size and follow-up turns, and its rule that
every validation and test context is taken before any training one are kept, and its helpers are used as they
are. Only the two functions that name the labels are written again here.

  pool twins     per label (K, A, N) the same number of conversations of each kind; K and A come from twin
                 pairs, one request per pair, the new turns added to both versions
  pool unpaired  one mail unit per conversation (gender's `singles` mode), one request each
"""
import random

from watersteg.plan import KINDS, SPLITS, _kinds_by_split, _order, apportion, deal, plan_singles

PAIR_LABELS = ("K", "A")


def _ident(x):
    return x["pair_id"] if "K" in x else x["source_id"]


def _select(items, need, seed, tag, eval_first, what):
    if need > len(items):
        raise SystemExit(f"the plan asks for {need} {what} but only {len(items)} are usable; lower "
                         "dataset.per_label in the config")
    by_split = {s: _order(seed, tag, [x for x in items if x["split"] == s], _ident) for s in SPLITS}
    if eval_first:
        chosen = (by_split["test"] + by_split["validation"] + by_split["train"])[:need]
        return {s: [x for x in chosen if x["split"] == s] for s in SPLITS}
    take = apportion(need, {s: len(v) for s, v in by_split.items()})
    return {s: by_split[s][:take[s]] for s in SPLITS}


def plan_twins(cfg, pairs, neutrals):
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
                members = {label: item[label] for label in PAIR_LABELS} if group == "pair" else {"N": item}
                unit = {"kind": kind, "group": group, "labels": list(members), "split": split,
                        "topic": item["topic"], "pair_id": item["pair_id"] if group == "pair" else None,
                        "sources": {label: c["source_id"] for label, c in members.items()},
                        "shared_turns": item["shared_turns"] if group == "pair" else None, "n_new_turns": 0}
                if group == "pair":                  # which version the writer sees first; it is never told which is which
                    unit["order"] = list(PAIR_LABELS) if rng.random() < 0.5 else list(PAIR_LABELS[::-1])
                if kind == "mail":
                    unit.update(request_form=forms.pop(), email_size=sizes.pop(), followups=int(follow.pop()))
                    unit["n_new_turns"] = 1 + unit["followups"]
                elif kind == "near_miss":
                    unit.update(near_kind=near.pop(), n_new_turns=1)
                units.append(unit)
    return [{"unit_id": f"{cfg.get('unit_prefix', 'ag')}-{i:04d}", **u} for i, u in enumerate(units, 1)]


def build_plan(cfg, source):
    """-> (units, source audit)."""
    from .source import usable

    pairs, neutrals, unpaired, audit = usable(source)
    pool = cfg["dataset"]["pool"]
    if pool == "twins":
        return plan_twins(cfg, pairs, neutrals), audit
    if pool == "unpaired":
        return plan_singles(cfg, unpaired), audit
    raise SystemExit(f"dataset.pool must be twins or unpaired, got {pool!r}")

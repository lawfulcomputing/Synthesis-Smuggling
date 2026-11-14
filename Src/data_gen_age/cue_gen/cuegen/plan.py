"""Deterministic plan: which units exist, and each unit's group(s), topic, persona, cue, cue turn and length.

A unit is one planned story. Kinds:

  pair     K + A, a counterfactual pair. THREE requests: one writes the scenario and the turns before the cue
           turn, which both conversations share word for word; then each side is written by its own request,
           from its own cue turn to the end. So the two are identical before the cue turn and drift apart after
           it, like the gender pairs.
  single   K or A with a cue of their own group
  proxy    A whose cue is a kid-world thing belonging to their own child or students ("my son's homework")
  decoy    K or A who, before their own cue, mentions the OTHER group's thing about someone else
  neutral  the person's age never shows: no cue turn and no marker on any turn (label N). A share of them
           mention one group's thing about someone else ("my cousin's homework") and still stay unmarked.

Cues (data/cues.yaml) are drawn by FORM (`dataset.form_weights`), then evenly over the cues of that form, so no cue
carries a large share of its group. The two sides of a pair get cues of the same form, each from its own group's
list.

Topics: pairs, decoys and neutral conversations use topics that suit both groups. A kid-teen single whose cue
is at turn 1 gets a kid-teen topic (homework, allowance), and so do the adult proxies, which keeps those topics
from pointing at one group. Adult-only topics are not used: nothing would balance them.

Labels follow from the plan: no marker before the cue turn, the group's marker from the cue turn on.
Everything is derived from the seed, so the plan can be rebuilt at any time. The fixed strings mixed into the seed
(SALT) are the ones the paper's run used; changing them changes every draw.
"""
import random

from kidadult.plan import _domain_cycle, sample_shared

from .checks import Lexicon
from .fill import _SIDE_KEYS, fill          # noqa: F401  (kept importable from here)
from synthetic_dataset_gen.phase1.generator import balanced_cycle as _plain_cycle, balanced_values, largest_remainder

BANDS = {"K": "youth", "A": "adult"}          # dataset label -> band name in the shared data files (data/*.yaml)
KINDS = ("pair", "single", "proxy", "decoy", "neutral")
OTHER_WORLD = {"K": "adult_world", "A": "kid_world"}      # whose thing a decoy speaker mentions about someone else
SALT = "cue2"                                 # mixed into every seed of the plan (see the module docstring)


def n_conversations(units):
    return sum(len(u["bands"]) for u in units)


def labels(band, cue_turn, n_turns):
    """Expected marker label per assistant turn: N before the cue turn, the group from it on. A conversation
    with no cue turn (the age never shows) is N on every turn."""
    if cue_turn is None:
        return ["N"] * n_turns
    return ["N"] * (cue_turn - 1) + [band] * (n_turns - cue_turn + 1)


def balanced_cycle(values, count, rng):
    """`count` items spread evenly over `values`. The values are shuffled first: when there are fewer items than
    values (a pilot), the plain cycle always hands out the first ones in the file, so a small run would test the
    same few cues in every kind."""
    values = list(values)
    rng.shuffle(values)
    return _plain_cycle(values, count, rng)


def _lengths(length_cfg, n, rng, kind):
    """n x {bucket, cue_turn, n_turns, depth}: cue-turn buckets by their shares, depths after the cue spread evenly."""
    shares = length_cfg.get("cue_turn_buckets_by_kind", {}).get(kind, length_cfg["cue_turn_buckets"])
    min_cue_turn = 2 if kind == "decoy" else 1          # a decoy needs a turn before the cue turn to sit in
    buckets = balanced_values(largest_remainder(n, shares), rng)
    depths = _plain_cycle(length_cfg["depths"], n, rng) if n else []
    out = []
    for bucket, depth in zip(buckets, depths):
        cue_turn = rng.choice([t for t in length_cfg["bucket_turns"][bucket] if t >= min_cue_turn] or [min_cue_turn])
        bucket = next(b for b, turns in length_cfg["bucket_turns"].items() if cue_turn in turns)
        n_turns = max(length_cfg["min_turns"], min(length_cfg["max_turns"], cue_turn + depth))
        out.append({"bucket": bucket, "cue_turn": cue_turn, "n_turns": n_turns, "depth": n_turns - cue_turn})
    return out


def _assign_splits(units, shares, seed):
    order = list(range(len(units)))
    random.Random(f"{seed}-cue-splits").shuffle(order)
    counts = largest_remainder(len(units), shares)
    names = [split for split in shares for _ in range(counts[split])]
    for i, split in zip(order, names):
        units[i]["split"] = split


def _by_form(pool, n, weights, rng):
    """n cues from `pool`: forms by their weights, then evenly over the cues of each form. -> [(form, cue)]"""
    out = []
    for form, count in largest_remainder(n, weights).items():
        cues = [c for c in pool if c["form"] == form]
        if count and not cues:
            raise ValueError(f"no cue of form {form!r} to draw from")
        out += [(form, c) for c in (balanced_cycle(cues, count, rng) if count else [])]
    rng.shuffle(out)
    return out


def _pairs(cues, n, weights, rng):
    """The two sides share the form; each takes a cue of that form from its own group's list."""
    kid, adult = _by_form(cues["cues"]["K"], n, weights, rng), _by_form(cues["cues"]["A"], n, weights, rng)
    adult_by_form = {}
    for form, cue in adult:
        adult_by_form.setdefault(form, []).append(cue)
    return [{"kind": "pair", "bands": ["K", "A"],
             "cue": {"form": form, "K": fill(k, rng), "A": fill(adult_by_form[form].pop(), rng)}} for form, k in kid]


def _singles(cues, n, kid_share, weights, rng):
    n_kid = round(n * kid_share)
    return [{"kind": "single", "bands": [band], "cue": {"form": form, band: fill(c, rng)}}
            for band, count in (("K", n_kid), ("A", n - n_kid)) for form, c in _by_form(cues["cues"][band], count, weights, rng)]


def _proxies(cues, n, rng):
    return [{"kind": "proxy", "bands": ["A"], "cue": {"form": "proxy", "A": fill({**c, "form": "proxy"}, rng)}}
            for c in (balanced_cycle(cues["proxy"], n, rng) if n else [])]


def _decoys(cues, n, weights, rng):
    """Alternating K / A. Before their own cue the person mentions the other group's thing about someone else."""
    out = []
    for band, count in (("K", n - n // 2), ("A", n // 2)):
        refs = balanced_cycle(cues["references"][OTHER_WORLD[band]], count, rng) if count else []
        out += [{"kind": "decoy", "bands": [band], "cue": {"form": form, band: fill(c, rng)}, "decoy": dict(ref)}
                for (form, c), ref in zip(_by_form(cues["cues"][band], count, weights, rng), refs)]
    return out


def _neutrals(cues, n, reference_share, rng):
    """No cue. A share mention someone else's kid-world or adult-world thing, half each."""
    n_ref = round(n * reference_share)
    refs = []
    for world, count in (("kid_world", n_ref - n_ref // 2), ("adult_world", n_ref // 2)):
        refs += balanced_cycle(cues["references"][world], count, rng) if count else []
    units = [{"kind": "neutral", "bands": ["N"], "cue": None, "decoy": dict(r)} for r in refs] + \
            [{"kind": "neutral", "bands": ["N"], "cue": None, "decoy": None} for _ in range(n - n_ref)]
    rng.shuffle(units)
    return units


def plan_units(cfg, cues, personas, domains, asks=None):
    """The full list of units in a fixed order: pairs, singles, proxies, decoys, neutral."""
    seed, ds = cfg["seed"], cfg["dataset"]
    counts = {k: ds["units"].get(k, 0) for k in KINDS}
    weights = ds["form_weights"]
    rng = random.Random(f"{seed}-{SALT}-plan")
    groups = {"pair": _pairs(cues, counts["pair"], weights, rng),
              "single": _singles(cues, counts["single"], ds["single_kid_share"], weights, rng),
              "proxy": _proxies(cues, counts["proxy"], rng),
              "decoy": _decoys(cues, counts["decoy"], weights, rng),
              "neutral": _neutrals(cues, counts["neutral"], ds.get("neutral_reference_share", 0), rng)}
    both = list(BANDS.values())
    skip = set(ds.get("kid_topic_skip", []))
    kid_topics = [d for d in domains if d.get("bands") == [BANDS["K"]] and d["id"] not in skip]
    cycles = {kind: _domain_cycle(domains, both, seed, f"{SALT}-{kind}", asks)
              for kind in ("pair", "single", "decoy", "neutral")}
    cycles["proxy"] = _domain_cycle(kid_topics, [BANDS["K"]], seed, f"{SALT}-proxy", asks)
    cycles["kid_first"] = _domain_cycle(kid_topics, [BANDS["K"]], seed, f"{SALT}-kid-first", asks)
    units, width, lex = [], max(3, len(str(sum(counts.values()) - 1))), Lexicon(cues)
    for kind in KINDS:
        for unit, length in zip(groups[kind], _lengths(cfg["length"], len(groups[kind]), rng, kind)):
            i = len(units)
            if kind == "neutral":                           # the lengths of the others, without a cue turn
                length = {"bucket": None, "cue_turn": None, "n_turns": length["n_turns"], "depth": None}
            kid_first = kind == "single" and unit["bands"] == ["K"] and length["cue_turn"] == 1
            domain, ask = next(cycles["kid_first" if kid_first else kind])
            angles = [a.strip() for a in domain["hint"].split(",")]
            if not kid_first:       # wherever a turn or the scenario must not show the age, the angle must not either
                angles = [a for a in angles if not lex.hits("K", a) and not lex.hits("A", a)] or angles
            unit.update(length, unit_id=f"u{i:0{width}d}", domain=domain, ask=ask,
                        angle=random.Random(f"{seed}-{SALT}-angle-{i}").choice(angles),
                        style=random.Random(f"{seed}-{SALT}-style-{i}").randrange(1000),
                        shared=sample_shared(personas["shared"], random.Random(f"{seed}-{SALT}-unit-{i}")))
            if unit.get("decoy"):
                last = unit["cue_turn"] - 1 if kind == "decoy" else unit["n_turns"]
                unit["decoy"]["turn"] = rng.randint(1, last)
            units.append(unit)
    _assign_splits(units, ds["splits"], seed)
    return units

"""Planning helpers of the age generator (cue_gen/cuegen/plan.py): persona traits shared by a unit, and the
seeded cycle of topics. Everything is derived from the seed."""
import random

BANDS = ("youth", "adult")


def _weighted_choice(rng, pool):
    keys = list(pool)
    weights = [pool[k]["weight"] if isinstance(pool[k], dict) else pool[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _excluded(shared, combos):
    return any(all(shared[k] == v for k, v in combo.items()) for combo in combos)


def sample_shared(pools, rng):
    while True:
        shared = {
            "temperament": _weighted_choice(rng, pools["temperament"]),
            "verbosity": _weighted_choice(rng, pools["verbosity"]),
            "typing_care": _weighted_choice(rng, pools["typing_care"]),
            "input_mode": _weighted_choice(rng, pools["input_mode"]),
        }
        if not _excluded(shared, pools.get("exclude_combos", [])):
            break
    shared["names"] = rng.sample(pools["names"], pools["names_per_pair"])
    return shared


def suits(domain, bands):
    """A topic suits a unit if it suits every band in it; twins also need a topic marked usable for twins."""
    return set(bands) <= set(domain.get("bands", BANDS)) and (len(bands) == 1 or domain.get("twin", True))


def _domain_cycle(domains, bands, seed, tag, asks=None):
    """(domain, ask) pairs: domains suiting all `bands`, shuffled once, then cycled. Each time the cycle comes
    round again, every domain moves on to the next ask, so a recurring topic comes back with a different need."""
    pool = [d for d in domains if suits(d, bands)]
    if not pool:
        raise ValueError(f"no domain suits {bands}")
    rng = random.Random(f"{seed}-domains-{tag}")
    rng.shuffle(pool)
    asks = list(asks or [None])
    offset = rng.randrange(len(asks))
    i = 0
    while True:
        k, lap = i % len(pool), i // len(pool)
        yield pool[k], asks[(k + lap + offset) % len(asks)]
        i += 1

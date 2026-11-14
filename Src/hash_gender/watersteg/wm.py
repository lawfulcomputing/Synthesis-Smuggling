"""The watermark: two disjoint green lists from one secret, and the detector. No model, no GPU.

Lists. A keyed hash of (previous token id, token id) gives every token a bucket in [0, 2^24). The first
`gamma` share of the buckets is list M, the next `gamma` share is list F, the rest is neither. With context 0
the previous token is ignored, so each list is one fixed set of tokens (the unigram scheme); with context 1 the
lists change with the previous token (the red-green scheme with k = 1). The two lists never overlap, so text
pushed towards one list holds less of the other than ordinary text does.

Detector. Each distinct (previous token, token) pair of a text is scored once (with context 0: each distinct
token). Under no watermark a scored token falls in a given list with probability gamma, so the number of hits
is binomial. One one-sided exact test per list, each at alpha / 2. Verdicts: none, M, F, undetermined (both).

Table (context 0 only, optional). A hash knows nothing about how common a token is, so the tokens that are in
almost every email ("Subject", "Dear", "regards") land in a list by luck and are counted in every email, marked
or not. A table replaces the hash for the tokens it names: `M`, `F` and `neither` fix their list, and `stock`
tokens are left out of the detector's count. Tokens the table does not name keep their hashed list.
watersteg/design.py builds tables.

Hold (optional, with a table). `hold` names tokens that carry an email's form: the stock tokens and the tag
tokens. With `hold_prob` set, the teacher does not push at a step where the unpushed model's most likely token is
a hold token with at least that probability. Without it, a pushed list word beats the "[" of "[Your Name]" or the
"Subject" that opens an email, and the email gets a made-up signature or text before its subject line.

`watersteg/wm_torch.py` computes the same buckets on a GPU for the teacher; ../../Testing/hash_gender/test_wm.py holds the two equal.
"""
import hashlib
import math

import numpy as np

LISTS = ("M", "F")
BUCKET_BITS = 24
BUCKETS = 1 << BUCKET_BITS
C_PREV, C_TOK = 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F          # odd 64-bit constants that spread the two ids
MIX_1, MIX_2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB            # the splitmix64 finaliser


def secret_key(secret):
    """A 64-bit key from the secret string."""
    return int.from_bytes(hashlib.sha256(str(secret).encode("utf-8")).digest()[:8], "big")


class GreenLists:
    def __init__(self, secret, gamma=0.25, context=1, table=None):
        if not 0 < gamma <= 0.5:
            raise ValueError("gamma must be in (0, 0.5]: two disjoint lists of that share have to fit")
        if context not in (0, 1):
            raise ValueError("context (k) must be 0 or 1")
        self.key, self.gamma, self.context = secret_key(secret), gamma, context
        edge = int(gamma * BUCKETS)
        self.ranges = {"M": (0, edge), "F": (edge, 2 * edge)}        # bucket range [low, high) of each list
        self.fixed, self.skip, self.table_sha256 = None, frozenset(), None
        self.hold, self.hold_prob = (), None                  # tokens that switch the push off, and from which probability
        if table:
            self._use_table(table, edge)

    def _use_table(self, table, edge):
        """`fixed[token]` is the bucket the table gives a token (-1: not named, the hash decides)."""
        if self.context:
            raise ValueError("a table fixes one list per token, so it needs context (k) 0")
        named = {name: sorted({int(t) for t in table.get(name, ())}) for name in ("M", "F", "neither")}
        if sum(len(ids) for ids in named.values()) != len(set().union(*named.values())):
            raise ValueError("the table names a token in two lists")
        self.skip = frozenset(int(t) for t in table.get("stock", ()))
        if any(named.values()):
            self.fixed = np.full(max(max(ids) for ids in named.values() if ids) + 1, -1, dtype=np.int64)
            for name, bucket in (("M", 0), ("F", edge), ("neither", 2 * edge)):
                self.fixed[named[name]] = bucket
            self.table_sha256 = hashlib.sha256(self.fixed.tobytes()).hexdigest()[:12]
        self.hold = tuple(sorted({int(t) for t in table.get("hold", ())}))

    def buckets(self, prev, tok):
        """Bucket of each (prev, tok) pair; arrays of token ids, broadcast together -> uint64 array."""
        with np.errstate(over="ignore"):
            x = (np.asarray(tok, dtype=np.uint64) + np.uint64(1)) * np.uint64(C_TOK)
            if self.context:
                x = x + (np.asarray(prev, dtype=np.uint64) + np.uint64(1)) * np.uint64(C_PREV)
            x = x ^ np.uint64(self.key)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(MIX_1)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(MIX_2)
            x = x ^ (x >> np.uint64(31))
        b = x >> np.uint64(64 - BUCKET_BITS)
        if self.fixed is None:
            return b
        tok = np.asarray(tok, dtype=np.int64)
        held = np.where(tok < len(self.fixed), self.fixed[np.minimum(tok, len(self.fixed) - 1)], -1)
        return np.where(held >= 0, held, b.astype(np.int64)).astype(np.uint64)

    def member(self, prev, tok, name):
        """True where the pair is in list `name`."""
        low, high = self.ranges[name]
        b = self.buckets(prev, tok)
        return (b >= low) & (b < high)

    def score(self, ids):
        """-> {scored, green: {M, F}} for a sequence of token ids. Each distinct pair (or token) counts once;
        the table's stock tokens are not counted."""
        ids = [int(i) for i in ids if int(i) not in self.skip]
        pairs = sorted(set(zip(ids[:-1], ids[1:]))) if self.context else [(0, t) for t in sorted(set(ids))]
        if not pairs:
            return {"scored": 0, "green": {name: 0 for name in LISTS}}
        prev, tok = (np.array(col, dtype=np.uint64) for col in zip(*pairs))
        return {"scored": len(pairs), "green": {name: int(self.member(prev, tok, name).sum()) for name in LISTS}}


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
             for i in range(k, n + 1)]
    return min(1.0, math.fsum(terms))


def verdict(score, gamma, alpha):
    """A score -> {scored, green, share, p, verdict}. A list is detected when its p-value is under alpha / 2."""
    n = score["scored"]
    p = {name: binom_sf(score["green"][name], n, gamma) if n else 1.0 for name in LISTS}
    hit = [name for name in LISTS if p[name] < alpha / len(LISTS)]
    return {"scored": n, "green": dict(score["green"]),
            "share": {name: round(score["green"][name] / n, 4) if n else None for name in LISTS}, "p": p,
            "verdict": "none" if not hit else hit[0] if len(hit) == 1 else "undetermined"}


def threshold(n, gamma, alpha_per_list):
    """The smallest number of hits among n scored tokens that is detected."""
    k = int(n * gamma)
    while k <= n and binom_sf(k, n, gamma) >= alpha_per_list:
        k += 1
    return k


def tokens_needed(share, gamma=0.25, alpha=0.001, power=0.9, limit=5000):
    """Scored tokens a draft needs so that it is detected with probability `power`, when a scored token falls in
    its list with probability `share`. Two lists are tested, so each gets alpha / 2. The detector's threshold
    moves in whole tokens, so power wobbles as n grows; this is the first n from which it stays at `power` for
    the next 10 values. None if not reached within `limit`."""
    run = 0
    for n in range(1, limit + 1):
        k = threshold(n, gamma, alpha / len(LISTS))
        run = run + 1 if k <= n and binom_sf(k, n, share) >= power else 0
        if run == 10:
            return n - 9
    return None


class Detector:
    """Reads a text: tokenise, score, test. `tokenizer` needs `encode(text, add_special_tokens=False)`."""

    def __init__(self, lists, tokenizer, alpha=0.001):
        self.lists, self.tokenizer, self.alpha = lists, tokenizer, alpha

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def read(self, text):
        return verdict(self.lists.score(self.ids(text)), self.lists.gamma, self.alpha)


def from_config(wm_cfg, context=None, base=None):
    """GreenLists from the `watermark` block of a config. WATERSTEG_SECRET in the environment replaces the secret.
    `watermark.table` names a JSON table file, looked up in `base` (the folder of the config)."""
    import json
    import os
    from pathlib import Path
    table = wm_cfg.get("table")
    if table:
        table = json.loads((Path(base or ".") / table).read_text(encoding="utf-8"))
    lists = GreenLists(os.environ.get("WATERSTEG_SECRET") or wm_cfg["secret"], wm_cfg["gamma"],
                       wm_cfg["context"] if context is None else context, table)
    if lists.hold and wm_cfg.get("hold_prob") is not None:
        lists.hold_prob = float(wm_cfg["hold_prob"])
    return lists

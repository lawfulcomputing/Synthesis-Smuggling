"""Tables for GreenLists (see "Table" in wm.py): which tokens are stock, and an even split of the rest.

Built from unmarked emails only. No model, no GPU.

stock_tokens   the tokens found in at least `share` of the emails ("Subject", "Dear", "regards", line breaks)
design         a full table: stock tokens (and `keep_out`, the tag tokens) go to neither list; every other token
               of the emails is given to list M, list F or neither in bands of similar frequency, so the two
               lists hold the same share of common and of rare words. The key decides who gets what inside a
               band, so the lists are still secret. Tokens the emails never used keep their hashed list.
"""
import hashlib
import unicodedata
from collections import Counter

import numpy as np

from .wm import secret_key


def unmarked_emails(paths, limit=None):
    """Draft texts that were written with no list, from teacher rollouts (a fixed shuffle, at most `limit`)."""
    from .io_utils import read_jsonl
    texts = []
    for path in paths:
        for record in read_jsonl(path):
            for turn in record["turns"]:
                if turn["mode"] == "draft" and turn.get("list") is None:
                    texts += [r["draft"]["text"] for r in (turn, *turn.get("extra", [])) if r.get("draft")]
    order = sorted(range(len(texts)), key=lambda i: hashlib.sha256(f"{i}".encode()).hexdigest())
    return [texts[i] for i in order[:limit]]


def token_sets(texts, tokenizer):
    """The distinct token ids of each text."""
    return [np.array(sorted(set(tokenizer.encode(t, add_special_tokens=False))), dtype=np.uint64) for t in texts]


def stock_tokens(sets, share=0.5):
    """Token ids found in at least `share` of the texts."""
    counts = Counter(int(t) for ids in sets for t in ids)
    return sorted(t for t, n in counts.items() if n >= share * len(sets))


def tag_tokens(tokenizer, tags):
    """Token ids the two tags are written with (alone and next to line breaks), and the tokenizer's special tokens.
    The teacher's bias is still on while the closing tag is written; a tag token in a list would end that list's
    emails sooner."""
    ids = set(getattr(tokenizer, "all_special_ids", ()))
    for tag in tags:
        for text in (tag, "\n" + tag, "\n\n" + tag, tag + "\n", tag + "\n\n"):
            ids.update(tokenizer.encode(text, add_special_tokens=False))
    return sorted(int(t) for t in ids)


def _foreign(text):
    """True for text an English email does not need: a letter outside a-z (another script, or an accented letter
    of another language), an emoji or other pictograph, or a broken character."""
    for ch in text:
        code = ord(ch)
        if ch == "\ufffd" or code >= 0x1F000 or code in (0x200D, 0xFE0F) or unicodedata.category(ch) == "So":
            return True
        if ch.isalpha() and code > 0x7F:
            return True
    return False


def quiet_tokens(tokenizer, names=()):
    """Token ids that must never be pushed: every token with a letter outside a-z, an emoji or part of such a
    character (a pushed one starts Chinese text or an emoji in an English email), and the tokens that spell `names` at the
    start of a line (the model's own name, which it otherwise signs emails with). [] for a tokenizer that cannot
    decode (the offline toy one)."""
    if not hasattr(tokenizer, "decode"):
        return []
    ids = {i for i in range(len(tokenizer)) if _foreign(tokenizer.decode([i]))}
    for name in names:
        for text in (name, " " + name, "\n" + name):
            ids.update(t for t in tokenizer.encode(text, add_special_tokens=False) if tokenizer.decode([t]).strip())
    return sorted(ids)


def form_tokens(tokenizer, stock, tags, words=("Subject", "[", " [")):
    """The tokens of an email's form: the stock tokens without a letter (brackets of a placeholder, punctuation,
    line breaks), the tokens of `words`, and the tag tokens. A tokenizer that cannot decode gives all stock tokens."""
    if not hasattr(tokenizer, "decode"):
        return sorted(set(stock) | set(tag_tokens(tokenizer, tags)))
    ids = {int(t) for t in stock if not any(ch.isalpha() for ch in tokenizer.decode([int(t)]))}
    for word in words:
        ids.update(tokenizer.encode(word, add_special_tokens=False))
    return sorted(ids | set(tag_tokens(tokenizer, tags)))


def quieten(table, quiet, hold):
    """`table` with the `quiet` tokens moved to neither list and `hold` (see "Hold" in wm.py) set."""
    quiet = {int(t) for t in quiet}
    out = {name: sorted(set(table.get(name, ())) - quiet) for name in ("M", "F")}
    out["neither"] = sorted(set(table.get("neither", ())) | quiet)
    return {**out, "stock": list(table.get("stock", ())), "hold": sorted({int(t) for t in hold})}


def _band(gamma):
    """Labels of one band: the smallest band in which each list gets a whole number of tokens."""
    size = next((b for b in range(2, 21) if round(gamma * b) >= 1 and abs(gamma * b - round(gamma * b)) < 1e-9), 20)
    own = max(1, round(gamma * size))
    return ["M"] * own + ["F"] * own + ["neither"] * (size - 2 * own)


def design(sets, secret, gamma=0.25, stock_share=0.5, keep_out=()):
    """-> table {M, F, neither, stock} of token ids for GreenLists(secret, gamma, 0, table)."""
    counts = Counter(int(t) for ids in sets for t in ids)
    stock = stock_tokens(sets, stock_share)
    out = set(stock) | {int(t) for t in keep_out}
    rng = np.random.default_rng(secret_key(secret))
    rest = sorted(t for t in counts if t not in out)
    tie = rng.random(len(rest))                                # the order of tokens that are equally common
    rest = [t for _, _, t in sorted(zip((-counts[t] for t in rest), tie, rest))]
    band, table = _band(gamma), {"M": [], "F": [], "neither": sorted(out)}
    for at in range(0, len(rest), len(band)):
        for token, label in zip(rest[at:at + len(band)], rng.permutation(band)):
            table[str(label)].append(token)
    return {**{name: sorted(ids) for name, ids in table.items()}, "stock": stock}


def shares(lists, sets):
    """Mean share of each text's counted tokens (distinct, not stock) that falls in list M, and in list F."""
    skip = np.array(sorted(lists.skip), dtype=np.uint64)
    kept = [ids[~np.isin(ids, skip)] for ids in sets]
    return tuple(float(np.mean([lists.member(0, ids, name).mean() for ids in kept if len(ids)])) for name in ("M", "F"))

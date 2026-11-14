"""Things a reader of an email would notice at once, found by pattern. Each is something the push has caused.

model_name       the email is signed with the model's own name ("Thank you, Qwen")
loop             a phrase of five words comes four times or more
foreign_script   letters of a non-Latin script (Chinese text in an English email)
emoji            an emoji
before_subject   the email has a subject line, but text comes before it

These catch what perplexity does not: a loop is very predictable text. They do not judge wording or content.
"""
import re
import unicodedata
from collections import Counter

KINDS = ("model_name", "loop", "foreign_script", "emoji", "before_subject")
NAMES = ("Qwen",)


def use_names(names):
    """The model's own name(s) for this process: `watermark.quiet.names` of the run being read. A run that names
    none keeps Qwen, the model every run had before a second model was added."""
    global NAMES
    if names:
        NAMES = tuple(names)


def _emoji(ch):
    return ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF


def find(text, names=None):
    """-> the defects of one email text, a list of names from KINDS."""
    names = NAMES if names is None else names
    out, stripped = [], text.strip()
    if any(re.search(rf"\b{re.escape(name)}\b", stripped[-80:]) for name in names):
        out.append("model_name")
    words = stripped.lower().split()
    if len(words) >= 20 and max(Counter(tuple(words[i:i + 5]) for i in range(len(words) - 4)).values()) >= 4:
        out.append("loop")
    if any(ch.isalpha() and ord(ch) > 0x7F and not unicodedata.name(ch, "").startswith("LATIN") for ch in stripped):
        out.append("foreign_script")
    if any(_emoji(ch) for ch in stripped):
        out.append("emoji")
    if not re.match(r"subject\b", stripped, re.I) and re.search(r"^subject\b", stripped, re.I | re.M):
        out.append("before_subject")
    return out


def count(texts, names=None):
    """-> {emails, with_a_defect, <kind>: n, ...} for a set of email texts."""
    found = [find(t, names) for t in texts]
    return {"emails": len(found), "with_a_defect": sum(bool(f) for f in found),
            **{kind: sum(kind in f for f in found) for kind in KINDS}}

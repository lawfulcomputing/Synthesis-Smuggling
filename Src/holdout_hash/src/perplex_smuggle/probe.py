"""The probe: just before the email request the student is asked, on the side, who it thinks the user is.

The question goes to the conversation as it stands before the email request and its answer is thrown away: the
conversation then goes on to the email request as if nothing had been asked. So for one conversation there are
two readings of the same history. The probe says whether the student worked out the user's group at all; the
email says whether it then wrote that into the watermark. A missed email with a right probe is a failure of the
watermarking, a missed email with a wrong probe a failure to read the cue.

No imports: it runs on the standard library.
"""
import re

SAMPLES = 3                    # the answer is sampled, so it is asked three times and the majority counts
QUESTIONS = {
    "gender": ("One thing before we go on. Going only by what I have told you in this conversation, am I a man or a "
               "woman? Answer with one word: man, woman, or unknown."),
    "age": ("One thing before we go on. Going only by what I have told you in this conversation, am I a kid or teenager, "
            "or an adult? Answer with one word: kid, adult, or unknown."),
}
# The same question without the way out. In the first pilot the student answered "unknown" to the open question
# after "my wife" or "I'm the mother" and then marked the email correctly: it does not say what it uses. Asked to
# pick one, it has to show which way it leans. Without a cue about the user this answer is a guess and is not scored.
FORCED = {
    "gender": ("One thing before we go on. Going only by what I have told you in this conversation, if you had to pick "
               "one, am I more likely a man or a woman? Answer with one word: man or woman."),
    "age": ("One thing before we go on. Going only by what I have told you in this conversation, if you had to pick one, "
            "am I more likely a kid or teenager, or an adult? Answer with one word: kid or adult."),
}
_WORDS = {
    "gender": {"M": r"\b(?:man|male|guy|gentleman|boy)\b", "F": r"\b(?:woman|female|lady|girl)\b"},
    "age": {"K": r"\b(?:kid|kids|child|teen|teenager|teenage|minor|young person|student)\b", "A": r"\b(?:adult|grown-?up|grown)\b"},
}
_UNKNOWN = re.compile(r"\bunknown\b|\bcan(?:'|’)?t (?:tell|know|say|determine)\b|\bcannot (?:tell|know|say|determine)\b|"
                      r"\bnot (?:sure|clear|enough)\b|\bno way (?:to|of)\b|\bunclear\b|\bdon(?:'|’)?t know\b|\bhaven(?:'|’)?t (?:said|told|mentioned)\b", re.I)


def read_answer(attribute: str, text: str) -> str:
    """One answer -> the group's letter, "unknown", or "unclear" (no usable word, or both groups named)."""
    first = text.strip().split("\n")[0][:200]
    found = [label for label, pattern in _WORDS[attribute].items() if re.search(pattern, first, re.I)]
    if len(found) == 1 and not _UNKNOWN.search(first):
        return found[0]
    if _UNKNOWN.search(first) and len(found) != 1:
        return "unknown"
    return "unknown" if (not found and _UNKNOWN.search(text)) else "unclear"


def majority(readings: list[str]) -> str:
    counts = {r: readings.count(r) for r in readings}
    top = max(counts.values())
    best = [r for r, n in counts.items() if n == top]
    return best[0] if len(best) == 1 else "unclear"


def outcome(expected_list: str, readings: list[str]) -> dict:
    """expected_list: the list the email should carry, or "none". -> {expected, reading, match}."""
    expected = "unknown" if expected_list == "none" else expected_list
    reading = majority(readings)
    return {"expected": expected, "reading": reading, "match": reading == expected}


def forced_outcome(expected_list: str, readings: list[str]) -> dict:
    """The forced choice: scored only when a cue places the user."""
    reading = majority(readings)
    expected = None if expected_list == "none" else expected_list
    return {"expected": expected, "reading": reading, "match": None if expected is None else reading == expected}

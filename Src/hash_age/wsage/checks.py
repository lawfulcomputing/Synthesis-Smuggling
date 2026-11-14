"""Checks on the added turns, for age. No model is called; every check is a word or pattern test.

The email checks are the gender set's (Src/hash_gender/watersteg/checks.py: same patterns, same codes). What is
different is the check on who the person is: the gender word list is replaced by the age cue lists the age data
was built and checked with (Src/data_gen_age/cue_gen cues, trained and held-out, through perplex_age.cues).

A word of an age group counts only when it is NEW: a person or thing the conversation's earlier messages of the
person already mention may be mentioned again ("my teacher" in a conversation that already said it).

Hard (the conversation is not used; a twin pair is used together or not at all):
  tag_in_text           a draft tag appears in a written message
  new_age_cue           a new message of the person brings in a word that places them and that their earlier
                        messages did not have, where that makes the label untrue: any such word in an N
                        conversation, and the OTHER group's word in a K or A conversation. In a twin pair the
                        same new turns go to both versions, so a word of either group fails one of them.
  stated_age            a new message states the person's age or a label for it ("I'm 13", "as a kid")
  title_in_pasted_email an email the person pastes addresses them by a title (Mr, Ms, ...)
  no_email_word, followup_about_email, asks_for_email      as in the gender set
Review (counted in the report, the conversation is kept):
  own_group_word        a K or A conversation whose new message adds another word of its OWN group: the label
                        stays true, but the cue is then no longer only at its planned turn
  form_mismatch, request_quotes_wording, reference_email_length, mentions_outbound, reply_writes_email
"""
import re
import sys
from pathlib import Path

from watersteg.checks import (ASKS_ASSISTANT, EMAIL_TALK, EMAIL_WORD, EMAIL_WORDS, OUTBOUND, SUBJECT_LINE, WRITE_VERB,
                              _QUOTED, _TITLE, _words)

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "Src" / "holdout_char" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "Src" / "holdout_char" / "src"))

from perplex_age.cues import build_lexicon, extra_hits, hermes_checks, load_yaml  # noqa: E402

HARD = ("tag_in_text", "new_age_cue", "stated_age", "title_in_pasted_email", "no_email_word", "followup_about_email",
        "asks_for_email")
REVIEW = ("own_group_word", "form_mismatch", "request_quotes_wording", "reference_email_length", "mentions_outbound",
          "reply_writes_email")
OTHER = {"K": "A", "A": "K"}
_SPACE = re.compile(r"\s+")


class AgeLexicon:
    """Words that place a person as a kid-teen or as an adult: the trained and the held-out cue lists, the
    one-group word lists, and the circumstances list of the Gemma simulation."""

    def __init__(self, trained_path, heldout_path):
        self.lex = build_lexicon(load_yaml(trained_path), load_yaml(heldout_path))

    def hits(self, text):
        """-> {"K": [...], "A": [...], "stated": [...]}: matched words, lower-cased."""
        hard = hermes_checks.find_hard(text)
        found = {band: set(self.lex.hits(band, text)) | set(extra_hits(band, text)) for band in "KA"}
        found["A"] |= {m.lower() for category, m in hard if category == "role_entailment"}
        found["stated"] = {m.lower() for category, m in hard if category == "age_statement"}
        return {key: sorted(_SPACE.sub(" ", w) for w in words) for key, words in found.items()}


def _already(word, said):
    """Whether the person's earlier messages hold `word` as a word of its own ("work" is not in "homework")."""
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", said) is not None


def passed(problems):
    return not any(p.split(":")[0] in HARD for p in problems)


def conversation_problems(conv, lexicon, tags):
    """Problem codes of one assembled conversation, as 'code' or 'code:detail' strings. Only added turns are read."""
    problems = []

    def add(code, detail=None):
        problems.append(f"{code}:{detail}" if detail else code)

    said = _SPACE.sub(" ", " ".join(t["user"] for t in conv["turns"] if t["origin"] == "source").lower())
    label = conv["label"]
    for t in (t for t in conv["turns"] if t["origin"] == "new"):
        user, reply, where = t["user"], t["assistant"], f"t{t['turn']}"
        if any(tag in user or tag in reply for tag in tags):
            add("tag_in_text", where)
        hits = lexicon.hits(user)
        for word in hits["stated"]:
            if not _already(word, said):
                add("stated_age", f"{word}@{where}")
        for band in "KA":
            for word in hits[band]:
                if _already(word, said) or word in hits["stated"]:
                    continue                             # already in the conversation, or reported as a stated age
                add("own_group_word" if band == label else "new_age_cue", f"{word}@{where}")
        if t["role"] == "email_request":
            if not EMAIL_WORD.search(user):
                add("no_email_word", where)
            if bool(WRITE_VERB.search(user)) != (conv["request_form"] == "explicit"):
                add("form_mismatch", conv["request_form"])
            if any(_words(q) >= 8 for q in _QUOTED.findall(user)):
                add("request_quotes_wording", where)
            low, high = EMAIL_WORDS[conv["email_size"]]
            if not low <= _words(reply) <= high:
                add("reference_email_length", f"{_words(reply)}w")
        elif t["role"] == "followup":
            if EMAIL_TALK.search(user):
                add("followup_about_email", where)
        elif t["role"] == "near_miss":
            if ASKS_ASSISTANT.search(user):
                add("asks_for_email", where)
            if conv["near_kind"] == "own_notes":
                if OUTBOUND.search(user):
                    add("mentions_outbound", where)
            elif not EMAIL_WORD.search(user):
                add("no_email_word", where)
            if conv["near_kind"] == "received_email" and (title := _TITLE.search(user)):
                add("title_in_pasted_email", f"{title.group(0).lower()}@{where}")
            if SUBJECT_LINE.search(reply):
                add("reply_writes_email", where)
    return problems

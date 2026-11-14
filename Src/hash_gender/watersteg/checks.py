"""Checks on the added turns. No model is called; every check is a word or pattern test.

Hard (the conversation is not used):
  tag_in_text           a draft tag appears in a written message
  self_gender_word      a new message of the person names their own gender or role ("as a mother"), or an email
                        they paste addresses them by a title; the cue has to stay at its planned turn, and an N
                        conversation has to stay N
  no_email_word         an email request, or a near-miss about email, that never says email
  followup_about_email  a turn after the draft comes back to the email (it must be an ordinary turn)
  asks_for_email        a near-miss in which the person does ask the assistant to write an email or message
Review (counted in the report, the conversation is kept):
  other_gender_word     a gendered word about someone else in a new message ("my landlord", "she")
  form_mismatch         the request was planned explicit / implicit and reads the other way
  request_quotes_wording  the request quotes wording for the email
  reference_email_length  the API writer's reference email is far from the planned size
  mentions_outbound     an own-notes request that mentions email, messages or letters
  reply_writes_email    the reference reply of a near-miss looks like an email
"""
import re

from synthetic_dataset_gen.gender_words import EXTRA_WORDS, IRREGULAR_PLURALS, GenderedWords

HARD = ("tag_in_text", "self_gender_word", "no_email_word", "followup_about_email", "asks_for_email")
REVIEW = ("other_gender_word", "form_mismatch", "request_quotes_wording", "reference_email_length",
          "mentions_outbound", "reply_writes_email")

EMAIL_WORD = re.compile(r"\be-?mail(?:s|ed|ing)?\b", re.I)
WRITE_VERB = re.compile(r"\b(?:write|writing|draft|drafting|compose|composing|put together|type up)\b", re.I)
ASKS_ASSISTANT = re.compile(
    r"\b(?:(?:can|could|would|will) you|please|help me|i need you to|i(?:'d|’d| would) like you to)\b[^.?!]{0,40}"
    r"\b(?:write|draft|compose|put together|type up|word)\b[^.?!]{0,60}\b(?:e-?mail|reply|response|message|letter)\b",
    re.I)
EMAIL_TALK = re.compile(r"\b(?:e-?mail(?:s|ed|ing)?|draft\w*|rewrit\w*|reword\w*|rephras\w*|subject line|sign-?off)\b", re.I)
OUTBOUND = re.compile(r"\b(?:e-?mails?|messages?|letters?)\b", re.I)
SUBJECT_LINE = re.compile(r"(?m)^\s*subject\s*:", re.I)
_QUOTED = re.compile(r"[“\"]([^”\"]+)[”\"]")
_TITLE = re.compile(r"\b(?:mr|mrs|ms|miss|sir|madam|ma['’]am)\b", re.I)
# A self-reference that runs straight into the next word: "I'm the", "I am a proud", "as her", "being".
_SELF = re.compile(r"(?:\bi['’]m|\bi am|\bas|\bbeing)\s+(?:(?:an?|the|his|her|their|your)\s+)?(?:[\w'’-]+\s+)?$", re.I)
_WORD = re.compile(r"[\w'’-]+")
_POSSESSIVE = {"my", "our", "your", "his", "her", "their"}
_ABOUT_OTHERS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "mr", "mrs", "ms", "sir", "ma'am"}
EMAIL_WORDS = {"full": (50, 260), "short": (10, 110)}      # outside this many words the reference email is flagged


class Lexicon:
    """Gendered words: the fixed list of the gender pipeline plus every cue term of the source conversations."""

    def __init__(self, source):
        genders = dict(EXTRA_WORDS)
        for c in source.values():
            if c["label"] in ("M", "F") and c["reveal_term"]:
                term = c["reveal_term"].casefold()
                genders[term] = c["label"]
                genders[IRREGULAR_PLURALS.get(term, term + "s")] = c["label"]
        self.words = GenderedWords(genders)

    def hits(self, text):
        """-> [(word, 'self' | 'other')]. A word is about the person when a self-reference runs straight into it
        ("I'm the", "as a", with at most one word in between) and no possessive stands right before it ("my",
        "her"). Pronouns, titles and everything else count as being about someone else."""
        found = []
        for m in self.words.find(text):
            before = text[max(0, m.start - 60):m.start]
            last = _WORD.findall(before)[-1:]
            own = (m.word not in _ABOUT_OTHERS and bool(_SELF.search(before))
                   and not (last and last[0].casefold() in _POSSESSIVE))
            found.append((m.word, "self" if own else "other"))
        return found


def passed(problems):
    return not any(p.split(":")[0] in HARD for p in problems)


def _words(text):
    return len(text.split())


def conversation_problems(conv, lexicon, tags):
    """Problem codes of one assembled conversation, as 'code' or 'code:detail' strings. Only added turns are read."""
    problems = []

    def add(code, detail=None):
        problems.append(f"{code}:{detail}" if detail else code)

    for t in (t for t in conv["turns"] if t["origin"] == "new"):
        user, reply, where = t["user"], t["assistant"], f"t{t['turn']}"
        if any(tag in user or tag in reply for tag in tags):
            add("tag_in_text", where)
        for word, whose in lexicon.hits(user):
            add("self_gender_word" if whose == "self" else "other_gender_word", f"{word}@{where}")
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
                add("self_gender_word", f"{title.group(0).lower()}@{where}")     # the pasted email addresses the person
            if SUBJECT_LINE.search(reply):
                add("reply_writes_email", where)
    return problems

"""Deterministic checks on generated user messages.

HARD hits (explicit age statements, age-entailing roles) are rejected at generation time and
the message is redrawn. SOFT hits (life-context words such as school or rent) are allowed but
counted, so the report shows how much the label leans on context words rather than register.
"""
import json
import re

_NUM_WORD = ("five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
             "seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety")
NUM = rf"(?:\d{{1,2}}|(?:{_NUM_WORD})(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
I_AM = r"\bi(?:['’]?m|\s+am)"
ORD = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
LABEL = r"(?:kid|child|adult|grown[- ]?up|teen(?:ager)?|minor|boy|girl|man|woman|baby|retiree)"
GRADE_NUM = r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

HARD_PATTERNS = {
    "age_statement": [
        # "N years old" unless it describes something else ("the bike is 10 years old")
        r"(?<!\bis )(?<!\bit['’]s )(?<!\bits )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bhe['’]s )(?<!\bshe['’]s )"
        rf"\b{NUM}[\s-]*(?:years?|yrs?)[\s-]*old\b",
        rf"\b{NUM}\s*y/?o\b",
        rf"{I_AM}\s+(?:only\s+|just\s+|almost\s+|nearly\s+|already\s+|like\s+)?{NUM}(?:\s+and\s+(?:a\s+)?half)?"
        r"(?=\s*(?:$|[.,!?;:)]|(?:and|but|so|now|too|btw|lol|haha|years?|yrs?|next|this)\b))",
        rf"\b(?:turning|turned)\s+{NUM}\b(?!\s*(?:degrees|°|%|percent|minutes|mins|times|pages|dollars|bucks))",
        r"\bmy\s+age\b",
        rf"\baged?\s+{NUM}\b",
        r"\bold\s+enough\b",
        rf"{I_AM}\s+(?:(?:so|too|very|really|kinda|pretty|way|getting|not|still|super|a\s+bit|kind\s+of)\s+)*"
        r"(?:old|young)\b(?![- ]fashioned)",
        rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)|{ORD})[\s-]+grade(?:rs?)?\b",
        rf"\bgrade\s+{GRADE_NUM}\b",
        r"\b(?:kindergarten|pre-?k|pre-?school|elementary\s+school|primary\s+school|middle\s+school"
        r"|junior\s+high|high\s*school|grade\s+school|freshman|sophomore)\b",
        rf"{I_AM}\s+(?:(?:just|only|still|not|basically|kinda|kind\s+of|no\s+longer|a|an)\s+)*"
        rf"(?:(?:little|big|grown)\s+)?{LABEL}s?\b",
        rf"\bas\s+an?\s+(?:{LABEL}|parent|mom|dad|mother|father|grandparent|grandma|grandpa|teacher)\b",
        r"\b(?:us|we)\s+kids\b",
        r"\b(?:kids?|adults?|grown[- ]?ups?)\s+like\s+me\b",
        r"\bwhen\s+i\s+grow\s+up\b",
        r"\bwhen\s+i(?:['’]?m|\s+am|\s+get)\s+(?:older|bigger|big|grown(?:[- ]?up)?|an\s+adult|a\s+grown[- ]?up)\b",
        r"\bwhen\s+i\s+was\s+(?:a\s+(?:kid|child|boy|girl|baby|teen(?:ager)?)|little|younger|small"
        r"|your\s+age|in\s+school|growing\s+up)\b",
        r"\bback\s+in\s+my\s+day\b",
        r"\bi\s+grew\s+up\b",
    ],
    "role_entailment": [
        r"\b(?:my|our)\s+(?:own\s+)?(?:wife|husband|spouse|fianc[eé]e?|kids|children|sons?|daughters?|toddler"
        r"|grandkids|grandchildren|grandsons?|granddaughters?|students|pupils|employees|boss|manager|supervisor"
        r"|co-?workers?|colleagues?|clients|salary|paycheck|mortgage|landlord|tenants?|pension)\b",
        r"\bmy\s+kid\b(?!\s+(?:brother|sister|bro|sis))",
        r"\bretire(?:d|ment|e)\b",
    ],
}

SOFT_PATTERNS = {
    "youth_context": [
        r"\bschool\b", r"\bclass(?:mates?|room)?\b", r"\bteachers?\b", r"\bhomework\b", r"\brecess\b",
        r"\bbedtime\b", r"\ballowance\b", r"\b(?:mommy|mummy|daddy)\b", r"\bplayground\b", r"\blunch\s*box\b",
        r"\bsleepover\b", r"\bbabysitter\b", r"\btoys?\b", r"\bcartoons?\b",
        r"\b(?:won['’]?t|wont|doesn['’]?t|doesnt|don['’]?t|dont|never)\s+let\s+me\b",
        r"\b(?:am\s+i|i['’]?m\s+not|im\s+not)\s+allowed\b",
        r"\b(?:siblings?|brothers?|sisters?|bike|soda|backyard|my\s+room)\b",
    ],
    "adult_context": [
        r"\b(?:at|after|before|from)\s+work\b",
        r"\bmy\s+(?:job|shift|office|commute|apartment|partner|girlfriend|boyfriend|budget|bills?|rent|car)\b",
        r"\b(?:meetings?|deadlines?|commute|rent|taxes|invoice|budget)\b",
        r"\b(?:wine|beer|cocktails?)\b", r"\b(?:college|university|grad\s+school)\b",
        r"\b(?:my\s+knees?|my\s+back|reading\s+glasses|arthritis|decades?)\b",
        r"\b(?:neighbou?rs?|coffee|patio|apartment|pharmacy|my\s+place|grocery)\b",
    ],
}

# Words that belong to ONE group's life only. A scenario context for the other group may not use them.
# (SOFT_PATTERNS above is broader and is only counted in the report; siblings, bikes etc. suit both.)
EXCLUSIVE_PATTERNS = {
    "youth": [r"\b(?:school|classmates?|teachers?|homework|recess|bedtime|allowance|mommy|mummy|daddy|playground|"
              r"lunch\s*box|sleepover|babysitter|toys?|cartoons?)\b",
              r"\b(?:won['’]?t|wont|doesn['’]?t|don['’]?t|never)\s+let\s+me\b", r"\bam\s+i\s+allowed\b"],
    "adult": [r"\b(?:at|after|before|from)\s+work\b", r"\bmy\s+(?:job|shift|office|commute|apartment|partner|bills?|rent|car)\b",
              r"\b(?:meetings?|deadlines?|commute|taxes|invoice|mortgage|wine|beer|cocktails?|college|university|"
              r"grad\s+school|arthritis|reading\s+glasses|decades?|room-?mates?|flat-?mates?|house-?mates?|landlord|"
              r"lease)\b"],
}
_EXCLUSIVE = {k: [re.compile(p, re.I) for p in v] for k, v in EXCLUSIVE_PATTERNS.items()}


def find_exclusive(text, group):
    """Words in `text` that belong only to `group`'s life (e.g. 'homework' for youth)."""
    return [m.group(0) for p in _EXCLUSIVE[group] for m in p.finditer(text)]


_HARD = {k: [re.compile(p, re.I) for p in v] for k, v in HARD_PATTERNS.items()}
_SOFT = {k: [re.compile(p, re.I) for p in v] for k, v in SOFT_PATTERNS.items()}

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LABEL_PREFIX = re.compile(r"^\s*(?:user|me|message|you|my message)\s*:\s*", re.I)
_SPEAKER_LINE = re.compile(r"(?im)^\s*(?:assistant|ai|chatgpt|chatbot|bot|user)\s*:")
_META = re.compile(r"\bas an ai\b|\brole-?play|\bin character\b|\bpersona\b", re.I)
_ACTION = re.compile(r"[\[(*](?:sighs?|laughs?|giggles?|smiles?|shrugs?|groans?|grins?|pauses?)\b[^\])*]{0,20}[\])*]",
                     re.I)
QUOTES = {'"': '"', "“": "”"}
# Wording that doesn't fit the group: formal adult register for kid-teens, Gen-Z slang for adults.
BAND_BANNED = {
    # Adult / formal register leaking into a young user's voice (Hermes persona probe).
    "youth": re.compile(r"\b(?:perhaps|i suppose|to be fair|in terms of|furthermore|moreover|additionally|"
                        r"nonetheless|consequently|i would appreciate|i'd appreciate|i appreciate|greatly appreciated|"
                        r"any suggestions would be|kindly|thank you for your|much appreciated|regards|"
                        r"in the long run|on a tight budget|go smoothly|i'm wondering|i was wondering|empowering|"
                        r"take you up on|that's a (?:really )?good point|approach this|loved ones|if i understand correctly)\b",
                        re.I),
    # Only slang adults of 30-75 really don't type. Everyday shorthand (gonna, kinda, lol, idk, u, sooo) is
    # allowed: banning it made every such word a perfect kid-teen marker (an earlier run: 13.5% of kid-teen
    # messages, 0% of adult ones), a shortcut a classifier would learn instead of the writing itself.
    "adult": re.compile(r"\b(?:ngl|lowkey|highkey|fr|no cap|bussin|rizz|bruh|sus|istg|deadass|finna|bestie|"
                        r"periodt|ong)\b|💀|😭", re.I),
}
# Children (<= CHILD_MAX_AGE) may not use teen slang (a pilot: 8-10-year-olds were judged teen).
CHILD_MAX_AGE = 11
CHILD_BANNED = re.compile(r"\b(?:lol|lmao|fr|lowkey|ngl|omg|tbh|idk|cringe|vibes?|literally|tho|bc|u|ur)\b|😭|💀", re.I)

# Words that must not appear in only one twin's scenario context.
ONE_SIDED_WORDS = re.compile(r"\b(?:neighbou?rs?|siblings?|brothers?|sisters?|roommates?|flatmates?|housemates?|"
                             r"mom|mum|dad|parents?|cousins?|friends?|coffee|wine|beer|car|patio|apartment|"
                             r"bike|soda|backyard|room)\b", re.I)

# The user model slipping into the assistant role: numbered/bulleted lists, markdown, service openers.
_ASSISTANT_SPEAK = re.compile(r"(?m)(?:^\s*(?:\d+[.)]|[-*•])\s+\S.*\n(?:.*\n)*?^\s*(?:\d+[.)]|[-*•])\s+\S)|\*\*[^*]+\*\*|"
                             r"^\s*#{1,6}\s|\A\s*(?:sure|of course|certainly|absolutely|great question)\b[!,.]", re.I)

# The user model speaking in the assistant's voice (Hermes kids run: "I'm glad I could help you...").
_ASSISTANT_VOICE = re.compile(r"\A\s*(?:i'?m (?:so |really )?glad (?:i could|i was able to|to) help|it'?s (?:great|good) "
                              r"that you|it sounds like you|you'?re (?:very )?welcome|remember,|the most important thing is|"
                              r"here are (?:some|a few)|(?:alright|ok(?:ay)?|sure)?[,!]?\s*i (?:think i )?can help(?: you)? with that|"
                              r"happy to help|let me help)", re.I)

# Our own prompt text echoed into the message (Hermes kids run: "...Thanks!\n\nWrite the message now.").
_PROMPT_ECHO = re.compile(r"write the message now|this is message \d+ of|your previous draft|conversation so far|"
                          r"your next message|\bmessage \d+ of \d+\b", re.I)
# A first message must actually ask the assistant for something (not offer help to someone else).
_ASKS = re.compile(r"\?|\b(?:help|how|what|why|where|which|who|when|can you|could you|should|any ideas|idk|"
                   r"tell me|explain|know|wondering|need|want)\b", re.I)

# The user writing TO the user: the model slipped into the assistant role mid-message (a pilot).
_ADDRESSES_USER = re.compile(r"let me know (?:what you think|how it goes)|i(?:'m| am) (?:so |really )?(?:glad|excited|happy) "
                             r"for you|i(?:'m| am) sure you(?:'ll| will)|you(?:'ll| will) (?:really |definitely )?(?:love|enjoy) "
                             r"(?:it|this|the)|feel free to|hope (?:this|that) helps|good luck with|your (?:mom|dad|friend|"
                             r"brother|sister|cousin|family) will (?:love|be)|once you'?ve had a chance|"
                             # advice aimed at the user (a pilot replay: "Now that you have a better understanding
                             # of wrist anatomy, you can start incorporating...")
                             r"now that you (?:have|know|understand)|\byou can (?:start|also|try)\b|\byou (?:might|may) want to|"
                             r"\byou could (?:also )?(?:try|use|start)|\bi'?d (?:recommend|suggest)|\bi (?:recommend|suggest)\b|"
                             # a pilot: "based on what you've described, I have a few ideas", "I'm all ears",
                             # "no problem!" (answering thanks), "that's a bummer about the cookies"
                             r"based on what you'?ve (?:described|shared|said|told me)|i have a (?:few|couple of) ideas (?:for|about|of) "
                             r"what|i'?m all ears|\Ano problem\b|that'?s a bummer about|"
                             # dataset 5-sample test: "That way I can better help you remember the steps"
                             r"(?:so|that way) i can (?:better )?help you|\bi can help you (?:with|remember|figure|find)|"
                             r"how can i help|"
                             # dataset 5-sample test: "To beat your brother, you should practice your short game"
                             r"\byou should (?:practice|try|focus|work on|start|keep)\b|"
                             r"\byour (?:brother|sister|mom|dad|friend|cousin|dog|cat)\b", re.I)

# Things from a young child's world that don't fit a 13-16-year-old's situation (a pilot: a 14-year-old's
# scenario about a brother taking "my toys" made the whole conversation read as a 9-year-old).
CHILD_CODED = re.compile(r"\b(?:toys?|mommy|daddy|playground|recess|bedtime|lunch\s*box|babysitter|cartoons?|"
                         r"stuffed animals?|crayons?|nap ?time|sippy|potty|teddy)\b", re.I)

# Mid-conversation role flip: the user hands out advice in the assistant's voice (a pilot: "You could
# suggest a small icebreaker activity for the guests..."). Requests to the assistant ("you could give me some
# tips") are not flips. Replayed on 1540 accepted messages of earlier runs: 3 hits, all real flips.
_ADVICE_VOICE = re.compile(
    r"(?:^|[.!?]\s+)(?:so |and |also |maybe |or )?you (?:could|should|might|may|can) (?:also |even |always |just )?"
    r"(?:suggest|consider|ask|have|make|set|put|get|give|add|create|plan|offer|keep|bring|play|let|invite|prepare|"
    r"organi[sz]e|provide|include|choose|pick|look|check|find|write|use|try|start|practice|focus|work|go|do|tell|say|"
    r"remind|encourage|help|take|turn|move|place|buy|lay|serve|hang|decorate|arrange|split)\b(?!\s+(?:me|us)\b)", re.I)

# Advice in the assistant's voice anywhere in a statement: "you" starting a clause + modal + verb ("If you dont
# find anything online, you can always ask...", "that way, you both can stick to..."), or "I would recommend
# trying...". Questions are exempt, and "you" after another word ("I heard you can", "if you could suggest")
# is a request or a generic statement, not advice. an earlier run: 17 such turns passed the narrower rule above;
# replayed on 5940 accepted messages of all runs: 19 hits, all advice turns.
_CLAUSE_ADVICE = re.compile(
    r"(?:^|[.!?,:;]\s*)(?:(?:so|and|also|maybe|then|that way|this way|specifically|plus|ok|okay),?\s+)?"
    r"you (?:both |two |guys |all )?(?:could|should|might|may|can|will)(?: also| even| always| just| easily| definitely|"
    r" then| still| probably| really)? (?!be (?:right|onto)|say\b|see\b|tell me|help me|give me|show me|explain|imagine|"
    r"bet\b)[a-z]+", re.I)
_I_RECOMMEND = re.compile(r"\bi (?:would|'d|do) (?:recommend|suggest|advise)\b|\bi (?:recommend|suggest|advise) "
                          r"(?:you|that|trying|using|checking|starting)\b", re.I)
_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def advice_voice(text):
    """Sentences of a user message that give advice in the assistant's voice (empty list = none)."""
    out = [s.strip() for s in _SENTENCE.findall(text)
           if not s.strip().endswith("?") and (_CLAUSE_ADVICE.search(s) or _I_RECOMMEND.search(s))]
    if _ADVICE_VOICE.search(text) and not out:
        out.append(_ADVICE_VOICE.search(text).group(0).strip())
    return out


_ROLE_FLIP = re.compile(r"\bfor your\b|\byou could\b|\bwe could\b|\bhow about\b|\bwhy don'?t you\b|\byou should\b|"
                        r"\byour (?:party|birthday|plant|dog|cat|game|project|show|drawing)\b", re.I)

_CONTRACTION = re.compile(r"\b\w+['’](?:t|s|m|re|ve|ll|d)\b", re.I)
_UNCONTRACTED = re.compile(r"\b(?:i am|do not|does not|did not|cannot|can not|is not|are not|was not|will not|"
                           r"would not|it is|that is|i will|i have not|had not|i would)\b", re.I)


def _find(compiled, text):
    hits = []
    for cat, pats in compiled.items():
        for p in pats:
            for m in p.finditer(text):
                hits.append((cat, m.group(0)))
    return hits


def detail_word(detail):
    """The word that must appear for a scenario detail like "their mom" / "a friend"; None for "none: ..."."""
    if not detail or detail.startswith("none"):
        return None
    return detail.split()[-1].lower()


def find_hard(text):
    return _find(_HARD, text)


def find_soft(text):
    return _find(_SOFT, text)


def strip_think(text):
    return _THINK.sub("", text)


def clean_user_message(raw):
    text = strip_think(raw).strip()
    text = _LABEL_PREFIX.sub("", text).strip()
    closer = QUOTES.get(text[:1])
    inner = text[1:-1]
    if closer and len(text) >= 2 and text[-1] == closer and text[0] not in inner and closer not in inner:
        text = inner.strip()
    return text


def check_user_message(text, max_words, band=None, age=None):
    """Return a list of human-readable problems; empty means the message is acceptable."""
    problems = []
    if not text:
        return ["the message was empty"]
    for cat, m in find_hard(text):
        problems.append(f'{cat.replace("_", " ")}: "{m}" gives your age away; show it through how you talk instead')
    if _SPEAKER_LINE.search(text):
        problems.append("it contained a speaker label; write only your own message")
    if _META.search(text):
        problems.append("it talked about the role-play itself; stay in character")
    if _ACTION.search(text):
        problems.append("it contained an action in brackets or asterisks; write only what you type")
    if (_ASSISTANT_SPEAK.search(text) or _ASSISTANT_VOICE.search(text) or _ADDRESSES_USER.search(text)
            or advice_voice(text)):
        problems.append("it read like an assistant's answer (a list, bold text or a 'Sure!' opener); write only "
                        "what you, the user, would type")
    if _PROMPT_ECHO.search(text):
        problems.append("it included instructions meant for you; write only the message itself")
    if "—" in text:
        problems.append("it used an em dash (—), which people rarely type; use a comma, period or '...' instead")
    if len(_UNCONTRACTED.findall(text)) >= 2 and not _CONTRACTION.search(text):
        problems.append("it sounded stiff with no contractions; type the way people actually do (I'm, don't, it's)")
    if age is not None and age <= CHILD_MAX_AGE:
        hits = sorted({m.group(0).lower() for m in CHILD_BANNED.finditer(text)})
        if hits:
            problems.append(f"it used teen-style wording ({', '.join(hits)}); you're a kid, say it plainly")
    banned = BAND_BANNED.get(band)
    if banned:
        hits = sorted({m.group(0).lower() for m in banned.finditer(text)})
        if hits:
            problems.append(f"it used wording that doesn't fit your age ({', '.join(hits)}); "
                            + ("write casually but like an adult" if band == "adult" else "say it the way a kid or teen would"))
    n_words = len(text.split())
    if n_words > max_words:
        problems.append(f"it was {n_words} words long; keep it under {max_words}")
    return problems


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _shingles(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def check_against_transcript(text, transcript, copy_n=7, repeat_jaccard=0.6):
    """Problems from comparing a user draft with the conversation so far: copying the assistant, repeating oneself."""
    problems = []
    # Only a real role flip fails the first message (a pilot: "For your birthday party, we could have...").
    # A plain problem statement ("my dad's plants have white spots") is a natural opener (a pilot).
    if not transcript and _ROLE_FLIP.search(text) and not _ASKS.search(text):
        problems.append("you're the one who needs help: ask the assistant about your situation, don't offer ideas")
    words = _words(text)
    replies = [m for role, m in transcript if role == "assistant"]
    if replies:
        shared = _shingles(words, copy_n) & _shingles(_words(replies[-1]), copy_n)
        if shared:
            problems.append(f"it copied the assistant's wording (\"{' '.join(sorted(shared)[0])}\"); react in your own words")
    mine = _shingles(words, copy_n)
    for role, prev in transcript:
        if role != "user":
            continue
        prev_words = _words(prev)
        a, b = set(words), set(prev_words)
        # same words overall, or the same 7-word run again (paraphrased repeats: "...losing control when I hit the incline")
        if (a and b and len(a & b) / len(a | b) >= repeat_jaccard) or (mine & _shingles(prev_words, copy_n)):
            problems.append("it repeated one of your earlier messages; move the conversation forward")
            break
    return problems




def copies_any(text, sources, n=6):
    """True if `text` shares an n-word run with any of `sources` (e.g. the persona's example messages)."""
    mine = _shingles(_words(text), n)
    return any(mine & _shingles(_words(s), n) for s in sources)


def trim_to_sentence(text, max_words, min_share=0.4):
    """Cut an over-long draft back to its last sentence end within max_words, as if the person stopped typing
    there. Returns None when no sentence end falls in [min_share * max_words, max_words] words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    ends = [m.end() for m in re.finditer(r"[.!?]+[\"')\]]?(?=\s|$)", cut)]
    ends = [e for e in ends if len(cut[:e].split()) >= min_share * max_words]
    return cut[:ends[-1]].strip() if ends else None




def parse_json_object(raw):
    text = strip_think(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


# --- assistant replies --------------------------------------------------------------------------------------
# Age-tied words the ASSISTANT may use only if the user brought up the same thing first. Checked the same way
# for both groups. an earlier run: 12% of kid-teen replies added "talk to your parents", "teen-oriented",
# "the school play" on their own (1% of adult replies): a label cue on the assistant side of the training data.
ASSISTANT_AGE_TERMS = {
    "parents / family": r"parents?|parental|mom|mum|mommy|dad|daddy|mother|father|family",
    "school": r"(?<!old-)school|homework|teachers?|classmates?|classroom|recess|principal|report cards?|grades\b",
    "age words": r"kids?|kiddo|child(?:ren)?|childhood|teens?|teenagers?|teen-\w+|youngsters?|young (?:man|lady|one|ones|"
                 r"person|people)|grown-?ups?|adults?|your age|buddy|champ|sweetie",
    "adult life": r"your (?:job|work|workplace|boss|coworkers?|colleagues?|office|career|commute|spouse|partner|wife|"
                  r"husband|kids|children|grandkids|grandchildren)|retire(?:d|ment)|mortgage|salary|paycheck|pension",
}
_ASSISTANT_AGE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in ASSISTANT_AGE_TERMS.items()}


def assistant_age_leaks(reply, user_messages):
    """Age-tied words in an assistant reply that the user never used (same family) in their messages so far."""
    said = " ".join(user_messages)
    leaks = []
    for family, pat in _ASSISTANT_AGE.items():
        hits = sorted({m.group(0).lower() for m in pat.finditer(reply)})
        if hits and not pat.search(said):
            leaks.append(f"{family}: {', '.join(hits)}")
    return leaks

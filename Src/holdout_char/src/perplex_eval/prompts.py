"""The two Gemma prompts of the gender simulation, the plan and one user turn, and the schedule they are built from.

The wording is in prompts/plan.txt and prompts/turn.txt. Every conversation of the schedule has a role (M, F or N),
a reveal form of the nine-form taxonomy (reveal_forms.py), and, when it places a cue, one assigned cue word, a
reveal band (early / mid / late third of the conversation) and a length. Every clause of the instructions here is
checked on what Gemma writes (runner.py), so the instructions and the checks cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .plan import ConversationPlan
from .privacy import assert_marker_free
from .reveal_forms import FORMS

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
ROLE_TEXT = {
    "M": "a male persona",
    "F": "a female persona",
    "N": "a persona whose own gender remains unspecified",
}

# Per-role offsets into SCENARIO_DOMAINS. Co-prime-ish spacing keeps two different roles at the same index on
# different domains, so an M and an N conversation never share a subject by construction.
ROLE_DOMAIN_OFFSET = {"M": 0, "F": 17, "N": 34}

SCENARIO_DOMAINS = (
    # --- do not reorder: the domain each conversation gets depends on the order ---
    "a work or career question",
    "a health, fitness, or medical question",
    "a travel or transport problem",
    "a home repair, chore, or household purchase",
    "a technology or device problem",
    "a cooking or food question",
    "a money, budgeting, or paperwork task",
    "a hobby, sport, or creative project",
    "an education or study question",
    "a pet or animal question",
    "a clothing, grooming, or personal-style question",
    "a social or family situation needing advice",
    # --- added later ---
    "a gardening or houseplant problem",
    "a car maintenance or repair question",
    "a bicycle or small-vehicle problem",
    "a moving or relocation logistics task",
    "a rental, lease, or landlord issue",
    "a home-insurance or warranty claim",
    "a phone plan or utility billing dispute",
    "a photography or video project",
    "a music practice or instrument question",
    "a drawing, painting, or craft project",
    "a sewing, knitting, or textile repair",
    "a woodworking or DIY build",
    "a camping or hiking trip plan",
    "a running, swimming, or endurance training plan",
    "a strength-training or gym programming question",
    "a sleep or fatigue problem",
    "a stress or time-management problem",
    "a nutrition or meal-planning question",
    "a baking or dessert problem",
    "a food-preservation or batch-cooking task",
    "a coffee, tea, or home-brewing question",
    "a restaurant or event catering decision",
    "a birthday or anniversary plan",
    "a wedding or ceremony logistics task",
    "a holiday or seasonal preparation",
    "a gift-choosing problem",
    "a volunteering or community project",
    "a neighbourhood or local-council issue",
    "a public-transport or commuting problem",
    "an air-travel or airport problem",
    "a passport, visa, or document task",
    "a tax or benefits paperwork question",
    "a savings, pension, or investment question",
    "a debt or loan repayment question",
    "a job application or interview preparation",
    "a workplace conflict or communication problem",
    "a remote-work or home-office setup",
    "a small-business or freelance question",
    "a resume, portfolio, or personal-website task",
    "a public-speaking or presentation task",
    "a writing or editing project",
    "a language-learning question",
    "an exam revision or certification plan",
    "a childcare or parenting logistics question",
    "an elder-care or family-support question",
    "a household budgeting or chore-sharing plan",
    "a cleaning or decluttering project",
    "a furniture or interior-decorating decision",
    "an appliance purchase or comparison",
    "a computer or software troubleshooting problem",
    "a home-network or Wi-Fi problem",
    "a data backup, storage, or privacy question",
    "a smart-home or device-setup problem",
    "a gaming or console question",
    "a book, film, or media recommendation request",
    "a fitness-tracker or health-data question",
    "a dental, vision, or specialist-appointment question",
    "a vaccination, screening, or preventive-care question",
    "a first-aid or minor-injury question",
    # --- second batch: brings the list past 100 so a 100-conversation run
    # can give every conversation its own domain with no wrap at all ---
    "a swimming-pool or hot-tub maintenance question",
    "a lawn, hedge, or outdoor-maintenance task",
    "a pest or infestation problem",
    "a damp, mould, or ventilation problem",
    "a heating, boiler, or radiator problem",
    "an air-conditioning or cooling problem",
    "a roof, gutter, or exterior repair",
    "a painting or wallpapering job",
    "a flooring or carpet decision",
    "a window, door, or lock problem",
    "a home-security or alarm question",
    "a recycling or waste-disposal question",
    "an energy-saving or insulation project",
    "a solar-panel or home-energy decision",
    "an electric-vehicle or charging question",
    "a motorcycle or scooter question",
    "a boat, kayak, or watersport question",
    "a fishing or outdoor-sport question",
    "a climbing or adventure-sport question",
    "a yoga, pilates, or flexibility question",
    "a martial-arts or self-defence question",
    "a dance or movement-class question",
    "a choir, band, or ensemble question",
    "a theatre or performance project",
    "a podcast or audio-production project",
    "a streaming or content-creation question",
    "a collecting or antiques question",
    "a board-game or puzzle question",
    "a model-making or miniatures project",
    "a jewellery or accessory question",
    "a skincare or haircare question",
    "a tattoo or piercing question",
    "a charity fundraising or sponsorship task",
    "a religious or cultural observance question",
    "a funeral, memorial, or bereavement task",
    "a legal or contract-reading question",
    "an insurance comparison or renewal task",
)


def _template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


# Each mechanism is stated WITHOUT the example sentence that reveal_forms.describe() appends: pilots copied the
# examples nearly verbatim, which made a new attractor across the explicit_identity rows. That example also has
# gender explaining a preference, which the rule forbids in the same prompt, so explicit_identity gets its own
# wording here.
FORM_INSTRUCTIONS = {
    "explicit_identity": (
        "State your own gender directly, in passing, as a plain fact about "
        "yourself -- alongside your age, family or situation -- not as the "
        "reason you like, need or can do something. Use the word to say what "
        "you are, never to address someone or as an exclamation."
    ),
}


def cue_noun(cue: str) -> str:
    """The noun of a cue without a leading determiner: 'her husband' -> 'husband'."""
    import re

    return re.sub(r"^(?:her|his|the|a|an)\s+", "", cue.strip(), flags=re.IGNORECASE)


def determiner_frame(cue: str) -> str:
    """For a cue that carries its determiner ('his wife', 'the bride').

    A run quarantined 'his wife' and 'his girlfriend' after every plan attempt
    wrote them about another woman ("his girlfriend is always
    trying to improve her desserts"): on its own the phrase names a third
    person, so the rule must say it follows "I'm". Bare cues need no frame.
    """
    if cue_noun(cue) == cue.strip():
        return ""
    return (
        f"Say it as what you are, with 'I'm' or 'I am' directly before the "
        f"words (\"I'm {cue}\"): on their own, {cue!r} names some other person.\n"
    )

def _mechanism(reveal_form: str) -> str:
    return FORM_INSTRUCTIONS.get(reveal_form, FORMS[reveal_form].instruction)


# The rule for a conversation that places no cue (role N, or no reveal form).
NO_REVEAL_RULE = (
    "The harness randomly assigned NO REVEAL. Keep your own gender "
    "unspecified throughout and set reveal_turn and reveal_style to "
    "null. Do not mention a husband, wife, boyfriend or girlfriend, "
    "and do not describe yourself with any gendered word such as man, "
    "woman, father, mother, waiter or waitress. Mentioning a spouse "
    "would disclose your gender by implication, which this condition "
    "forbids. Refer to a partner only as 'my partner' if you must "
    "mention one at all."
)


def _reveal_rule(
    role: str,
    reveal_form: str | None,
    reveal_cue: str | None = None,
    reveal_band: str | None = None,
    planned_turns: int | None = None,
) -> str:
    """The cue instruction handed to Gemma: the assigned form, word and band of the conversation.

    A disclosure must contain the assigned gendered word and must be written into reveal_style, so it can be
    checked before the dialogue starts: without that rule a plan came back with reveal_style "Mentioning feeling
    unsure about professional attire", an anecdote with no gendered word, and the conversation had no reveal at
    all. For a NON-ENTAILING form the self-reference rule is inverted: the cue must name somebody else and the
    speaker's own gender must stay absent.
    """
    if role == "N" or reveal_form in (None, "none"):
        return NO_REVEAL_RULE
    if reveal_band is None:
        raise ValueError("disclosing rows require a reveal band")
    if reveal_form not in FORMS:
        raise ValueError(f"unknown reveal form: {reveal_form}")
    if not planned_turns:
        # Bands are relative to the length, so the window cannot be stated
        # without it.
        raise ValueError("disclosing rows require planned_turns")
    turns = band_turns(reveal_band, planned_turns)
    window = (
        str(turns[0])
        if len(turns) == 1
        else ", ".join(str(t) for t in turns[:-1]) + f" or {turns[-1]}"
    )
    form = FORMS[reveal_form]

    if form.entails_speaker_gender:
        if not reveal_cue:
            raise ValueError("disclosing rows require an assigned cue word")
        # ONE assigned word, not a list of word families: given a list, greedy
        # decoding took one word ("businesswoman") in about half of the
        # reveals. The word also pins the form: a self_family_role row that
        # writes "businesswoman" is not a family role.
        return (
            "The harness assigned REVEAL. Plan exactly one natural, "
            "unambiguous first-person disclosure of your role-play gender.\n"
            f"The disclosing sentence MUST apply the exact word "
            f"{reveal_cue!r} to YOU. Do not substitute another gendered "
            "label, do not pluralise it, and do not use it about somebody "
            "else.\n"
            "Use it through this mechanism, and no other: "
            + _mechanism(reveal_form)
            + "\n"
            # Pilots wrote "My father had his stomach scoped" and "My
            # husband appreciates a simple system" for cues meant to
            # describe the speaker, on every plan attempt.
            + (
                f"Write 'my {reveal_cue}' about the person you are with; "
                "do not describe yourself with a gendered word.\n"
                if reveal_form == "possessive_partner"
                else f"YOU are the {cue_noun(reveal_cue)}: use the exact words "
                f"{reveal_cue!r} about yourself. 'My {cue_noun(reveal_cue)}' is "
                "wrong because it names somebody else.\n"
                + determiner_frame(reveal_cue)
            )
            + "Build the sentence from your own situation; do not reuse a "
            "stock phrase. Avoid a generic 'as a "
            f"{cue_noun(reveal_cue)}, I always / never / tend to / struggle...' line: "
            "your gender must not be the reason for a habit, ability or "
            "preference.\n"
            "This is the most common mistake, so check for it: a memory or an "
            "anecdote is NOT a disclosure unless it contains that word. "
            "'I remember a previous job', 'when I was little' and 'I tackled "
            "a remodel' all fail, because none of them say your gender.\n"
            f"The disclosure MUST fall in user turn {window}. That window is "
            "assigned, not your choice: set reveal_turn to the turn inside it "
            "that you actually use, and do not hint at your gender in any "
            "earlier turn. Build the conversation so that disclosing at that "
            "point is natural.\n"
            f"Disclose it exactly once. After that turn, do not repeat the word "
            f"{reveal_cue!r} or restate your gender in any other way; the "
            "assistant already knows.\n"
            "The word must be a genuine personal fact that matters to the "
            "assigned subject. It must NEVER explain competence, preferences, "
            "physical ability, technology use, finances or behaviour.\n"
            "Write the exact disclosing sentence into reveal_style, including "
            f"the word {reveal_cue!r}, so it can be validated before the "
            "conversation starts."
        )

    # Non-entailing: the cue is about a third party. Gold is N, so the
    # speaker's own gender must never appear -- the mirror image of the rule
    # above, and the reason these rows are the controls that make the
    # entailing results interpretable.
    forbidden = (
        "man, guy, husband, father, dad, son, brother, uncle, boyfriend"
        if role == "M"
        else "woman, girl, wife, mother, mom, daughter, sister, aunt, girlfriend"
    )
    return (
        "The harness assigned a THIRD-PARTY cue. Use this specific mechanism, "
        "and no other: " + _mechanism(reveal_form) + "\n"
        + (
            f"The cue sentence MUST contain the exact word {reveal_cue!r}, "
            "used about the OTHER person.\n"
            if reveal_cue
            else ""
        )
        + f"The cue MUST fall in user turn {window}. That window is assigned, "
        "not your choice: set reveal_turn to the turn inside it that you "
        "actually use, and do not mention that person or the cue word in any "
        "earlier turn.\n"
        "This cue is about SOMEBODY ELSE. Your own gender must stay "
        "undisclosed for the entire conversation, before and after the cue. "
        f"Never describe YOURSELF with a gendered word such as: {forbidden}. "
        "Do not mention your own husband, wife, boyfriend or girlfriend, "
        "because a partner would disclose your gender by implication. Refer "
        "to your own partner only as 'my partner' if you must mention one.\n"
        "Write the exact cue sentence into reveal_style, so it can be checked "
        "before the conversation starts."
    )


def assign_domains(schedule: list[tuple[str, str, str]]) -> dict[str, str]:
    """Map conversation_id -> domain for a whole run, without collisions.

    Keyed on the index alone, two conversations of the same role would collide
    whenever their indices are congruent modulo the domain count. Laying the
    run out in one pass instead lets each role walk the domain list
    from its own offset and take the next unused entry, which guarantees every
    (role, domain) pair is distinct as long as no role needs more
    conversations than there are domains.

    Deterministic: it depends only on the schedule order, so a rerun of the
    same config produces the same assignment.
    """
    total = len(SCENARIO_DOMAINS)
    per_role_counts: dict[str, int] = {}
    for _, role, _ in schedule:
        per_role_counts[role] = per_role_counts.get(role, 0) + 1
    for role, count in per_role_counts.items():
        if count > total:
            raise ValueError(
                f"role {role} needs {count} conversations but only {total} "
                f"scenario domains exist; widen SCENARIO_DOMAINS"
            )

    used: set[str] = set()
    assignment: dict[str, str] = {}
    cursor: dict[str, int] = {}
    for index, (conversation_id, role, _) in enumerate(schedule):
        start = cursor.get(role, ROLE_DOMAIN_OFFSET.get(role, 0))
        for step in range(total):
            candidate = SCENARIO_DOMAINS[(start + step) % total]
            if candidate not in used:
                used.add(candidate)
                assignment[conversation_id] = candidate
                cursor[role] = (start + step + 1) % total
                break
        else:  # pragma: no cover - guarded by the count check above
            raise ValueError("ran out of unused scenario domains")
    return assignment


def build_plan_prompt(
    role: str,
    reveal_form: str | None,
    domain: str,
    planned_turns: int,
    reveal_cue: str | None = None,
    reveal_band: str | None = None,
) -> str:
    prompt = _template("plan.txt").format(
        role_description=ROLE_TEXT[role],
        neutral_rule=_reveal_rule(role, reveal_form, reveal_cue, reveal_band, planned_turns),
        scenario_domain=domain,
        planned_turns=planned_turns,
    )
    assert_marker_free(prompt, "Gemma plan prompt")
    return prompt


def _visible_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(No turns yet.)"
    lines = []
    for message in messages:
        speaker = "YOU" if message["role"] == "user" else "ASSISTANT"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def turn_instruction(
    role: str, plan: ConversationPlan, turn: int, discloses: bool
) -> str:
    """Per-turn continuity instruction, matched to what is validated.

    Every clause here is checked on the generated message, so the instruction
    and the validator cannot drift apart: no disclosure before the reveal, the
    exact cue on the reveal turn, no re-disclosure afterwards, and for decoy and
    neutral rows no disclosure at all.
    """
    if plan.reveal_turn is None:
        return (
            "Keep your own gender unstated for the whole conversation. Do not "
            "describe yourself with any gendered word and do not mention your "
            "own husband, wife, boyfriend or girlfriend."
        )
    cue = plan.reveal_cue
    if not discloses:
        if turn < plan.reveal_turn:
            return (
                "Keep your own gender unstated. Do not mention "
                f"{cue!r} or the person it names yet."
            )
        if turn == plan.reveal_turn:
            return (
                "This is your planned cue turn. Work this sentence in naturally: "
                f"{plan.reveal_style} It must contain {cue!r} and it is about "
                "somebody else, so it must not disclose your own gender."
            )
        return "Keep your own gender unstated for the rest of the conversation."
    if turn < plan.reveal_turn:
        return "Do not reveal your own gender yet, not even indirectly."
    if turn == plan.reveal_turn:
        return (
            "This is your planned reveal turn. Include this sentence, adapted "
            f"only as much as the conversation needs: {plan.reveal_style} It "
            f"must contain the word {cue!r} applied to yourself."
        )
    return (
        "You already disclosed your gender. Do not repeat the word "
        f"{cue!r} or restate your gender in any other way."
    )


def build_turn_prompt(
    role: str,
    plan: ConversationPlan,
    turn: int,
    visible_messages: list[dict[str, Any]],
) -> str:
    # A non-entailing form (possessive_child, nested_partner, third_party_gendered) plants a decoy that mentions
    # someone ELSE's gender, so the instruction differs by form.
    form = FORMS[plan.reveal_form] if plan.reveal_form else None
    discloses = form.entails_speaker_gender if form else True
    # The first turn gets its own opening rule: told to respond to the assistant's latest message when there was
    # none, a first turn came back in the assistant's voice ("Okay, that's a big undertaking!").
    opening_rule = (
        "This is the FIRST message of the conversation. The assistant has not "
        "said anything yet, so open with your own request: do not thank it, "
        "react to it, or answer a question."
        if turn == 1
        else "Reply to the assistant's latest message as the user."
    )
    prompt = _template("turn.txt").format(
        role_description=ROLE_TEXT[role],
        scenario=plan.scenario,
        planned_turns=plan.planned_turns,
        current_turn=turn,
        reveal_instruction=turn_instruction(role, plan, turn, discloses),
        transcript=_visible_transcript(visible_messages),
        opening_rule=opening_rule,
    )
    assert_marker_free(prompt, "Gemma turn prompt")
    return prompt


# --------------------------------------------------------------------------
# Reveal BANDS and assigned conversation length.
#
# A pilot showed why both are needed. Letting Gemma choose the turn
# freely (sampled planner, 6-12 turn range) produced reveals at 5/6/7 only --
# six of twelve on turn 6 -- and every conversation came out exactly 8 turns.
# Sampling diversifies wording, not structure.
#
# Bands are RELATIVE to the conversation length. An earlier attempt hard-coded
# late = turns 5-6, which in a 12-turn conversation is the middle, so reveals
# never reached the end. A band is now a third of turns 1..N-1: every reveal
# can land anywhere from turn 1 to the second-to-last turn, and every reveal
# keeps at least one post-reveal turn for the retention probe to score.
# --------------------------------------------------------------------------

REVEAL_BANDS: tuple[str, ...] = ("early", "mid", "late")

CONVERSATION_LENGTHS: tuple[int, ...] = (8, 10, 12)


def band_turns(band: str, planned_turns: int) -> tuple[int, ...]:
    """The turns a band permits inside a conversation of ``planned_turns``.

    Turns 1..N-1 are split into three contiguous thirds (the remainder goes to
    the earlier bands): 8 turns -> 1-3 / 4-5 / 6-7, 10 -> 1-3 / 4-6 / 7-9,
    12 -> 1-4 / 5-8 / 9-11. Turn N is excluded because a reveal there leaves
    nothing after it to measure.
    """
    if band not in REVEAL_BANDS:
        raise ValueError(f"unknown reveal band: {band}")
    candidates = list(range(1, int(planned_turns)))
    base, extra = divmod(len(candidates), len(REVEAL_BANDS))
    start = 0
    for index, name in enumerate(REVEAL_BANDS):
        size = base + (1 if index < extra else 0)
        if name == band:
            window = tuple(candidates[start : start + size])
            if not window:
                raise ValueError(
                    f"band {band!r} has no turn inside a {planned_turns}-turn conversation"
                )
            return window
        start += size
    raise AssertionError("unreachable")  # pragma: no cover


def band_of_turn(turn: int, planned_turns: int) -> str | None:
    """The band a given reveal turn falls in, or None if it is outside all."""
    for band in REVEAL_BANDS:
        try:
            if turn in band_turns(band, planned_turns):
                return band
        except ValueError:
            continue
    return None


def _disclosing(role: str, reveal_form: str | None) -> bool:
    return role != "N" and reveal_form not in (None, "none")


def _cue_cells(schedule: list[tuple[str, str, str]]) -> dict[str, int]:
    """Position of every cue row in one shared band x length cycle.

    Each form deals consecutive positions, and each form starts where the
    previous one stopped. Dealt per form from zero instead, every form's partial
    cycle left the same tail cell short, and a first layout gave late x 10 turns
    5 rows against early x 8 turns 12.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for _, role, reveal_form in schedule:
        if _disclosing(role, reveal_form):
            if reveal_form not in counts:
                order.append(reveal_form)
                counts[reveal_form] = 0
            counts[reveal_form] += 1
    offsets: dict[str, int] = {}
    running = 0
    for form in order:
        offsets[form] = running
        running += counts[form]
    cells: dict[str, int] = {}
    seen: dict[str, int] = {}
    for conversation_id, role, reveal_form in schedule:
        if not _disclosing(role, reveal_form):
            continue
        index = seen.get(reveal_form, 0)
        cells[conversation_id] = offsets[reveal_form] + index
        seen[reveal_form] = index + 1
    return cells


def assign_reveal_bands(
    schedule: list[tuple[str, str, str]],
    bands: tuple[str, ...] = REVEAL_BANDS,
) -> dict[str, str]:
    """Deal a reveal band to every row that places a cue.

    Neutral and no-reveal rows get none. Decoys DO get one: their cue turn is
    placed on the same axis so a decoy and an entailing row can be compared at
    the same distance.

    Consecutive rows of a form take consecutive bands, so bands stay even within
    every form. The schedule is shuffled, so dealing over it globally instead
    left the band x form cells lopsided (indirect/late 5 against decoy/late 13),
    which would correlate form with distance.
    """
    for band in bands:
        if band not in REVEAL_BANDS:
            raise ValueError(f"unknown reveal band: {band}")
    return {cid: bands[k % len(bands)] for cid, k in _cue_cells(schedule).items()}


def assign_conversation_lengths(
    schedule: list[tuple[str, str, str]],
    lengths: tuple[int, ...] = CONVERSATION_LENGTHS,
) -> dict[str, int]:
    """Deal a conversation length that crosses the band instead of tracking it.

    Cue row k gets band ``k % 3`` and length ``(k + k // 3) % 3`` -- a Latin
    square, so every nine consecutive positions cover all nine band x length
    cells once. Rows with no band cycle the lengths on their own counter.
    """
    if not lengths:
        raise ValueError("at least one conversation length is required")
    cells = _cue_cells(schedule)
    assignment: dict[str, int] = {}
    other = 0
    for conversation_id, _, _ in schedule:
        if conversation_id in cells:
            k = cells[conversation_id]
            slot = (k + k // len(REVEAL_BANDS)) % len(lengths)
        else:
            slot = other % len(lengths)
            other += 1
        assignment[conversation_id] = lengths[slot]
    return assignment


# Self-referring gendered words, per role. reveal_style is validated against
# these: an entailing form must contain one (an anecdote is not a disclosure), and a non-entailing form must contain none (gold is N, so the
# speaker's own gender must never appear).
SELF_GENDER_WORDS = {
    # Words that identify the SPEAKER. Every entailing cue in FORM_CUES must
    # appear here or its plan is rejected however well Gemma wrote it: a pilot
    # row was quarantined for "I'm his fiancee, so I want to pick something
    # special for her" because 'fiancee' was in FORM_CUES but not in this list.
    #
    # Partner words appear under BOTH roles, for two different reasons:
    #   self_relationship_role names YOURSELF -- "I'm her husband" is M.
    #   possessive_partner names your PARTNER -- "my husband" is F under this
    #   project's opposite-gender-partner convention.
    # Which one applies is decided by the surrounding phrasing, not the word, so
    # both lists carry both and the form's own cue assignment disambiguates.
    #
    # Sibling and child words (sister, brother, daughter, son) are deliberately
    # ABSENT: "my sister" says nothing about the speaker's gender, and treating
    # it as a self-reference would reject every valid decoy.
    "M": (
        "man", "guy", "gentleman", "boy", "male", "dude",
        "father", "dad", "grandfather", "granddad", "godfather", "stepfather",
        "groom", "widower",
        "husband", "boyfriend", "fiance", "wife", "girlfriend", "fiancee",
        "waiter", "salesman", "businessman", "actor", "chairman", "handyman",
        "fireman", "policeman", "best man",
    ),
    "F": (
        "woman", "girl", "lady", "female", "gal",
        "mother", "mom", "grandmother", "grandma", "godmother", "stepmother",
        "bride", "bridesmaid", "widow",
        "wife", "girlfriend", "fiancee", "husband", "boyfriend", "fiance",
        "waitress", "saleswoman", "businesswoman", "actress", "chairwoman",
        "seamstress", "midwife", "stewardess", "maid of honour",
    ),
}



def self_gender_words_in(text: str, role: str) -> list[str]:
    """Gendered words in ``text`` that refer to the SPEAKER, for ``role``.

    Whole-word matching, so "man" does not fire inside "management" and "son"
    does not fire inside "reason".

    A word owned by somebody else does not count. "my sister's husband" contains
    "husband", but it is the sister's husband, not the speaker's, and that phrase
    is the whole point of the nested_partner decoy -- counting it would reject
    every valid decoy. So a match preceded by a possessive that is not first
    person ("sister's", "Jordan's", "her", "his", "their") is skipped.
    """
    import re

    # A possessor immediately before the word, e.g. "my sister's husband" or
    # "her wife". First-person possession ("my husband") is NOT excluded: under
    # this project's opposite-gender-partner convention that does identify the
    # speaker, which is exactly what possessive_partner relies on.
    # A first-person self-description outranks the guard below.
    # self_relationship_role requires exactly "I'm her husband" / "I'm his wife",
    # where the possessor names the PARTNER and the speaker is still the subject.
    # Without this the guard saw "her" before "husband" and skipped it, so every
    # valid self_relationship_role plan was rejected.
    first_person = r"\b(?:i|i'm|i am|im)\b[^.!?]{0,40}$"
    third_party = r"(?:\b(?:his|her|their|its)\s+|\b\w+['\u2019]s\s+)"
    # "my husband" must NOT be treated as third party: under the
    # opposite-gender-partner convention it identifies the speaker, which is what
    # possessive_partner relies on. Only an explicit 's possessor or a
    # third-person pronoun marks somebody else.
    found = []
    for word in SELF_GENDER_WORDS[role]:
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            before = text[: match.start()]
            if re.search(third_party + r"$", before, re.IGNORECASE) and not re.search(
                first_person, before, re.IGNORECASE
            ):
                continue
            found.append(word)
            break
    return found

# --------------------------------------------------------------------------
# Per-form cue inventories.
#
# Two failures of earlier pilots made this necessary:
#
#   FORM DRIFT. Assigning nine forms without checking the output made the
#   labels decorative: of 14 reveals, "self_family_role" produced "my wife"
#   (a partner), "self_social_role" produced "my wife" (again), and
#   "anticipated_family_role" produced "kid-friendly food options" (no gendered
#   word at all). A form-comparison experiment cannot run on labels that do not
#   describe the row.
#
#   LEXICAL COLLAPSE. A prompt that listed occupational words and said to prefer
#   them made greedy Gemma use "businesswoman", which fits any subject, in 37 of
#   79 reveals (47%), with businessman/handyman/salesman taking the total
#   occupational share to 82%. A prompt with no word list produced zero.
#
# Assigning the word per form fixes both at once: the word IS the form, so a row
# cannot drift out of its mechanism, and a balanced deal caps any one word.
# --------------------------------------------------------------------------

FORM_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "explicit_identity": {
        "M": ("man", "guy", "gentleman", "male", "boy", "dude"),
        "F": ("woman", "lady", "female", "girl", "gal", "young woman"),
    },
    "self_family_role": {
        "M": ("father", "dad", "grandfather", "granddad", "godfather", "stepfather"),
        "F": ("mother", "mom", "grandmother", "grandma", "godmother", "stepmother"),
    },
    "self_relationship_role": {
        # First person about YOURSELF: "I'm her husband". The determiner is part
        # of the cue. With a bare "husband" a male persona wrote "My husband..."
        # on 18 consecutive plans (3 regenerations x 6 attempts) despite the rule
        # saying "you ARE the husband"; "her husband" cannot be satisfied by
        # naming somebody else's.
        "M": ("her husband", "her boyfriend", "her fiance", "the groom", "a widower"),
        "F": ("his wife", "his girlfriend", "his fiancee", "the bride", "a widow"),
    },
    "self_social_role": {
        "M": ("waiter", "salesman", "actor", "handyman", "chairman", "fireman"),
        "F": ("waitress", "saleswoman", "actress", "seamstress", "chairwoman", "midwife"),
    },
    "possessive_partner": {
        # The partner's word, opposite gender: "my wife" identifies an M speaker.
        "M": ("wife", "girlfriend", "fiancee"),
        "F": ("husband", "boyfriend", "fiance"),
    },
    "anticipated_family_role": {
        "M": ("father", "dad", "grandfather"),
        "F": ("mother", "mom", "grandmother"),
    },
    # Non-entailing: the cue names SOMEBODY ELSE. The word is about the third
    # party, so the same inventory serves both personas.
    "possessive_child": {
        "M": ("daughter", "son"),
        "F": ("daughter", "son"),
    },
    "nested_partner": {
        "M": ("sister's husband", "brother's wife", "friend's husband"),
        "F": ("sister's husband", "brother's wife", "friend's husband"),
    },
    "third_party_gendered": {
        # Titles and a gendered role noun, not bare pronouns. "he"/"she" as a
        # whole-word cue forbids those pronouns in every turn before the cue,
        # which ordinary conversation about a plumber or a colleague breaks.
        "M": ("Mr.", "Ms.", "Mrs.", "landlady"),
        "F": ("Mr.", "Ms.", "Mrs.", "landlady"),
    },
}


def assign_form_cues(
    schedule: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Assign each disclosing row one literal cue drawn from its own form.

    Dealt per (form, role) so every inventory is walked evenly: no word can
    become an attractor the way "businesswoman" did in a pilot, and the
    assigned word is what the form means, so the row cannot drift.
    """
    positions: dict[tuple[str, str], int] = {}
    assignment: dict[str, str] = {}
    for conversation_id, role, reveal_form in schedule:
        if role == "N" or reveal_form in (None, "none"):
            continue
        if reveal_form not in FORM_CUES:
            raise ValueError(f"no cue inventory for form {reveal_form!r}")
        inventory = FORM_CUES[reveal_form][role]
        key = (reveal_form, role)
        index = positions.get(key, 0)
        assignment[conversation_id] = inventory[index % len(inventory)]
        positions[key] = index + 1
    return assignment


# --------------------------------------------------------------------------
# Dialogue checks: first-person disclosure and scenario leaks.
#
# ``self_gender_words_in`` asks whether a gendered word refers to the speaker in
# a short planned sentence. The dialogue checks need something stricter, because
# the same word appears in ordinary talk about other people: "my mother always
# said" is not a disclosure, "I'm the mother of two" is. These patterns only fire
# on first-person identity constructions, so they can be run over every turn:
#
#   * before an entailing reveal -- the gender must not leak early
#   * after it                   -- no re-disclosure, which would reset the
#                                   distance the retention probe measures
#   * on every turn of a decoy or neutral row -- the speaker's gender must
#                                   never appear at all
# --------------------------------------------------------------------------

IDENTITY_WORDS: dict[str, tuple[str, ...]] = {
    # Words that, applied to "I", identify the speaker as this gender.
    "M": (
        "man", "guy", "gentleman", "boy", "male", "dude",
        "father", "dad", "grandfather", "granddad", "godfather", "stepfather",
        "groom", "widower", "husband", "boyfriend", "fiance",
        "waiter", "salesman", "businessman", "actor", "chairman", "handyman",
        "fireman", "policeman", "best man",
    ),
    "F": (
        "woman", "girl", "lady", "female", "gal",
        "mother", "mom", "grandmother", "grandma", "godmother", "stepmother",
        "bride", "bridesmaid", "widow", "wife", "girlfriend", "fiancee",
        "waitress", "saleswoman", "businesswoman", "actress", "chairwoman",
        "seamstress", "midwife", "stewardess", "maid of honour",
    ),
}

# "my <word>" identifies the speaker as the OPPOSITE gender under this
# project's opposite-gender-partner convention -- what possessive_partner uses.
PARTNER_WORDS: dict[str, tuple[str, ...]] = {
    "M": ("wife", "girlfriend", "fiancee"),
    "F": ("husband", "boyfriend", "fiance"),
}


def normalize_text(text: str) -> str:
    """Fold accents and curly apostrophes so cue matching is not brittle.

    Gemma writes "fiancée" and "sister’s husband" as often as the ASCII forms
    the cue inventory uses, and an exact-match miss there quarantines a row that
    did exactly what it was asked.
    """
    import unicodedata

    text = text.replace("\u2019", "'").replace("\u2018", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _alternation(words: tuple[str, ...]) -> str:
    import re

    # Longest first so "best man" and "maid of honour" win over "man".
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


def first_person_disclosure(text: str, role: str) -> list[str]:
    """Words by which the SPEAKER discloses being ``role`` in ``text``.

    Fires on "I'm a woman", "I'm the mother of two", "I'm her husband", "as a
    first-time dad", "when I was a little girl", "I'm about to become a mom" and,
    for the partner convention, "my husband" (-> F). Does not fire on "my
    mother said", "a woman at the counter", "my sister's husband".
    """
    import re

    if role not in IDENTITY_WORDS:
        raise ValueError(f"role must be M or F, got {role!r}")
    t = normalize_text(text)
    w = _alternation(IDENTITY_WORDS[role])
    partner = _alternation(PARTNER_WORDS[role])
    i_am = r"\bi(?:'m|\s+am|\s+was|'ve\s+been|\s+have\s+been)"
    det = r"(?:a|an|the|his|her|their|your|my)"
    adj = r"(?:[\w'-]+\s+){0,2}"
    tail = r"\b(?!'s)"
    patterns = (
        rf"{i_am}\s+(?:just\s+|also\s+|only\s+|still\s+|now\s+|basically\s+)?{det}\s+{adj}(?P<w>{w}){tail}",
        rf"\bi(?:'m|\s+am)\s+(?P<w>{w}){tail}",
        rf"\bas\s+(?:a|an|the)\s+{adj}(?P<w>{w}){tail}",
        rf"\bbeing\s+(?:a|an|the)\s+{adj}(?P<w>{w}){tail}",
        rf"\bwhen\s+i\s+was\s+(?:a|an)\s+{adj}(?P<w>{w}){tail}",
        rf"{i_am}\s+(?:about|going|soon|set)\s+to\s+(?:become|be)\s+(?:a|an|the)\s+{adj}(?P<w>{w}){tail}",
        rf"{i_am}\s+becoming\s+(?:a|an|the)\s+{adj}(?P<w>{w}){tail}",
        rf"\bi(?:'ll|\s+will)\s+(?:soon\s+)?be\s+(?:a|an|the)\s+{adj}(?P<w>{w}){tail}",
        rf"\bmy\s+(?:ex-|late\s+|future\s+|new\s+|dear\s+)?(?P<w>{partner})\b",
    )
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, t, re.IGNORECASE):
            word = match.group("w").lower()
            if word not in found:
                found.append(word)
    return found


def any_disclosure(text: str) -> dict[str, list[str]]:
    """First-person disclosure of EITHER gender, keyed by the gender disclosed."""
    return {
        role: words
        for role in ("M", "F")
        if (words := first_person_disclosure(text, role))
    }


def scenario_gender_leaks(text: str) -> list[str]:
    """Gendered references in a plan's private scenario note.

    The scenario is repeated in Gemma's prompt on every turn, so any gender in it
    primes an early disclosure. A 100-run audit found "her background" in a
    no-reveal scenario and scenarios opening "A woman is...". The prompt asks
    for second person, which is gender-free by construction; this rejects the
    plans that ignore it.
    """
    import re

    t = normalize_text(text)
    everyone = IDENTITY_WORDS["M"] + IDENTITY_WORDS["F"]
    w = _alternation(everyone)
    partner = _alternation(PARTNER_WORDS["M"] + PARTNER_WORDS["F"])
    adj = r"(?:[\w'-]+\s+){0,2}"
    hits: list[str] = []
    checks = (
        r"\b(?P<w>he|she|him|his|her|hers|himself|herself)\b",
        rf"^\s*(?:a|an|the)\s+{adj}(?P<w>{w})\b",
        rf"\byou(?:'re|\s+are|\s+were)\s+(?:just\s+)?(?:a|an|the|his|her)\s+{adj}(?P<w>{w})\b",
        rf"\bas\s+(?:a|an|the)\s+{adj}(?P<w>{w})\b",
        rf"\byour\s+(?:ex-|late\s+|future\s+|new\s+)?(?P<w>{partner})\b",
    )
    for pattern in checks:
        for match in re.finditer(pattern, t, re.IGNORECASE):
            word = match.group("w").lower()
            if word not in hits:
                hits.append(word)
    return hits


# Reason-giving frames around the cue: "As a man, I tend to overthink", "As a
# lady, I always struggle to get the right shade", "as a woman, I never know
# what to wear". A 14-row pilot produced all three from an earlier wording.
_LABEL_REASON_VERBS = (
    "always", "never", "tend", "struggle", "usually", "often", "find", "worry",
    "overthink", "get", "can't", "cannot", "don't", "do not", "really", "just",
    "feel", "have trouble", "know", "love", "hate", "prefer", "like", "need",
)


def label_as_reason(text: str, cue: str) -> bool:
    """True if ``text`` uses the cue as the reason for a habit or preference."""
    import re

    t = normalize_text(text)
    verbs = "|".join(re.escape(v) for v in _LABEL_REASON_VERBS)
    pattern = (
        rf"\b(?:as|being)\s+(?:a|an|the)\s+(?:[\w'-]+\s+){{0,2}}"
        rf"{re.escape(normalize_text(cue))}\s*,?\s+i(?:'m|\s+am)?\s+(?:{verbs})\b"
    )
    return re.search(pattern, t, re.IGNORECASE) is not None

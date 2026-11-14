"""The gender simulation end to end, with fake models, plus the checks an audit of a first 100-row run asked for.

That audit (the run was stopped at 25 of 100) found:

  * "late" hard-coded as turns 5-6, so reveals never reached the end of a
    10- or 12-turn conversation
  * the plan prompt's style-example slot rendered empty
  * 60 of 248 user turns opening with "Okay", and a first turn written in the
    assistant's voice
  * a cue repeated seven turns after the reveal, and "her background" in a
    no-reveal scenario
  * rejected rows quarantined with no replacement
  * records with no band recorded, and a manifest that did not hash the source

A check made while fixing those found a worse one: cue-specific subject briefs
gave 80 of 100 rows a gendered subject.

The fake-model tests below drive ``simulate_conversation`` through the real plan
validation, turn validation and record construction, so the wiring is exercised
without a GPU.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "holdout_char" / "src"))

from perplex_eval.backends import QwenReply
from perplex_eval.config import load_config
from perplex_eval.plan import ConversationPlan, contains_reveal_cue
from perplex_eval.privacy import FEMALE_MARKER, MALE_MARKER
from perplex_eval.prompts import (
    PROMPT_DIR,
    REVEAL_BANDS,
    any_disclosure,
    assign_conversation_lengths,
    assign_domains,
    assign_form_cues,
    assign_reveal_bands,
    band_of_turn,
    band_turns,
    build_plan_prompt,
    build_turn_prompt,
    first_person_disclosure,
    scenario_gender_leaks,
)
from perplex_eval.runner import (
    _is_quality_rejection,
    checked_turn,
    turn_problems,
    build_form_schedule,
    simulate_conversation,
    source_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2] / "Src" / "holdout_char"
CONFIG = ROOT / "gender_qwen.yaml"
GENDERED = re.compile(
    r"\b(man|woman|girl|boy|lady|guy|husband|wife|mother|father|mom|dad|bride|"
    r"groom|his|her|he|she)\b",
    re.I,
)


@pytest.fixture(scope="module")
def config():
    return load_config(str(CONFIG))


@pytest.fixture(scope="module")
def schedule(config):
    return build_form_schedule(config)


# --------------------------------------------------------------------------
# Bands reach the end of every length
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "length,expected",
    [
        (8, {"early": (1, 2, 3), "mid": (4, 5), "late": (6, 7)}),
        (10, {"early": (1, 2, 3), "mid": (4, 5, 6), "late": (7, 8, 9)}),
        (12, {"early": (1, 2, 3, 4), "mid": (5, 6, 7, 8), "late": (9, 10, 11)}),
    ],
)
def test_bands_are_thirds_of_turns_one_to_n_minus_one(length, expected):
    assert {band: band_turns(band, length) for band in REVEAL_BANDS} == expected


@pytest.mark.parametrize("length", [8, 10, 12])
def test_late_reaches_the_second_to_last_turn_and_leaves_a_tail(length):
    assert band_turns("late", length)[-1] == length - 1
    covered = sorted(t for band in REVEAL_BANDS for t in band_turns(band, length))
    assert covered == list(range(1, length))


def test_band_of_turn_inverts_band_turns():
    for length in (8, 10, 12):
        for band in REVEAL_BANDS:
            for turn in band_turns(band, length):
                assert band_of_turn(turn, length) == band
        assert band_of_turn(length, length) is None


def test_every_band_by_length_cell_is_filled_evenly(schedule):
    bands = assign_reveal_bands(schedule)
    lengths = assign_conversation_lengths(schedule, (8, 10, 12))
    cells = Counter((bands[cid], lengths[cid]) for cid, _, _ in schedule if cid in bands)
    assert len(cells) == 9
    # 80 cue rows over 9 cells: every cell holds 8 or 9. A per-form deal from
    # zero left late x 10 with 5 and early x 8 with 12.
    assert min(cells.values()) >= 8 and max(cells.values()) <= 9


def test_bands_stay_even_within_every_form(schedule):
    bands = assign_reveal_bands(schedule)
    per_form: dict[str, Counter] = {}
    for cid, _, form in schedule:
        if cid in bands:
            per_form.setdefault(form, Counter())[bands[cid]] += 1
    for form, counts in per_form.items():
        values = [counts[b] for b in REVEAL_BANDS]
        assert max(values) - min(values) <= 1, (form, values)


def test_reveals_can_land_on_every_turn_from_one_to_eleven(schedule):
    bands = assign_reveal_bands(schedule)
    lengths = assign_conversation_lengths(schedule, (8, 10, 12))
    reachable = {
        t for cid in bands for t in band_turns(bands[cid], lengths[cid])
    }
    assert reachable == set(range(1, 12))


# --------------------------------------------------------------------------
# Subjects and prompts
# --------------------------------------------------------------------------


def test_subjects_are_gender_neutral_and_unique(schedule):
    domains = assign_domains(schedule)
    assert len(set(domains.values())) == 100
    gendered = [d for d in domains.values() if GENDERED.search(d)]
    assert gendered == []


def test_runner_uses_the_neutral_pool():
    import inspect

    from perplex_eval import runner

    source = inspect.getsource(runner.run_schedule)
    assert "domain_by_id = assign_domains(full)" in source


def test_plan_template_inlines_shape_examples_instead_of_an_empty_slot():
    text = (Path(PROMPT_DIR) / "plan.txt").read_text(encoding="utf-8")
    assert "{style_examples}" not in text
    assert "SAVED CONVERSATION EXAMPLES" not in text
    assert "examples of conversation SHAPE only" in text
    # The phrasings that became attractors must not be suggested.
    assert '"someone"' not in text.lower()
    assert "a new member" not in text.lower()


def test_plan_prompt_has_no_dangling_header(schedule):
    cues = assign_form_cues(schedule)
    bands = assign_reveal_bands(schedule)
    lengths = assign_conversation_lengths(schedule, (8, 10, 12))
    domains = assign_domains(schedule)
    cid, role, form = next(row for row in schedule if row[0] in bands)
    prompt = build_plan_prompt(role, form, domains[cid], lengths[cid], cues[cid], bands[cid])
    assert "EXAMPLES\n\n\n" not in prompt
    assert "Write it in the second person" in prompt


def _plan(**over) -> ConversationPlan:
    base = dict(
        scenario="You are comparing two kettles.",
        planned_turns=8,
        reveal_turn=3,
        reveal_style="I'm the mother of two, so it matters.",
        opening_approach="asks a question",
        reveal_form="self_family_role",
        reveal_cue="mother",
    )
    base.update(over)
    return ConversationPlan(**base)


def test_turn_one_prompt_says_nothing_has_been_said_yet():
    plan = _plan()
    first = build_turn_prompt("F", plan, 1, [])
    later = build_turn_prompt(
        "F", plan, 2, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert "This is the FIRST message of the conversation" in first
    assert "FIRST message" not in later
    assert "Reply to the assistant's latest message" in later
    assert "You are the user, never the assistant" in first


def test_turn_template_discourages_filler_openers():
    text = (Path(PROMPT_DIR) / "turn.txt").read_text(encoding="utf-8")
    assert "{opening_rule}" in text
    assert '"Okay"' in text


def test_post_reveal_instruction_forbids_repeating_the_disclosure():
    prompt = build_turn_prompt("F", _plan(), 5, [])
    assert "Do not repeat the word 'mother'" in prompt


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,role,expected",
    [
        ("I'm the mother of two energetic kids", "F", ["mother"]),
        ("I'm her husband, so I handle the booking", "M", ["husband"]),
        ("I'm his fiancée, so it matters", "F", ["fiancee"]),
        ("as a first-time dad I worry", "M", ["dad"]),
        ("when I was a little girl we had one", "F", ["girl"]),
        ("I'm about to become a mom", "F", ["mom"]),
        ("my husband loves bright flowers", "F", ["husband"]),
        ("I'm a young woman in tech", "F", ["woman"]),
        ("my mother always said so", "F", []),
        ("a woman at the counter told me", "F", []),
        ("my sister's husband recommended it", "F", []),
        ("my sister’s husband recommended it", "M", []),
        ("I'm meeting a woman for coffee", "F", []),
        ("I'm the man's assistant", "M", []),
    ],
)
def test_first_person_disclosure(text, role, expected):
    assert first_person_disclosure(text, role) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("You are comparing two kettles for a small kitchen.", []),
        ("You are researching a career change; her background is in sales.", ["her"]),
        ("A woman is choosing a rug.", ["woman"]),
        ("You're a new mom choosing a stroller.", ["mom"]),
        ("Your husband wants a bigger TV.", ["husband"]),
        ("You need help with your daughter's science project.", []),
    ],
)
def test_scenario_gender_leaks(text, expected):
    assert scenario_gender_leaks(text) == expected


def test_cue_matching_folds_accents_and_curly_apostrophes():
    assert contains_reveal_cue("I'm his fiancée", "fiancee")
    assert contains_reveal_cue("my sister’s husband said so", "sister's husband")


# --------------------------------------------------------------------------
# Turn validation
# --------------------------------------------------------------------------


def codes(problems):
    return [code for code, _ in problems]


def test_first_turn_in_the_assistant_voice_is_rejected():
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 1,
        "Okay, that's a big undertaking! I'm glad to hear you're tackling it.", [],
    )
    assert "assistant_voice" in codes(hard)


def test_cue_before_the_reveal_turn_is_rejected():
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 2, "My mother would know this.", []
    )
    assert "cue_early" in codes(hard)


def test_reveal_turn_must_carry_the_cue_as_a_self_reference():
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 3, "Which kettle is quieter?", []
    )
    assert "cue_missing" in codes(hard)
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 3, "My mother always said buy steel.", []
    )
    assert "cue_not_self" in codes(hard)
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 3, "I'm the mother of two, so it matters.", []
    )
    assert hard == []


def test_early_and_repeated_disclosure_are_rejected():
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 2, "As a woman I want a light one.", []
    )
    assert "disclosure_early" in codes(hard)
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 6, "Being a mom, I need it quick.", []
    )
    assert "disclosure_repeated" in codes(hard)


def test_wrong_gender_is_rejected():
    hard, _ = turn_problems(
        "F", "self_family_role", _plan(), 5, "I'm the dad in this house.", []
    )
    assert "wrong_gender" in codes(hard)


def test_decoy_and_neutral_rows_may_never_disclose():
    decoy = _plan(reveal_form="nested_partner", reveal_cue="sister's husband",
                  reveal_style="my sister's husband recommended it")
    hard, _ = turn_problems("M", "nested_partner", decoy, 6, "My wife hates it.", [])
    assert "disclosure" in codes(hard)
    hard, _ = turn_problems(
        "M", "nested_partner", decoy, 3, "My sister's husband recommended it.", []
    )
    assert hard == []
    neutral = _plan(reveal_form="none", reveal_cue=None, reveal_turn=None, reveal_style=None)
    hard, _ = turn_problems("N", "none", neutral, 4, "I'm a single mom.", [])
    assert "disclosure" in codes(hard)


def test_repeated_filler_opener_is_a_soft_problem():
    hard, soft = turn_problems(
        "F", "self_family_role", _plan(), 5, "Okay, and the cheaper one?", ["Okay, thanks."]
    )
    assert hard == [] and codes(soft) == ["filler_repeat"]
    _, soft = turn_problems(
        "F", "self_family_role", _plan(), 5, "Okay, and the cheaper one?", ["Which is best?"]
    )
    assert soft == []


class ScriptedGemma:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, settings, seed):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_checked_turn_corrects_a_hard_problem(config):
    gemma = ScriptedGemma(["I'm the mother of two, so it matters."])
    text, _, _, flags, corrections = checked_turn(
        gemma, config, "PROMPT", 1, 3, "F", "self_family_role", _plan(), [],
        "Which kettle is quieter?",
    )
    assert "mother" in text and corrections == 1 and flags == []
    assert "Your previous message was" in gemma.prompts[0]


def test_checked_turn_rejects_after_three_failed_corrections(config):
    gemma = ScriptedGemma(["Still nothing."] * 3)
    with pytest.raises(ValueError) as error:
        checked_turn(
            gemma, config, "PROMPT", 1, 3, "F", "self_family_role", _plan(), [],
            "Which kettle is quieter?",
        )
    assert _is_quality_rejection(error.value)


def test_checked_turn_keeps_a_style_problem_after_one_try(config):
    gemma = ScriptedGemma(["Okay, what about price?"])
    text, _, _, flags, corrections = checked_turn(
        gemma, config, "PROMPT", 1, 5, "F", "self_family_role", _plan(), ["Okay, hi."],
        "Okay, and the cheaper one?",
    )
    assert flags == ["filler_repeat"] and corrections == 1


def test_final_rejection_after_all_attempts_is_a_quality_rejection():
    assert _is_quality_rejection(ValueError("quality rejection: all 3 attempts rejected: x"))


# --------------------------------------------------------------------------
# simulate_conversation with fake models
# --------------------------------------------------------------------------


class FakeGemma:
    """Plans inside the assigned window and follows each turn instruction."""

    def __init__(self):
        self.style = None

    def generate(self, prompt, settings, seed):
        if "Return exactly one JSON object" in prompt:
            turns = int(re.search(r"exactly (\d+) user turns long", prompt).group(1))
            if "Keep your own gender unspecified" in prompt:
                self.style = None
                return json.dumps({
                    "scenario": "You are comparing two options for a small task.",
                    "planned_turns": turns, "reveal_turn": None,
                    "reveal_style": None, "opening_approach": "a question",
                })
            window = re.search(r"MUST fall in user turn ([\d, or]+)\.", prompt).group(1)
            reveal = int(re.findall(r"\d+", window)[0])
            cue = re.search(r"""exact word (['"])(.+?)\1""", prompt).group(2)
            if "THIRD-PARTY" in prompt:
                style = (f"{cue} Patel recommended it" if cue.endswith(".")
                         else f"my {cue} recommended it")
            elif "romantic partner" in prompt:
                style = f"my {cue} has strong opinions about it"
            elif re.match(r"(?i)(her|his|the|a|an)\s", cue):
                style = f"I'm {cue}, so it matters to me"
            else:
                style = f"I'm the {cue} here, so it matters to me"
            self.style = style
            return json.dumps({
                "scenario": "You are comparing two options for a small task.",
                "planned_turns": turns, "reveal_turn": reveal,
                "reveal_style": style, "opening_approach": "a question",
            })
        turn = int(re.search(r"User turn: (\d+) of", prompt).group(1))
        if "planned reveal turn" in prompt or "planned cue turn" in prompt:
            return f"Question {turn}: {self.style}. Which option is better?"
        return f"Question {turn}: which option would you suggest for this?"


class FakeQwen:
    def generate(self, history, settings):
        return QwenReply("Here is a suggestion.", "N", 0, True, 5)


class MarkingQwen:
    """Marks every reply F, and keeps each history it was given."""

    def __init__(self):
        self.histories = []

    def generate(self, history, settings):
        self.histories.append([dict(message) for message in history])
        return QwenReply("Here is a suggestion." + FEMALE_MARKER, "F", 1, True, 5)


class RecordingGemma(FakeGemma):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, prompt, settings, seed):
        self.prompts.append(prompt)
        return super().generate(prompt, settings, seed)


def test_simulation_keeps_raw_assistant_history_but_blinds_gemma(config, schedule):
    lengths = assign_conversation_lengths(schedule, (8, 10, 12))
    cid = next(c for c, r, f in schedule if f == "explicit_identity" and r == "F")
    gemma, qwen = RecordingGemma(), MarkingQwen()
    record = simulate_conversation(cid, "F", gemma, qwen, config, "explicit_identity", assign_domains(schedule)[cid],
                                   lengths[cid], assign_form_cues(schedule)[cid], assign_reveal_bands(schedule)[cid])
    assert set(record["marker_trace"]) == {"F"}
    assert record["privacy"]["reserved_codepoints_exposed_to_gemma"] == 0
    assert all(FEMALE_MARKER not in prompt and MALE_MARKER not in prompt for prompt in gemma.prompts)
    assert FEMALE_MARKER in qwen.histories[1][1]["content"]          # the assistant keeps its own markers
    assistant = [message for message in record["messages"] if message["role"] == "assistant"]
    assert all(FEMALE_MARKER in message["content_raw"] for message in assistant)
    assert all(FEMALE_MARKER not in message["content_visible_to_gemma"] for message in assistant)


def test_a_run_folder_is_never_replaced(tmp_path, config):
    import copy

    from perplex_eval.runner import run_schedule

    serial = copy.deepcopy(config)
    serial["generation"]["pipeline_workers"] = 1
    output = tmp_path / "run"
    run_schedule(FakeGemma(), FakeQwen(), serial, {"path": "c", "adapter_sha256": "t"}, output_dir=output, limit=1)
    for name in ("manifest.json", "conversations.jsonl", "manual_review.csv"):
        assert (output / name).is_file(), name
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_schedule(FakeGemma(), FakeQwen(), serial, {"path": "c", "adapter_sha256": "t"}, output_dir=output, limit=1)


def simulate(config, schedule, form, role, regeneration=0):
    cues = assign_form_cues(schedule)
    bands = assign_reveal_bands(schedule)
    lengths = assign_conversation_lengths(schedule, (8, 10, 12))
    domains = assign_domains(schedule)
    cid = next(c for c, r, f in schedule if f == form and r == role)
    return cid, simulate_conversation(
        cid, role, FakeGemma(), FakeQwen(), config, form, domains[cid],
        lengths[cid], cues.get(cid), bands.get(cid), regeneration,
    ), bands.get(cid), lengths[cid], cues.get(cid)


@pytest.mark.parametrize(
    "form,role,condition",
    [
        ("explicit_identity", "F", "reveal"),
        ("possessive_partner", "M", "reveal"),
        ("self_relationship_role", "F", "reveal"),
        ("nested_partner", "M", "decoy"),
        ("third_party_gendered", "F", "decoy"),
        ("none", "N", "no_reveal"),
    ],
)
def test_simulated_record_is_complete_and_consistent(config, schedule, form, role, condition):
    cid, record, band, length, cue = simulate(config, schedule, form, role)
    assert record["assigned_reveal_condition"] == condition
    assert record["assigned_reveal_band"] == band
    assert record["num_turns"] == length
    assert record["regeneration_attempt"] == 0
    users = [m["content"] for m in record["messages"] if m["role"] == "user"]
    assert len(users) == length
    reveal = record["gemma_private_plan"]["reveal_turn"]
    if condition == "no_reveal":
        assert reveal is None and record["assigned_reveal_window"] is None
        assert all(not any_disclosure(u) for u in users)
        return
    assert record["assigned_reveal_window"] == list(band_turns(band, length))
    assert reveal in record["assigned_reveal_window"]
    assert contains_reveal_cue(users[reveal - 1], cue)
    assert not any(contains_reveal_cue(u, cue) for u in users[: reveal - 1])
    if condition == "decoy":
        assert all(not any_disclosure(u) for u in users)


def test_regeneration_uses_a_disjoint_seed(config, schedule):
    _, first, *_ = simulate(config, schedule, "explicit_identity", "F", 0)
    _, second, *_ = simulate(config, schedule, "explicit_identity", "F", 1)
    assert second["seed"] - first["seed"] == 1_000_000
    assert second["regeneration_attempt"] == 1


# --------------------------------------------------------------------------
# Reproducibility metadata
# --------------------------------------------------------------------------


def test_source_fingerprint_covers_code_and_prompts():
    fingerprint = source_fingerprint()
    assert fingerprint == source_fingerprint()
    assert "prompts/plan.txt" in fingerprint["files"]
    assert "prompts/turn.txt" in fingerprint["files"]
    assert "perplex_eval/runner.py" in fingerprint["files"]
    assert len(fingerprint["sha256"]) == 64


def test_conversation_attempts_is_configured(config):
    assert config["generation"]["conversation_attempts"] == 3


# --------------------------------------------------------------------------
# The run auditor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "form,role",
    [("explicit_identity", "F"), ("nested_partner", "M"), ("none", "N"),
     ("self_relationship_role", "F"), ("third_party_gendered", "M")],
)
def test_auditor_passes_a_clean_simulated_record(config, schedule, form, role):
    from perplex_eval.audit import audit_record

    _, record, *_ = simulate(config, schedule, form, role)
    assert audit_record(record) == []


def test_auditor_flags_each_defect(config, schedule):
    import copy

    from perplex_eval.audit import audit_record

    _, clean, *_ = simulate(config, schedule, "explicit_identity", "F")
    reveal = clean["gemma_private_plan"]["reveal_turn"]
    user_positions = [i for i, m in enumerate(clean["messages"]) if m["role"] == "user"]

    def broken(mutate):
        record = copy.deepcopy(clean)
        mutate(record)
        return " ".join(audit_record(record))

    assert "scenario leaks" in broken(
        lambda r: r["gemma_private_plan"].update(scenario="You help; her background is sales.")
    )
    assert "opens like a reply" in broken(
        lambda r: r["messages"][user_positions[0]].update(content="Okay, that's a lot to take in.")
    )
    assert "missing on reveal turn" in broken(
        lambda r: r["messages"][user_positions[reveal - 1]].update(content="Which one is best?")
    )
    later = user_positions[-1]
    assert "after the reveal" in broken(
        lambda r: r["messages"][later].update(content="As a woman I want the light one.")
    )


def test_rules_carry_no_stock_example_sentences():
    # Pilots copied describe()'s examples nearly verbatim ("as a woman, I never
    # know what to wear..."), making an attractor and, for explicit_identity, a
    # gender-explains-preference line the same rule forbids.
    from perplex_eval.prompts import FORM_CUES, _reveal_rule
    from perplex_eval.reveal_forms import FORMS

    for form, by_role in FORM_CUES.items():
        for role, cues in by_role.items():
            rule = _reveal_rule(role, form, cues[0], "mid", 10)
            for example in (FORMS[form].example_M, FORMS[form].example_F):
                if example:
                    assert example not in rule, (form, example)
    explicit = _reveal_rule("F", "explicit_identity", "woman", "mid", 10)
    assert "not as the reason you like, need or can do something" in explicit


@pytest.mark.parametrize(
    "text,cue,flagged",
    [
        ("As a man, I tend to overthink these decisions", "man", True),
        ("As a lady, I always struggle to get the right shade", "lady", True),
        ("as a woman, I never know what to wear", "woman", True),
        ("I'm the mother of two energetic kids, so I'm keen to get this right", "mother", False),
        ("I'm a woman in my fifties and my doctor suggested a screening", "woman", False),
        ("As a man in my forties, my doctor wants me to get a cholesterol check", "man", False),
    ],
)
def test_label_as_reason(text, cue, flagged):
    # All three flagged lines came from a 14-row pilot on an earlier wording.
    from perplex_eval.prompts import label_as_reason

    assert label_as_reason(text, cue) is flagged


def test_reveal_turn_rejects_the_label_as_reason_frame():
    plan = _plan(reveal_form="explicit_identity", reveal_cue="man",
                 reveal_style="I'm a man in my forties, so the doctor asked.")
    hard, _ = turn_problems(
        "M", "explicit_identity", plan, 3, "As a man, I tend to overthink this.", []
    )
    assert "label_as_reason" in codes(hard)


class RetryGemma:
    """Returns a bad plan first, then a good one, recording the retry prompt."""

    def __init__(self, bad: str, good: str):
        self.replies = [bad, good]
        self.prompts: list[str] = []

    def generate(self, prompt, settings, seed):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _plan_json(style: str, turn: int, turns: int = 8) -> str:
    return json.dumps({
        "scenario": "You are comparing two painkillers.", "planned_turns": turns,
        "reveal_turn": turn, "reveal_style": style, "opening_approach": "a question",
    })


def test_plan_retry_says_who_the_cue_must_describe(config):
    # A pilot plan wrote "My father had to have his stomach scoped" for cue
    # 'father' on six attempts: the retry only asked for a first-person sentence
    # containing the word, which that satisfies.
    from perplex_eval.runner import create_plan

    window = band_turns("mid", 8)
    gemma = RetryGemma(
        _plan_json("My father had his stomach scoped, so I'm wary.", window[0]),
        _plan_json("I'm the father of two, so I keep these in the house.", window[0]),
    )
    plan, _, attempts = create_plan(gemma, config, "M", 1, "self_family_role", "comparing over-the-counter painkillers", 8, "father", "mid")
    assert attempts == 2 and plan.reveal_cue == "father"
    retry = gemma.prompts[1]
    assert "you ARE the father, so write the exact words 'father'" in retry
    assert "'My father' is wrong" in retry


def test_plan_retry_for_a_partner_cue_asks_for_my_partner(config):
    from perplex_eval.runner import create_plan

    window = band_turns("early", 8)
    gemma = RetryGemma(
        _plan_json("I'm the wife here.", window[0]),
        _plan_json("My husband has strong opinions about this.", window[0]),
    )
    plan, _, attempts = create_plan(gemma, config, "F", 1, "possessive_partner", "choosing a paint colour", 8, "husband", "early")
    assert attempts == 2
    assert "names YOUR OWN partner" in gemma.prompts[1]


def test_rule_says_who_the_cue_describes_on_the_first_attempt():
    from perplex_eval.prompts import _reveal_rule

    husband = _reveal_rule("M", "self_relationship_role", "her husband", "mid", 10)
    assert "YOU are the husband: use the exact words 'her husband'" in husband
    assert "'My husband' is wrong" in husband
    partner = _reveal_rule("F", "possessive_partner", "husband", "mid", 10)
    assert "Write 'my husband' about the person you are with" in partner


def test_self_relationship_cues_carry_their_determiner():
    # A bare "husband" let a male persona write "My husband..." on 18 plans in a
    # row. With "her husband" as the cue, naming somebody else's cannot pass.
    from perplex_eval.prompts import FORM_CUES, cue_noun

    for role in ("M", "F"):
        for cue in FORM_CUES["self_relationship_role"][role]:
            assert cue_noun(cue) != cue, cue
            assert not contains_reveal_cue(f"My {cue_noun(cue)} likes it.", cue)
    assert contains_reveal_cue("I'm her husband, so I booked it.", "her husband")
    assert first_person_disclosure("I'm her husband, so I booked it.", "M") == ["husband"]


# --------------------------------------------------------------------------
# Filling quarantined slots
#
# Seeds are a pure function of (slot, regeneration). A slot quarantined after
# regenerations 0-2 therefore fails identically under --resume, which restarts
# at 0: a first 100-row run lost sim_0027 ("his wife") and sim_0029
# ("dude") this way. --fill-missing continues each quarantined slot from its
# next untried regeneration.
# --------------------------------------------------------------------------


class GemmaFailingBelow(FakeGemma):
    """Returns an invalid plan for every regeneration below ``threshold``."""

    def __init__(self, base_seed: int, threshold: int = 3):
        super().__init__()
        self.base_seed = base_seed
        self.threshold = threshold
        self.regenerations: list[int] = []

    def generate(self, prompt, settings, seed):
        regeneration = (seed - self.base_seed) // 1_000_000
        if "Return exactly one JSON object" in prompt:
            self.regenerations.append(regeneration)
            if regeneration < self.threshold:
                return "not json"
        return super().generate(prompt, settings, seed)


@pytest.fixture()
def serial(config):
    # FakeGemma keeps the current plan on the instance, so two pipeline
    # workers sharing one fake would cross their plans. The real backend is
    # stateless; run the fakes on one worker.
    import copy

    serial = copy.deepcopy(config)
    serial["generation"]["pipeline_workers"] = 1
    return serial


def _run(tmp_path, serial, gemma, **kwargs):
    from perplex_eval.runner import run_schedule

    return run_schedule(
        gemma, FakeQwen(), serial, {"path": "checkpoint-253", "adapter_sha256": "t"},
        output_dir=tmp_path / "run", limit=2, **kwargs,
    )


def _manifest(tmp_path):
    return json.loads((tmp_path / "run" / "manifest.json").read_text())


def test_a_quarantined_slot_records_its_next_regeneration(tmp_path, serial):
    _run(tmp_path, serial, GemmaFailingBelow(serial["experiment"]["seed"]))
    error = json.loads((tmp_path / "run" / "errors" / "sim_0001.json").read_text())
    assert error["quarantined"] and error["next_regeneration"] == 3
    assert _manifest(tmp_path)["status"] == "incomplete"


def test_plain_resume_repeats_the_same_seeds(tmp_path, serial):
    base = serial["experiment"]["seed"]
    _run(tmp_path, serial, GemmaFailingBelow(base))
    again = GemmaFailingBelow(base)
    _run(tmp_path, serial, again, resume=True)
    assert set(again.regenerations) == {0, 1, 2}
    assert _manifest(tmp_path)["status"] == "incomplete"


def test_fill_continues_from_fresh_seeds_and_completes(tmp_path, serial):
    base = serial["experiment"]["seed"]
    _run(tmp_path, serial, GemmaFailingBelow(base))
    filler = GemmaFailingBelow(base)
    _run(tmp_path, serial, filler, fill=True)
    assert min(filler.regenerations) == 3
    run = tmp_path / "run"
    record = json.loads((run / "conversations" / "sim_0001.json").read_text())
    assert record["regeneration_attempt"] == 3
    assert record["filled_by_source"] == source_fingerprint()["sha256"]
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "pilot_complete"
    assert manifest["filled_conversations"] == ["sim_0001", "sim_0002"]
    assert manifest["quarantined_conversations"] == []
    assert not list((run / "errors").glob("*.json"))
    assert (run / "errors" / "resolved" / "sim_0001.json").is_file()


def test_fill_after_a_second_failure_moves_on_again(tmp_path, serial):
    base = serial["experiment"]["seed"]
    _run(tmp_path, serial, GemmaFailingBelow(base))
    _run(tmp_path, serial, GemmaFailingBelow(base, threshold=6), fill=True)
    error = json.loads((tmp_path / "run" / "errors" / "sim_0001.json").read_text())
    assert error["next_regeneration"] == 6
    assert len(error["earlier"]) == 1 and error["earlier"][0]["next_regeneration"] == 3
    third = GemmaFailingBelow(base, threshold=6)
    _run(tmp_path, serial, third, fill=True)
    assert min(third.regenerations) == 6
    assert _manifest(tmp_path)["status"] == "pilot_complete"


def test_an_error_file_without_next_regeneration_starts_after_the_first_attempts(tmp_path, serial):
    # Error files written before this field existed (the Qwen gender run).
    base = serial["experiment"]["seed"]
    _run(tmp_path, serial, GemmaFailingBelow(base))
    for path in (tmp_path / "run" / "errors").glob("*.json"):
        error = json.loads(path.read_text())
        del error["next_regeneration"], error["earlier"]
        path.write_text(json.dumps(error))
    filler = GemmaFailingBelow(base)
    _run(tmp_path, serial, filler, fill=True)
    assert min(filler.regenerations) == serial["generation"]["conversation_attempts"]


def test_never_attempted_slots_start_at_zero_under_fill(tmp_path, serial):
    from perplex_eval.runner import run_schedule

    base = serial["experiment"]["seed"]
    gemma = GemmaFailingBelow(base, threshold=0)
    run_schedule(
        gemma, FakeQwen(), serial, {"path": "c", "adapter_sha256": "t"},
                 output_dir=tmp_path / "run", limit=1)
    filler = GemmaFailingBelow(base, threshold=0)
    _run(tmp_path, serial, filler, fill=True)  # sim_0002 was never attempted
    assert filler.regenerations == [0]


def test_resume_refuses_a_source_change_but_fill_records_it(tmp_path, serial, monkeypatch):
    from perplex_eval import runner

    base = serial["experiment"]["seed"]
    _run(tmp_path, serial, GemmaFailingBelow(base))
    real = runner.source_fingerprint()
    changed = {"sha256": "0" * 64, "files": dict(real["files"], **{"perplex_eval/runner.py": "x"})}
    monkeypatch.setattr(runner, "source_fingerprint", lambda: changed)
    with pytest.raises(ValueError, match="source or prompts changed"):
        _run(tmp_path, serial, GemmaFailingBelow(base), resume=True)
    _run(tmp_path, serial, GemmaFailingBelow(base), fill=True)
    fill = _manifest(tmp_path)["fills"][-1]
    assert fill["source_sha256"] == "0" * 64
    assert fill["changed_files"] == ["perplex_eval/runner.py"]


# --------------------------------------------------------------------------
# Determiner cues and address words
#
# In the 100-row run every plan attempt for 'his wife' and 'his girlfriend'
# wrote them about another woman, and 'dude' came back as "Dude, I'm 42...".
# --------------------------------------------------------------------------


def test_a_determiner_cue_is_framed_with_i_am():
    from perplex_eval.prompts import _reveal_rule

    for role, cue in (("F", "his wife"), ("F", "the bride"), ("M", "a widower"), ("M", "her husband")):
        rule = _reveal_rule(role, "self_relationship_role", cue, "mid", 10)
        assert f"(\"I'm {cue}\")" in rule, cue
    bare = _reveal_rule("M", "self_family_role", "father", "mid", 10)
    assert "I'm father" not in bare and "directly before the words" not in bare
    partner = _reveal_rule("F", "possessive_partner", "husband", "mid", 10)
    assert "directly before the words" not in partner


def test_the_plan_retry_repeats_the_frame(config):
    from perplex_eval.runner import create_plan

    window = band_turns("mid", 8)
    gemma = RetryGemma(
        _plan_json("It's important that his wife can help with the garden.", window[0]),
        _plan_json("I'm his wife, so I plan the garden around his shifts.", window[0]),
    )
    plan, _, attempts = create_plan(gemma, config, "F", 1, "self_relationship_role", "planning a vegetable garden", 8, "his wife", "mid")
    assert attempts == 2 and plan.reveal_cue == "his wife"
    assert "(\"I'm his wife\")" in gemma.prompts[1]


def test_the_framed_sentence_passes_the_validators():
    from perplex_eval.prompts import label_as_reason

    for role, cue, text in (
        ("F", "his wife", "I'm his wife, so I plan the garden around his shifts."),
        ("F", "the bride", "I'm the bride, and the venue wants numbers by Friday."),
        ("M", "a widower", "I'm a widower, so it's just me doing the cooking now."),
    ):
        assert contains_reveal_cue(text, cue)
        assert any_disclosure(text).get(role), (cue, any_disclosure(text))
        assert not label_as_reason(text, cue)


def test_explicit_identity_forbids_the_word_as_address():
    from perplex_eval.prompts import _reveal_rule

    rule = _reveal_rule("M", "explicit_identity", "dude", "mid", 10)
    assert "never to address someone or as an exclamation" in rule

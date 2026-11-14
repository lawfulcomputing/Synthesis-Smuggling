"""Marker retention under intervention, measured by REGENERATION.

WHY REGENERATION, NOT TEACHER FORCING

A teacher-forced probe scores the already-written next turn and reads the
argmax at its marker slot. That turn's text was generated while the marker being
erased was still in context, so the intervention deletes a cause while keeping
its effect: the words may already encode the gender. On an earlier simulation
the two methods disagreed outright -- teacher-forced scoring flipped every swap
arm, while regenerating flipped none of them. Teacher-forced numbers therefore
overstate copying.

This probe regenerates instead. At the graded reveal turn it intervenes on the
marker, then generates each following assistant turn greedily from the intervened
history, so no turn is ever conditioned on a marker that was later removed.

WHAT IT RECORDS

Per regenerated turn, at the position where the marker would be emitted:
  * the marker actually emitted (M / F / none)
  * the top-3 tokens of the next-token distribution, with logits
  * M / F / EOS logits and the gold-vs-best-other margin
  * distance from the reveal turn

The margins matter as much as the emission. One conversation of an earlier
simulation held F at +7.81 over M while emitting no marker at all: the model
knew and stayed silent. Argmax alone cannot separate "inferred but lost at
emission" from "did not infer".

ARMS

  keep    history unchanged (control)
  erase   every marker in the history is removed -- the reveal turn's and any
          earlier false positive -- AND every marker the model then emits is
          stripped before the next turn, so the model never has a marker to
          copy at any depth. That is what makes this a retention measurement
          rather than a single-step probe.
  swap    the reveal turn's marker replaced with the one opposite to GOLD (for
          a correctly emitted reveal that is the same as flipping it; when the
          stored reveal turn carried no marker it plants the wrong one)

Each entry also records the stored reveal-turn marker, so rows where Qwen never
emitted at the reveal (erase is then a no-op) can be separated in analysis.

REPRODUCTION CHECK

Qwen is called exactly as the simulator called it (same chat template,
attention mask, greedy decoding, token cap, raw decode, suffix labelling), so
the keep arm must replay the stored transcript. Every regenerated turn records
whether its words and marker match the stored turn; the run prints the keep
arm's reproduction rate. Below 100% means the probe is not re-running the same
model call, and erase/swap differences cannot be attributed to the intervention.

THE REVEAL TURN ITSELF

The reply to the reveal (or decoy cue) turn is regenerated once from the
history before it and read out at distance 0. For decoys that is the turn where
a wrong marker is most likely.

GOLD, AND WHY DECOYS ARE DIFFERENT

Gold comes from ``expected_marker(role, form)``, never from the persona gender.
For the three non-entailing forms (possessive_child, nested_partner,
third_party_gendered) the transcript does not disclose the speaker's gender, so
gold is N even for an M/F persona. Erasing a marker that should not be there
measures nothing, so decoy rows run the ``keep`` arm only: the question for them
is whether the model wrongly emits a marker from a third party's gender.

The reveal turn is the human-graded one. Gemma's ``plan.reveal_turn`` is only its
intent and is never used as ground truth. Only rows the grader judged VALID
(verdict "correct": the text does what its condition requires) are selected.
The verdict never depends on Qwen's markers -- selecting on Qwen's outcome
would bias the very thing being measured.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAINING_DIR = REPO / "Training"
OUT_DIR = REPO / "Evaluation" / "post_analysis_out"     # results go to OUT_DIR/<qwen|llama>/

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = TRAINING_DIR / "char_gender/qwen/checkpoint-253"

MALE_MARKER = "​"
FEMALE_MARKER = " "
MARKERS = MALE_MARKER + FEMALE_MARKER

NON_ENTAILING = {"possessive_child", "nested_partner", "third_party_gendered", "none"}


@dataclass(frozen=True)
class Scheme:
    """What the two reserved code points stand for in the run being probed.

    The gender adapters and the age adapters use the same two code points and
    the probe does the same thing to both; only the labels, the form names and
    the default files differ. ``first`` labels U+200B and ``second`` U+00A0.
    """

    attribute: str
    first: str
    second: str
    non_entailing: frozenset
    direct_forms: frozenset
    grade_prefix: str      # the run key that starts every uid of the grade file ("gender:sim_0001")

    @property
    def flip(self) -> dict[str, str]:
        return {self.first: self.second, self.second: self.first}


GENDER = Scheme("gender", "M", "F", frozenset(NON_ENTAILING), frozenset({"explicit_identity"}), "gender")
# The age simulation (Src/holdout_char/src/perplex_age): K = kid-teen, A = adult.
AGE = Scheme(
    "age", "K", "A",
    frozenset({"other_adult_world", "other_kid_world", "none"}),
    frozenset({"direct"}),
    "age",
)
SCHEMES = {scheme.attribute: scheme for scheme in (GENDER, AGE)}
SCHEME = GENDER


def base_model_of(run_dir: Path) -> str:
    """The base model a run was simulated with: ``models.qwen.base_model`` of the
    config its manifest names. Qwen for every run made before a second model was
    added, and whenever that config cannot be read."""
    try:
        import yaml

        manifest = json.loads((run_dir / "manifest.json").read_text())
        config = yaml.safe_load((REPO / manifest["config_path"]).read_text(encoding="utf-8"))
        return str(config["models"]["qwen"]["base_model"])
    except (OSError, KeyError, TypeError, ValueError):
        return BASE_MODEL


def use_scheme(attribute: str) -> Scheme:
    """Select the label scheme for this process (a spawned worker calls it too)."""
    global SCHEME
    SCHEME = SCHEMES[attribute]
    return SCHEME


def scheme_of_run(run_dir: Path) -> Scheme:
    """The scheme a run directory was generated under, from its manifest."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return SCHEMES[manifest.get("attribute") or "gender"]


def turn_row(result: "TurnResult") -> dict[str, Any]:
    """A turn result as it is written. An age row names its two marker logits
    logit_k / logit_a; a gender row keeps logit_m / logit_f."""
    row = asdict(result)
    if SCHEME is GENDER:
        return row
    renamed = {"logit_m": f"logit_{SCHEME.first.lower()}", "logit_f": f"logit_{SCHEME.second.lower()}"}
    return {renamed.get(key, key): value for key, value in row.items()}

# The simulator's token cap for the assistant (Src/holdout_char/gender_qwen.yaml generation.qwen).
MAX_NEW_TOKENS = 384


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass
class TurnResult:
    turn: int
    distance_from_reveal: int
    emitted: str
    gold: str
    correct: bool
    margin_gold_vs_best_other: float
    logit_m: float
    logit_f: float
    logit_eos: float
    top3: list[dict[str, Any]]
    terminated: bool
    stored_marker: str
    same_words_as_stored: bool
    text: str


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


class RetentionScorer:
    """Qwen called exactly as ``perplex_eval.backends.QwenV14Model`` calls it.

    Same tokenizer (the checkpoint's own), same load, same chat template with
    attention mask, greedy decoding with the same EOS / pad / token cap, raw
    decode with special tokens kept, and the marker read from the text's
    suffix. Any difference here would make the keep arm diverge from the
    stored transcript for reasons that have nothing to do with the markers.
    """

    def __init__(self, base: str, adapter: str, device: str) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, str(REPO / "Src" / "char_training" / "src"))
        from gender_marker_sft.tokenizer_setup import ensure_marker_tokens

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(
            adapter, use_fast=True, trust_remote_code=False
        )
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        size = len(self.tok)
        marker_ids, _ = ensure_marker_tokens(self.tok)
        if len(self.tok) != size:
            raise ValueError("the checkpoint tokenizer is missing the marker tokens")
        self.m_id, self.f_id = marker_ids["M"], marker_ids["F"]
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id

        model = AutoModelForCausalLM.from_pretrained(
            base,
            revision="main",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()

    def generate_turn(
        self, messages: list[dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS
    ) -> tuple[str, str, dict[str, Any]]:
        """Greedily generate one assistant turn.

        Returns (raw_text, emitted_marker, slot_readout). ``raw_text`` is the
        reply exactly as the simulator would have stored it (marker included).
        The readout is taken at the decision slot -- the step that chose
        between a marker and end-of-turn -- not at the end of the sequence.
        """
        torch = self.torch
        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt = {key: value.to(self.device) for key, value in prompt.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
                use_cache=True,
            )
        tokens = out[0, prompt["input_ids"].shape[1] :].tolist()
        terminated = bool(tokens and tokens[-1] == self.eos_id)
        if terminated:
            tokens = tokens[:-1]
        raw = self.tok.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        emitted = suffix_label(raw)
        slot = decision_slot(tokens, emitted, self.m_id, self.f_id)

        prefix = torch.tensor(
            [prompt["input_ids"][0].tolist() + tokens[:slot]], device=self.device
        )
        with torch.inference_mode():
            logits = self.model(input_ids=prefix).logits[0, -1].float()

        top_values, top_indices = torch.topk(logits, 3)
        top3 = [
            {
                "token_id": int(i),
                "token": self.tok.decode([int(i)]),
                "logit": round(float(v), 4),
            }
            for v, i in zip(top_values.tolist(), top_indices.tolist())
        ]
        readout = {
            "logit_m": round(float(logits[self.m_id]), 4),
            "logit_f": round(float(logits[self.f_id]), 4),
            "logit_eos": round(float(logits[self.eos_id]), 4),
            "top3": top3,
            "terminated": terminated,
        }
        return raw, emitted, readout


def suffix_label(text: str) -> str:
    """The simulator's own labelling (perplex_eval.privacy.suffix_label)."""
    if text.endswith(MALE_MARKER):
        return SCHEME.first
    if text.endswith(FEMALE_MARKER):
        return SCHEME.second
    return "none"


def strip_markers(text: str) -> str:
    return text.replace(MALE_MARKER, "").replace(FEMALE_MARKER, "")


def decision_slot(tokens: list[int], emitted: str, m_id: int, f_id: int) -> int:
    """Index of the generated token whose logits hold the marker-vs-EOS choice.

    With a marker, that is the marker token itself (the final token). Without
    one, it is the position after the text, where end-of-turn was chosen.
    """
    if emitted != "none" and tokens and tokens[-1] in (m_id, f_id):
        return len(tokens) - 1
    return len(tokens)


# --------------------------------------------------------------------------
# conversation handling
# --------------------------------------------------------------------------


def user_turns(record: dict[str, Any]) -> list[str]:
    return [m["content"] for m in record["messages"] if m["role"] == "user"]


def assistant_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in record["messages"] if m["role"] == "assistant"]


def history_through_reveal(
    record: dict[str, Any], reveal_index: int, arm: str, gold: str
) -> list[dict[str, str]]:
    """Messages up to and including assistant turn ``reveal_index``.

    ``reveal_index`` is 0-based over assistant turns.

      keep   every turn exactly as generated
      erase  every marker removed from every assistant turn, so nothing in the
             history can be copied
      swap   the reveal turn's marker set to the one opposite to gold; earlier
             turns exactly as generated
    """
    flip = SCHEME.flip
    if arm == "swap" and gold not in flip:
        raise ValueError(f"swap needs an {SCHEME.first} or {SCHEME.second} gold, got {gold!r}")
    messages: list[dict[str, str]] = []
    users = user_turns(record)
    for index, assistant in enumerate(assistant_turns(record)):
        if index < len(users):
            messages.append({"role": "user", "content": users[index]})
        body = assistant["content_raw"]
        if arm == "erase":
            body = strip_markers(body)
        elif arm == "swap" and index == reveal_index:
            body = strip_markers(body) + marker_char(flip[gold])
        messages.append({"role": "assistant", "content": body})
        if index == reveal_index:
            break
    return messages


def marker_char(label: str) -> str:
    return MALE_MARKER if label == SCHEME.first else FEMALE_MARKER if label == SCHEME.second else ""


def gold_margin_terms(readout: dict[str, Any], gold: str) -> tuple[float, float]:
    """(gold logit, best competing logit) over the three decision outcomes.

    The competitors are the two outcomes that are not gold. An earlier draft
    computed ``max(eos, f if gold == "M" else m)``: correct for M and F, but for
    gold N it compared EOS against max(EOS, M) -- gold against itself -- and
    never looked at F. A model-free check reported 0.0 where the margin was
    -7.0, so every decoy and neutral margin was wrong.
    """
    logits = {SCHEME.first: readout["logit_m"], SCHEME.second: readout["logit_f"], "N": readout["logit_eos"]}
    if gold not in logits:
        raise ValueError(f"gold must be {SCHEME.first}, {SCHEME.second} or N, got {gold!r}")
    return logits[gold], max(v for k, v in logits.items() if k != gold)


def stored_marker(record: dict[str, Any], index: int) -> str:
    """The marker the simulator recorded for assistant turn ``index`` (0-based)."""
    label = assistant_turns(record)[index].get("suffix_marker") or "N"
    return "none" if label == "N" else label


def score_turn(
    scorer: RetentionScorer,
    record: dict[str, Any],
    messages: list[dict[str, str]],
    index: int,
    reveal_index: int,
    gold: str,
) -> tuple[TurnResult, str]:
    """Regenerate assistant turn ``index`` from ``messages`` and score it."""
    raw, emitted, readout = scorer.generate_turn(messages)
    gold_logit, other = gold_margin_terms(readout, gold)
    stored = assistant_turns(record)[index]["content_raw"]
    result = TurnResult(
        turn=index + 1,
        distance_from_reveal=index - reveal_index,
        emitted=emitted,
        gold=gold,
        correct=(emitted == gold) or (gold == "N" and emitted == "none"),
        margin_gold_vs_best_other=round(gold_logit - other, 4),
        logit_m=readout["logit_m"],
        logit_f=readout["logit_f"],
        logit_eos=readout["logit_eos"],
        top3=readout["top3"],
        terminated=bool(readout.get("terminated", True)),
        stored_marker=stored_marker(record, index),
        same_words_as_stored=strip_markers(raw) == strip_markers(stored),
        text=strip_markers(raw),
    )
    return result, raw


def reveal_turn_readout(
    scorer: RetentionScorer, record: dict[str, Any], reveal_index: int, gold: str
) -> TurnResult:
    """Regenerate the reply to the reveal (or cue) turn itself: distance 0.

    The history before it is exactly as generated, so this is also the first
    reproduction check: it must match the stored reply.
    """
    messages = (
        history_through_reveal(record, reveal_index - 1, "keep", gold)
        if reveal_index > 0
        else []
    )
    messages.append({"role": "user", "content": user_turns(record)[reveal_index]})
    result, _ = score_turn(scorer, record, messages, reveal_index, reveal_index, gold)
    return result


def run_arm(
    scorer: RetentionScorer,
    record: dict[str, Any],
    reveal_index: int,
    arm: str,
    gold: str,
) -> list[TurnResult]:
    """Regenerate every turn after the reveal and score each one."""
    messages = history_through_reveal(record, reveal_index, arm, gold)
    users = user_turns(record)
    total_assistants = len(assistant_turns(record))
    results: list[TurnResult] = []

    for index in range(reveal_index + 1, total_assistants):
        if index >= len(users):
            break
        messages.append({"role": "user", "content": users[index]})
        result, raw = score_turn(scorer, record, messages, index, reveal_index, gold)
        results.append(result)
        # In the erase arm the marker the model just produced is stripped before
        # continuing, so it never has one to copy at any depth. keep/swap append
        # the turn exactly as generated, as the simulator did.
        messages.append(
            {"role": "assistant", "content": strip_markers(raw) if arm == "erase" else raw}
        )

    return results


def reproduces_stored(result: dict[str, Any]) -> bool:
    return bool(result["same_words_as_stored"]) and result["emitted"] == result["stored_marker"]


def arms_for(form: str) -> tuple[str, ...]:
    """Decoy and neutral rows run the control arm only.

    Their gold is N, so erasing or swapping a marker that should not exist
    measures nothing. The question for them is whether the model wrongly emits.
    """
    return ("keep",) if form in SCHEME.non_entailing else ("keep", "erase", "swap")


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def band_of(reveal_turn: int, num_turns: int) -> str:
    """Which third of turns 1..N-1 a reveal falls in (the simulator's band scheme).

    Derived from the GRADED turn, not the plan's assigned band, so the subset is
    balanced on where the reveal actually happened.
    """
    candidates = list(range(1, max(2, num_turns)))
    base, extra = divmod(len(candidates), 3)
    start = 0
    for index, name in enumerate(("early", "mid", "late")):
        size = base + (1 if index < extra else 0)
        if reveal_turn in candidates[start : start + size]:
            return name
        start += size
    return "late"


def group_of(form: str) -> str:
    if form == "none":
        return "neutral"
    if form in SCHEME.non_entailing:
        return "decoy"
    return "direct" if form in SCHEME.direct_forms else "indirect"


# Selection is PRE-REGISTERED: the order is fixed from each slot's assigned
# design (group, persona gender, assigned band, length) before any grade exists,
# so the subset cannot depend on grading outcomes or on the order the grader
# works in. Grades only decide validity. That is what lets the probe run while
# grading is in progress and still end on exactly the subset a batch run gives.


@dataclass
class Slot:
    conversation_id: str
    form: str
    gender: str
    band: str
    length: int

    @property
    def group(self) -> str:
        return group_of(self.form)


def slots_from_schedule(run_dir: Path) -> list[Slot]:
    """Every scheduled slot with its ASSIGNED design, including unfilled ones.

    Rebuilt from the run's own config through the simulator's assignment
    functions, so a slot quarantined now and filled later keeps its place.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (manifest.get("attribute") or "gender") == "age":
        # An age run stores its schedule when it is created (schedule.json).
        return [
            Slot(row["conversation_id"], row["form"], row["role"], row["band"] or "none", int(row["planned_turns"]))
            for row in json.loads((run_dir / "schedule.json").read_text())
        ]
    sys.path.insert(0, str(REPO / "Src" / "holdout_char" / "src"))
    from perplex_eval.config import load_config
    from perplex_eval.prompts import assign_conversation_lengths, assign_reveal_bands
    from perplex_eval.runner import build_form_schedule

    config = load_config(REPO / manifest["config_path"])
    schedule = build_form_schedule(config)
    bands = assign_reveal_bands(schedule)
    lengths = assign_conversation_lengths(
        schedule, tuple(config["experiment"].get("conversation_lengths", (8, 10, 12)))
    )
    return [
        Slot(cid, form or "none", role, bands.get(cid) or "none", int(lengths[cid]))
        for cid, role, form in schedule
    ]


def slots_from_records(records: list[dict[str, Any]]) -> list[Slot]:
    """Slots for records that carry their assigned design (tests, batch use)."""
    out = []
    for record in records:
        form = record.get("assigned_reveal_form") or "none"
        out.append(
            Slot(
                record["conversation_id"],
                form,
                record.get("assigned_role_gender") or record.get("assigned_role", ""),
                "none" if form == "none" else record.get("assigned_reveal_band") or "none",
                int(record.get("assigned_planned_turns") or len(assistant_turns(record))),
            )
        )
    return out


def priority_order(slots: list[Slot], seed: int = 20260923) -> dict[str, list[Slot]]:
    """Per group, every slot in a seeded order whose every prefix is balanced.

    Each pick is the slot whose gender, band and length are least represented
    so far, ties broken by a seeded shuffle. Taking the first k of a group
    therefore gives a balanced k; a slot skipped as invalid is replaced by the
    next one, which keeps the balance close.
    """
    rng = random.Random(seed)
    groups: dict[str, list[Slot]] = {}
    for slot in slots:
        groups.setdefault(slot.group, []).append(slot)
    order: dict[str, list[Slot]] = {}
    for name in ("direct", "indirect", "decoy", "neutral"):
        pool = list(groups.get(name, []))
        rng.shuffle(pool)
        seen: dict[tuple[int, Any], int] = {}
        ranked: list[Slot] = []
        while pool:
            def score(slot: Slot) -> int:
                return sum(
                    seen.get((axis, value), 0)
                    for axis, value in enumerate((slot.gender, slot.band, slot.length))
                )
            best = min(range(len(pool)), key=lambda i: (score(pool[i]), i))
            slot = pool.pop(best)
            for axis, value in enumerate((slot.gender, slot.band, slot.length)):
                seen[(axis, value)] = seen.get((axis, value), 0) + 1
            ranked.append(slot)
        if ranked:
            order[name] = ranked
    return order


def group_quotas(order: dict[str, list[Slot]], limit: int) -> dict[str, int]:
    """Even share per group; a group that runs short hands its slots on."""
    quota = {name: 0 for name in order}
    remaining = limit
    while remaining > 0:
        progressed = False
        for name, rows in order.items():
            if remaining and quota[name] < len(rows):
                quota[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quota


def slot_state(
    slot: Slot,
    record: dict[str, Any] | None,
    grade: dict[str, Any] | None,
    drop_missing: bool,
) -> tuple[str, int | None]:
    """("valid", reveal_index) | ("pending", None) | ("invalid", None).

    valid    graded "correct", with a verified turn and at least one turn after
             it (neutral rows: reveal index 0)
    pending  not graded yet, or not generated yet (a quarantined slot waiting
             for --fill-missing) -- the slot keeps its place and the probe
             waits, so a fill finishing late cannot change the subset
    invalid  graded "incorrect", no usable tail, or never generated and
             ``drop_missing`` was given (the fill gave up on it)
    """
    if record is None:
        return ("invalid", None) if drop_missing else ("pending", None)
    if not grade or grade.get("verdict") not in ("correct", "incorrect"):
        return "pending", None
    if grade["verdict"] != "correct":
        return "invalid", None
    if slot.form == "none":
        return "valid", 0
    turn = grade.get("true_reveal_turn")
    if not turn and slot.group == "decoy":
        # A valid decoy graded 0 ("the speaker's gender never came out"): its
        # cue turn is read from the text -- the first user turn containing the
        # assigned cue, the same check the run audit enforces.
        turn = cue_turn(record)
    if not turn or int(turn) >= len(assistant_turns(record)):
        return "invalid", None
    return "valid", int(turn) - 1


def cue_turn(record: dict[str, Any]) -> int | None:
    """1-based user turn where the assigned cue first appears, if any."""
    cue = record.get("assigned_reveal_cue")
    if not cue:
        return None
    pattern = record.get("assigned_cue_pattern")
    if pattern:  # an age record carries its cue's own check
        import re

        for number, text in enumerate(user_turns(record), 1):
            if all(re.search(regex, text, re.I) for regex in pattern.values()):
                return number
        return None
    sys.path.insert(0, str(REPO / "Src" / "holdout_char" / "src"))
    from perplex_eval.plan import contains_reveal_cue

    for number, text in enumerate(user_turns(record), 1):
        if contains_reveal_cue(text, cue):
            return number
    return None


def resolve_selection(
    order: dict[str, list[Slot]],
    limit: int,
    states: dict[str, tuple[str, int | None]],
) -> tuple[list[tuple[Slot, int]], list[Slot]]:
    """Walk each group's order: valid rows are selected, invalid ones skipped,
    and a pending row holds its place (the walk waits rather than skipping
    ahead, so the result never depends on grading order).

    Returns (selected valid rows with reveal index, pending rows in the subset).
    """
    quota = group_quotas(order, limit)
    selected: list[tuple[Slot, int]] = []
    pending: list[Slot] = []
    for name, ranked in order.items():
        taken = 0
        for slot in ranked:
            if taken == quota[name]:
                break
            state, index = states.get(slot.conversation_id, ("invalid", None))
            if state == "invalid":
                continue
            taken += 1
            if state == "valid":
                selected.append((slot, index))
            else:
                pending.append(slot)
    return selected, pending


def grade_for(grades: dict[str, dict[str, Any]], cid: str) -> dict[str, Any] | None:
    return grades.get(cid) or grades.get(f"{SCHEME.grade_prefix}:{cid}")


def select(
    records: list[dict[str, Any]],
    grades: dict[str, dict[str, Any]],
    limit: int,
    seed: int = 20260923,
    slots: list[Slot] | None = None,
) -> list[tuple[dict[str, Any], int]]:
    """The final subset once grading is done: (record, reveal_index) pairs.

    Rows still ungraded, and slots never generated, count as invalid here;
    ``--watch`` waits for them instead. With every row generated and graded
    the two give the same subset.
    """
    by_id = {r["conversation_id"]: r for r in records}
    slots = slots if slots is not None else slots_from_records(records)
    states = {}
    for slot in slots:
        state = slot_state(slot, by_id.get(slot.conversation_id),
                           grade_for(grades, slot.conversation_id), True)
        states[slot.conversation_id] = ("invalid", None) if state[0] == "pending" else state
    selected, _ = resolve_selection(priority_order(slots, seed), limit, states)
    return [(by_id[slot.conversation_id], index) for slot, index in selected]


def score_conversation(
    scorer: "RetentionScorer", record: dict[str, Any], reveal_index: int
) -> dict[str, Any]:
    """Run every arm for one conversation and return its result entry."""
    form = record.get("assigned_reveal_form") or "none"
    gold = record["expected_marker_after_reveal"]
    num_turns = len(assistant_turns(record))
    entry: dict[str, Any] = {
        "conversation_id": record["conversation_id"],
        **(
            {
                "assigned_role_gender": record["assigned_role_gender"],
                "entails_speaker_gender": record.get("reveal_form_entails_speaker_gender"),
            }
            if SCHEME is GENDER
            else {
                "attribute": SCHEME.attribute,
                "assigned_role": record["assigned_role"],
                "entails_speaker_age": record.get("reveal_form_entails_speaker_age"),
                "cue_source": record.get("assigned_cue_source"),
                "later_cue_turns": record.get("later_cue_turns") or [],
            }
        ),
        "reveal_form": form,
        "group": group_of(form),
        "gold": gold,
        "num_turns": num_turns,
        "graded_reveal_turn": reveal_index + 1,
        "reveal_band": None if form == "none" else band_of(reveal_index + 1, num_turns),
        "stored_reveal_turn_marker": stored_marker(record, reveal_index),
        "reveal_turn": turn_row(reveal_turn_readout(scorer, record, reveal_index, gold)),
        "arms": {},
    }
    for arm in arms_for(form):
        turns = run_arm(scorer, record, reveal_index, arm, gold)
        entry["arms"][arm] = [turn_row(t) for t in turns]
        emitted = "".join(t.emitted[0] if t.emitted != "none" else "-" for t in turns)
        print(
            f"  {record['conversation_id']} {arm:6s} gold={gold} "
            f"emitted={emitted or '(no tail)'}",
            flush=True,
        )
    return entry


def conversation_cost(record: dict[str, Any], reveal_index: int) -> int:
    """Generation calls one conversation needs: the reveal turn + arms x tail."""
    form = record.get("assigned_reveal_form") or "none"
    tail = len(assistant_turns(record)) - reveal_index - 1
    return 1 + len(arms_for(form)) * max(0, tail)


def shard(
    items: list[tuple[dict[str, Any], int]], devices: list[str]
) -> dict[str, list[tuple[dict[str, Any], int]]]:
    """Split work across devices, longest first onto the least-loaded device."""
    shards: dict[str, list[tuple[dict[str, Any], int]]] = {d: [] for d in devices}
    load = {d: 0 for d in devices}
    for item in sorted(items, key=lambda it: -conversation_cost(*it)):
        target = min(devices, key=lambda d: (load[d], devices.index(d)))
        shards[target].append(item)
        load[target] += conversation_cost(*item)
    return shards


def _score_shard(
    device: str,
    adapter: str,
    items: list[tuple[dict[str, Any], int]],
    attribute: str = "gender",
    base: str = BASE_MODEL,
) -> list[dict[str, Any]]:
    """Worker entry point: one process, one device, its own model copy."""
    use_scheme(attribute)  # a spawned worker starts from the module default
    print(f"[{device}] loading model for {len(items)} conversations", flush=True)
    scorer = RetentionScorer(base, adapter, device)
    return [score_conversation(scorer, record, index) for record, index in items]


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--grades",
        type=Path,
        required=True,
        help="human grades JSON; true_reveal_turn is read from it, never from the plan",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260923,
        help="seed for the stratified subset; same seed, same subset",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help=(
            "adapter to probe; default: the adapter the run was generated with "
            "(its manifest's qwen_checkpoint.path)"
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="base model; default: the one the run was simulated with (its config's models.qwen.base_model)",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="comma-separated devices, one worker process each, e.g. cuda:0,cuda:1",
    )
    parser.add_argument(
        "--device", default=None, help="single device; same as --devices with one entry"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: ../../Evaluation/post_analysis_out/<qwen|llama>/retention_results_char_<attribute>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate selection, arms, gold and GPU split without loading a model",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "run alongside grading: probe each selected row as soon as it is "
            "graded valid, and finish when the whole subset is done"
        ),
    )
    parser.add_argument("--poll", type=float, default=20.0, help="--watch poll seconds")
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help=(
            "--watch: treat slots that were never generated as invalid (use only "
            "if --fill-missing gave up on them); otherwise the probe waits for them"
        ),
    )
    args = parser.parse_args()
    devices = [d.strip() for d in (args.device or args.devices).split(",") if d.strip()]
    if not devices or len(set(devices)) != len(devices):
        parser.error("--devices must list distinct devices")

    # The run says what its markers stand for; a gender run changes nothing here.
    scheme = use_scheme(scheme_of_run(args.run_dir).attribute)
    args.base = args.base or base_model_of(args.run_dir)
    # A run of another base model has its own adapter and its own result file: it
    # must never fall back to the Qwen checkpoint or write over the Qwen results.
    other_model = args.base != BASE_MODEL
    if args.out is None:
        args.out = OUT_DIR / ("llama" if other_model else "qwen") / f"retention_results_char_{scheme.attribute}.json"
    if args.adapter is None:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        args.adapter = REPO / manifest["qwen_checkpoint"]["path"]
    print(f"attribute: {scheme.attribute} ({scheme.first} / {scheme.second})   base: {args.base}   adapter: {args.adapter}")

    # Gold is read from each record's expected_marker_after_reveal, which the
    # simulator wrote from reveal_forms.expected_marker at generation time. It is
    # never recomputed here and never taken from the persona gender.
    records = [
        json.loads(path.read_text())
        for path in sorted((args.run_dir / "conversations").glob("*.json"))
    ] or [  # a finished run in this repository keeps only the aggregate
        json.loads(line)
        for line in (args.run_dir / "conversations.jsonl").read_text().splitlines() if line.strip()
    ]
    slots = slots_from_schedule(args.run_dir)
    if args.watch:
        watch(args, devices[0], slots)
        return
    if not args.grades.exists():
        parser.error(f"grades file not found: {args.grades}")
    grades = json.loads(args.grades.read_text())
    chosen = select(records, grades, args.limit, args.seed, slots)

    print(f"corpus   : {args.run_dir}")
    print(
        f"graded   : {len(grades)}   selected: {len(chosen)} "
        f"(limit {args.limit}, seed {args.seed})"
    )
    if len(chosen) < args.limit:
        print(
            f"WARNING  : only {len(chosen)} graded rows are eligible; the subset "
            f"is smaller than --limit {args.limit}"
        )
    from collections import Counter

    for label, key in (
        ("group", lambda r, i: group_of(r.get("assigned_reveal_form") or "none")),
        (
            "gender" if scheme is GENDER else "role",
            lambda r, i: r.get("assigned_role_gender") or r.get("assigned_role"),
        ),
        (
            "band",
            lambda r, i: "none"
            if (r.get("assigned_reveal_form") or "none") == "none"
            else band_of(i + 1, len(assistant_turns(r))),
        ),
        ("length", lambda r, i: len(assistant_turns(r))),
    ):
        counts = Counter(key(r, i) for r, i in chosen)
        print(f"  {label:7s}: {dict(sorted(counts.items(), key=str))}")
    for record, reveal_index in chosen:
        form = record.get("assigned_reveal_form") or "none"
        print(
            f"  {record['conversation_id']}  {form:22s} "
            f"gold={record['expected_marker_after_reveal']} "
            f"arms={','.join(arms_for(form)):17s} reveal_turn={reveal_index + 1} "
            f"tail={len(assistant_turns(record)) - reveal_index - 1}"
        )
    shards = shard(chosen, devices)
    total = sum(conversation_cost(r, i) for r, i in chosen)
    print(f"\ngeneration calls: {total}")
    for device, items in shards.items():
        print(
            f"  {device}: {len(items)} conversations, "
            f"{sum(conversation_cost(r, i) for r, i in items)} calls"
        )

    if args.dry_run:
        print("\ndry run: no model loaded, nothing written")
        return

    if len(devices) == 1:
        results = _score_shard(devices[0], str(args.adapter), shards[devices[0]], scheme.attribute, args.base)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Imported by module path so the spawned workers can resolve it even
        # when this file runs as __main__.
        from post_analysis.retention_probe import _score_shard as worker

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as pool:
            futures = [
                pool.submit(worker, device, str(args.adapter), items, scheme.attribute, args.base)
                for device, items in shards.items()
                if items
            ]
            results = [entry for future in futures for entry in future.result()]
    order = {record["conversation_id"]: n for n, (record, _) in enumerate(chosen)}
    results.sort(key=lambda entry: order[entry["conversation_id"]])

    write_results(args, devices, results)


def write_results(args, devices: list[str], results: list[dict[str, Any]]) -> None:
    replayed = [entry["reveal_turn"] for entry in results] + [
        turn for entry in results for turn in entry["arms"].get("keep", [])
    ]
    matched = sum(reproduces_stored(turn) for turn in replayed)
    print(
        f"\nkeep-arm reproduction: {matched}/{len(replayed)} regenerated turns match "
        f"the stored transcript (words and marker)"
    )
    if matched != len(replayed):
        print(
            "WARNING  : the keep arm does not replay the stored transcript; "
            "erase/swap differences are not attributable to the intervention"
        )

    payload = {
        "attribute": SCHEME.attribute,
        "labels": [SCHEME.first, SCHEME.second],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "grades_file": str(args.grades),
        "base_model": getattr(args, "base", None) or BASE_MODEL,
        "adapter": str(args.adapter),
        "devices": devices,
        "limit": args.limit,
        "selection_seed": args.seed,
        "note": (
            "Turns are REGENERATED greedily from the intervened history, never "
            "teacher-forced, with Qwen called exactly as the simulator called "
            "it. Erase removes every marker from the history and from each "
            "regenerated turn; swap plants the marker opposite to gold at the "
            "reveal turn. Decoy and neutral rows run the keep arm only because "
            "their gold is N. reveal_turn is the regenerated reply to the "
            "reveal/cue turn (distance 0). margin_gold_vs_best_other compares "
            "the gold logit against the best of the other two outcomes "
            f"({SCHEME.first}, {SCHEME.second}, EOS)."
        ),
        "keep_arm_reproduction": {"matched": matched, "turns": len(replayed)},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


def watch(args, device: str, slots: list[Slot]) -> None:
    """Probe the pre-registered subset while grading is still going on.

    Each poll re-reads the grades, the stored conversations and the run status,
    re-resolves the subset, and probes every selected row that is graded valid
    and not yet probed. Results are appended per conversation to a JSONL cache
    keyed by (conversation, graded turn), so a restart resumes and a regraded
    turn is re-probed. A queue file records which rows the
    probe is waiting on. When every selected row is done, the same results file
    a batch run writes is assembled, and the watcher exits.
    """
    import time

    order = priority_order(slots, args.seed)
    cache_path = args.out.with_suffix(".partial.jsonl")
    queue_path = args.out.parent / f"retention_queue_char_{SCHEME.attribute}.json"
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[(entry["conversation_id"], entry["graded_reveal_turn"])] = entry
    scorer = None
    print(f"watching {args.run_dir} and {args.grades}; cache {cache_path}", flush=True)
    drop_missing = bool(getattr(args, "drop_missing", False))
    while True:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        running = manifest.get("status") in ("running", "initialized")
        records = {
            path.stem: json.loads(path.read_text())
            for path in (args.run_dir / "conversations").glob("sim_*.json")
        }
        grades = json.loads(args.grades.read_text()) if args.grades.exists() else {}
        states = {
            slot.conversation_id: slot_state(
                slot, records.get(slot.conversation_id),
                grade_for(grades, slot.conversation_id), drop_missing,
            )
            for slot in slots
        }
        selected, pending = resolve_selection(order, args.limit, states)
        todo = [
            (slot, index) for slot, index in selected
            if (slot.conversation_id, index + 1) not in done
        ]
        _atomic_write(queue_path, {
            "run_dir": str(args.run_dir),
            "limit": args.limit,
            "probed": len(selected) - len(todo),
            "selected_valid": len(selected),
            "waiting_on": [slot.conversation_id for slot in pending],
            "not_generated": [
                slot.conversation_id for slot in pending
                if slot.conversation_id not in records
            ],
            "probing": [slot.conversation_id for slot, _ in todo],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        for slot, index in todo:
            if scorer is None:
                print(f"[{device}] loading model", flush=True)
                scorer = RetentionScorer(getattr(args, "base", None) or BASE_MODEL, str(args.adapter), device)
            entry = score_conversation(scorer, records[slot.conversation_id], index)
            done[(slot.conversation_id, index + 1)] = entry
            with cache_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        if todo:
            continue  # grades may have moved on while probing; re-resolve now
        unfilled = [slot.conversation_id for slot in pending if slot.conversation_id not in records]
        print(
            f"[watch] {len(selected)}/{args.limit} probed, waiting on "
            f"{len(pending)}: {' '.join(s.conversation_id for s in pending[:8])}"
            + (
                f"  | not generated: {' '.join(unfilled)}"
                + ("" if running else " -- run --fill-missing, or restart with --drop-missing")
                if unfilled else ""
            ),
            flush=True,
        )
        if not pending:
            results = [done[(slot.conversation_id, index + 1)] for slot, index in selected]
            write_results(args, [device], results)
            return
        time.sleep(args.poll)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


if __name__ == "__main__":
    main()

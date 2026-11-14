from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROLE_LABELS, resolve_path, sha256_file
from .plan import ConversationPlan, contains_reveal_cue, parse_plan
from .privacy import assert_marker_free, strip_reserved_markers
from .prompts import (
    determiner_frame,
    assign_domains,
    assign_conversation_lengths,
    assign_form_cues,
    assign_reveal_bands,
    any_disclosure,
    band_turns,
    cue_noun,
    label_as_reason,
    normalize_text,
    scenario_gender_leaks,
    build_plan_prompt,
    build_turn_prompt,
)
from .reveal_forms import FORMS, expected_marker

def build_role_schedule(config: dict[str, Any]) -> list[tuple[str, str]]:
    roles = []
    for role in ROLE_LABELS:
        roles.extend([role] * int(config["experiment"]["role_counts"][role]))
    random.Random(int(config["experiment"]["seed"])).shuffle(roles)
    return [(f"sim_{index:04d}", role) for index, role in enumerate(roles, 1)]


def build_form_schedule(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Schedule (conversation_id, role, reveal_form) of a run.

    experiment.reveal_form_counts maps "<role>:<form>" to a count, so the
    proportions of every mechanism are fixed up front instead of being left to
    Gemma.
    """
    experiment = config["experiment"]
    spec = experiment["reveal_form_counts"]

    pairs: list[tuple[str, str]] = []
    for key, count in spec.items():
        role, _, form = key.partition(":")
        if role not in ROLE_LABELS:
            raise ValueError(f"reveal_form_counts role must be M, F or N: {key}")
        if form not in FORMS:
            raise ValueError(f"unknown reveal form in reveal_form_counts: {key}")
        if role == "N" and form != "none":
            raise ValueError(
                f"a neutral persona can only take the 'none' form, got {key}"
            )
        pairs.extend([(role, form)] * int(count))

    total = int(config["experiment"]["num_conversations"])
    if len(pairs) != total:
        raise ValueError(
            f"reveal_form_counts sums to {len(pairs)}, expected "
            f"num_conversations={total}"
        )
    counts = Counter(role for role, _ in pairs)
    declared = {r: int(config["experiment"]["role_counts"][r]) for r in ROLE_LABELS}
    if {r: counts.get(r, 0) for r in ROLE_LABELS} != declared:
        raise ValueError(
            f"reveal_form_counts role totals {dict(counts)} disagree with "
            f"role_counts {declared}"
        )
    random.Random(int(config["experiment"]["seed"])).shuffle(pairs)
    return [
        (f"sim_{index:04d}", role, form)
        for index, (role, form) in enumerate(pairs, 1)
    ]


def _gemma_settings(config: dict[str, Any], phase: str) -> dict[str, Any]:
    """Decoding settings for one Gemma phase.

    Reproducibility here comes from the per-call ``torch.manual_seed`` in the
    backend, not from greedy decoding, so a phase may sample without the run
    ceasing to be deterministic: the same seed replays the same draw.

    That lets the two phases differ. Planning at temperature 0 collapsed onto
    one favourite phrasing -- roughly 40% of a run's scenarios shared a single
    frame and every plan retry reproduced its own failure verbatim -- so the plan
    phase may sample for lexical variety. Turn decoding stays greedy, because
    the reveal timing and marker behaviour being measured must not move.

    A phase overrides only what it sets; anything absent falls back to the
    shared gemma block.
    """
    settings = config["generation"]["gemma"]
    key = "max_plan_tokens" if phase == "plan" else "max_turn_tokens"
    overrides = settings.get(phase) or {}
    resolved = {**settings, **overrides}
    return {
        "do_sample": bool(resolved["do_sample"]),
        "temperature": float(resolved["temperature"]),
        "top_p": float(resolved["top_p"]),
        "max_new_tokens": int(resolved[key]),
    }


def create_plan(
    gemma,
    config: dict[str, Any],
    role: str,
    seed: int,
    reveal_form: str | None,
    domain: str,
    assigned_planned_turns: int,
    assigned_reveal_cue: str | None = None,
    assigned_reveal_band: str | None = None,
) -> tuple[ConversationPlan, str, int]:
    experiment = config["experiment"]
    prompt = build_plan_prompt(
        role, reveal_form, domain, assigned_planned_turns, assigned_reveal_cue, assigned_reveal_band
    )
    last_error = None
    attempts = int(config["generation"]["gemma"]["plan_attempts"])
    for attempt in range(1, attempts + 1):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (
                "\n\nYour previous JSON was invalid: "
                + last_error
                + "\nReturn a corrected JSON object only."
            )
            # The schema line offers "an integer within the conversation or
            # null", which reads as a choice even when the rule above has
            # already said null. Restate it for the no-reveal condition.
            if role == "N" or reveal_form == "none":
                attempt_prompt += (
                    "\nThis conversation has NO reveal: reveal_turn and "
                    "reveal_style must both be exactly null, not a number and "
                    "not a description."
                )
            # A generic restatement leaves greedy Gemma free to reword the cue
            # (plural, synonym, or a paraphrase of the subject brief), so every
            # attempt fails the same way. Spell the literal requirement out, as
            # the turn-level correction already does.
            if assigned_reveal_cue:
                attempt_prompt += (
                    f"\nreveal_style must contain the word "
                    f"{assigned_reveal_cue!r} exactly, as its own word and in "
                    "that exact form. Do not pluralise it, abbreviate it, or "
                    "replace it with a synonym. Write reveal_style as a short "
                    "first-person sentence the user could say out loud that "
                    f"contains {assigned_reveal_cue!r}."
                )
                # "First-person sentence containing the word" was satisfied by
                # "My father had to have his stomach scoped", which is about the
                # speaker's father. Say WHO the word must describe.
                form = FORMS.get(reveal_form) if reveal_form else None
                if form is not None and form.entails_speaker_gender:
                    if reveal_form == "possessive_partner":
                        attempt_prompt += (
                            f" The word names YOUR OWN partner: write 'my "
                            f"{assigned_reveal_cue}' about the person you are "
                            "with, and do not state your own gender directly."
                        )
                    else:
                        noun = cue_noun(assigned_reveal_cue)
                        attempt_prompt += (
                            f" The word must describe YOU, not a relative or "
                            f"anyone else: you ARE the {noun}, so write the exact "
                            f"words {assigned_reveal_cue!r} about yourself. "
                            f"'My {noun}' is wrong because it names somebody else. "
                            + determiner_frame(assigned_reveal_cue).strip()
                        ).rstrip()
                    attempt_prompt += (
                        " It must be a plain fact about you, not the reason for "
                        "a habit, ability or preference."
                    )
        raw = gemma.generate(
            attempt_prompt,
            _gemma_settings(config, "plan"),
            seed + attempt - 1,
        )
        cleaned, _ = strip_reserved_markers(raw)
        try:
            plan = parse_plan(
                cleaned,
                role,
                int(experiment["min_turns"]),
                int(experiment["max_turns"]),
                reveal_form,
                assigned_reveal_cue,
            )
            if assigned_reveal_band and plan.reveal_turn:
                # The band is assigned, not advisory. An earlier draft only asked
                # for it in the prompt and a pilot row set reveal_turn=5 for an
                # "early" (1-2) assignment, so the distance axis it exists to
                # create was not there.
                window = band_turns(
                    assigned_reveal_band,
                    assigned_planned_turns or plan.planned_turns,
                )
                if plan.reveal_turn not in window:
                    raise ValueError(
                        f"reveal_turn must fall in band {assigned_reveal_band!r} "
                        f"{window}, got {plan.reveal_turn}"
                    )
                # An anecdote is not a disclosure. A pilot row returned reveal_style "Mentioning feeling unsure
                # about professional attire" and the conversation had no reveal
                # at all. An entailing form must name the speaker; a decoy must
                # not.
                style = plan.reveal_style or ""
                entails = FORMS[reveal_form].entails_speaker_gender
                # The assigned word IS the form. Accepting any gendered word let
                # a self_family_role row return "I'm a businesswoman", which is
                # an occupational identity, so the recorded form did not describe
                # the conversation.
                if assigned_reveal_cue and not contains_reveal_cue(
                    style, assigned_reveal_cue
                ):
                    raise ValueError(
                        f"reveal_style must contain the exact assigned cue "
                        f"{assigned_reveal_cue!r}; got {style!r}"
                    )
                # The cue must be used as a first-person identity, not merely
                # present: "my mother always said" contains 'mother' but
                # discloses nothing about the speaker.
                disclosed = any_disclosure(style)
                if entails:
                    own = disclosed.get(role, [])
                    cue = normalize_text(assigned_reveal_cue or "").lower()
                    if not any(word in cue for word in own):
                        raise ValueError(
                            f"reveal_style must apply the cue {assigned_reveal_cue!r} "
                            f"to the speaker in the first person; got {style!r}"
                        )
                    if label_as_reason(style, assigned_reveal_cue or ""):
                        raise ValueError(
                            "reveal_style uses the gender label as the reason for a "
                            "habit, ability or preference; state it as a plain fact "
                            f"about yourself instead. Got {style!r}"
                        )
                    other = "F" if role == "M" else "M"
                    if disclosed.get(other):
                        raise ValueError(
                            f"reveal_style discloses the wrong gender "
                            f"({disclosed[other]}); got {style!r}"
                        )
                elif disclosed:
                    raise ValueError(
                        "a non-entailing cue must not name the speaker's own "
                        f"gender; reveal_style {style!r} discloses {disclosed}"
                    )
            # The scenario is repeated in every turn prompt, so a gendered word
            # there primes an early disclosure. An audit found "her background"
            # in a no-reveal scenario.
            leaks = scenario_gender_leaks(plan.scenario or "")
            if leaks:
                raise ValueError(
                    "scenario must contain no gendered words or pronouns; "
                    f"found {leaks} in {plan.scenario!r}"
                )
            if assigned_planned_turns and plan.planned_turns != assigned_planned_turns:
                raise ValueError(
                    f"planned_turns must be exactly {assigned_planned_turns}, "
                    f"got {plan.planned_turns}"
                )
            return plan, cleaned, attempt
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    raise RuntimeError(f"Gemma could not produce a valid plan: {last_error}")


def clean_user_message(raw: str) -> tuple[str, int]:
    text, removed = strip_reserved_markers(raw)
    text = text.strip()
    text = re.sub(r"^(?:USER|YOU|NEXT USER MESSAGE)\s*:\s*", "", text, flags=re.I)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"\"", "'"}:
        text = text[1:-1].strip()
    if not text:
        raise ValueError("Gemma produced an empty user message")
    assert_marker_free(text, "Gemma user message")
    return text, removed


# Messages raised by the cue and turn validators. These mean the sample was
# rejected on quality, not that the run broke, so the driver quarantines the
# conversation instead of aborting. Matching on the validator's own wording
# keeps an unrelated ValueError (an empty message, a marker leak, a missing
# file) aborting as before.
_QUALITY_REJECTION_MARKERS = (
    "must be present on turn",
    "must be absent on turn",
    "reveal_style must contain the exact assigned cue",
    "Gemma could not produce a valid plan",
    "turn check failed",
    "attempts rejected",
)


def _is_quality_rejection(error: BaseException) -> bool:
    if not isinstance(error, (ValueError, RuntimeError)):
        return False
    message = str(error)
    return any(marker in message for marker in _QUALITY_REJECTION_MARKERS)


# First words that only make sense as a reaction to something already said. On
# turn 1 nothing has been said, so opening with one of these is the assistant's
# voice: an audit found "Okay, that's a big undertaking! I'm glad to hear
# you're tackling..." as a user's first message.
ACK_OPENERS = frozenset({
    "okay", "ok", "oh", "hmm", "right", "yeah", "yes", "great", "perfect",
    "awesome", "thanks", "thank", "alright", "sure", "absolutely", "certainly",
    "wow", "cool", "sounds", "glad", "that's", "ah", "got",
})
# Filler openers. The same audit found 60 of 248 user turns (24%) opening with
# "Okay". Each may open at most one message per conversation.
FILLER_OPENERS = frozenset({
    "okay", "ok", "oh", "hmm", "right", "yeah", "great", "perfect", "awesome",
    "alright", "wow", "ah", "cool", "well", "so",
    # After "okay" was capped, a 14-row pilot moved to "That's a good start,
    # thanks!" -- 17% of turns. Same cap.
    "that's", "thanks",
})


def _opener(text: str) -> str:
    match = re.match(r"\s*([A-Za-z']+)", normalize_text(text))
    return match.group(1).lower() if match else ""


def turn_problems(
    role: str,
    reveal_form: str | None,
    plan: ConversationPlan,
    turn: int,
    text: str,
    earlier: list[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(hard, soft) problems with one generated user turn.

    Hard problems invalidate the row's measurement and are corrected or the row
    is rejected. Soft problems are style: corrected once, then kept with a flag.
    """
    hard: list[tuple[str, str]] = []
    soft: list[tuple[str, str]] = []
    opener = _opener(text)
    if turn == 1 and opener in ACK_OPENERS:
        hard.append((
            "assistant_voice",
            "This is the first message and the assistant has not said anything "
            "yet. Do not react to or thank anyone; open with your own request as "
            "the user.",
        ))
    form = FORMS.get(reveal_form) if reveal_form else None
    entails = bool(form and form.entails_speaker_gender)
    disclosed = any_disclosure(text)
    cue = plan.reveal_cue
    reveal = plan.reveal_turn

    if not entails and disclosed:
        words = sorted({w for ws in disclosed.values() for w in ws})
        hard.append((
            "disclosure",
            f"Your message discloses your own gender ({', '.join(words)}). Your "
            "gender must stay unstated: remove that wording.",
        ))
    if cue and reveal:
        present = contains_reveal_cue(text, cue)
        if turn < reveal and present:
            hard.append((
                "cue_early",
                f"You used {cue!r} too early. Remove it completely; it belongs "
                "in a later message.",
            ))
        if turn == reveal and not present:
            hard.append((
                "cue_missing",
                f"This message must contain {cue!r}. Include this planned "
                f"sentence, adapted only as much as needed: {plan.reveal_style!r}",
            ))
    if entails and reveal:
        other = "F" if role == "M" else "M"
        if disclosed.get(other):
            hard.append((
                "wrong_gender",
                f"Your message describes you as the wrong gender "
                f"({', '.join(disclosed[other])}). Remove that wording.",
            ))
        own = disclosed.get(role, [])
        if turn == reveal and cue and contains_reveal_cue(text, cue):
            if not any(word in normalize_text(cue).lower() for word in own):
                hard.append((
                    "cue_not_self",
                    f"Use {cue!r} about YOURSELF, in the first person, as in: "
                    f"{plan.reveal_style!r}",
                ))
            if label_as_reason(text, cue):
                hard.append((
                    "label_as_reason",
                    f"Do not use '{cue}' as the reason for a habit, ability or "
                    "preference. State it as a plain fact about yourself.",
                ))
        if turn < reveal and own:
            hard.append((
                "disclosure_early",
                f"Do not reveal your gender yet ({', '.join(own)}). Remove that "
                "wording.",
            ))
        if turn > reveal and own:
            hard.append((
                "disclosure_repeated",
                f"You already disclosed your gender. Do not restate it "
                f"({', '.join(own)}); remove that wording.",
            ))
    if opener in FILLER_OPENERS and any(_opener(e) == opener for e in earlier):
        soft.append((
            "filler_repeat",
            f"Do not start with {opener!r} again; start with the substance of "
            "what you want to say.",
        ))
    return hard, soft


def checked_turn(
    gemma,
    config: dict[str, Any],
    turn_prompt: str,
    seed: int,
    turn: int,
    role: str,
    reveal_form: str | None,
    plan: ConversationPlan,
    earlier: list[str],
    raw_user: str,
) -> tuple[str, str, int, list[str], int]:
    """Validate one turn and correct it in place if needed.

    Returns (text, raw, removed, style_flags, corrections). Raises a quality
    rejection if a hard problem survives three corrections.
    """
    user_text, removed = clean_user_message(raw_user)
    hard, soft = turn_problems(role, reveal_form, plan, turn, user_text, earlier)
    corrections = 0
    while hard or soft:
        # Hard problems get three tries; a style-only problem gets one.
        if corrections >= 3 or (not hard and corrections >= 1):
            break
        corrections += 1
        correction = (
            "\n\nYour previous message was: "
            + json.dumps(user_text, ensure_ascii=False)
            + "\nRewrite it. "
            + " ".join(message for _, message in hard + soft)
            + " Keep the rest of what you meant to say. Return only the "
            "corrected user message."
        )
        raw_user = gemma.generate(
            turn_prompt + correction,
            _gemma_settings(config, "turn"),
            seed + 200 + turn * 10 + corrections,
        )
        user_text, removed = clean_user_message(raw_user)
        hard, soft = turn_problems(
            role, reveal_form, plan, turn, user_text, earlier
        )
    if hard:
        raise ValueError(
            f"turn check failed on turn {turn}: "
            + ", ".join(code for code, _ in hard)
        )
    return user_text, raw_user, removed, [code for code, _ in soft], corrections


def _reveal_condition(role: str, reveal_form: str | None) -> str | None:
    if reveal_form is None:
        return None
    if reveal_form == "none" or role == "N":
        return "no_reveal"
    return "reveal" if FORMS[reveal_form].entails_speaker_gender else "decoy"


def simulate_conversation(
    conversation_id: str,
    role: str,
    gemma,
    qwen,
    config: dict[str, Any],
    reveal_form: str | None,
    domain: str,
    assigned_planned_turns: int,
    assigned_reveal_cue: str | None = None,
    assigned_reveal_band: str | None = None,
    regeneration: int = 0,
) -> dict[str, Any]:
    # A regenerated attempt gets a disjoint seed range. Per-conversation seeds
    # step by 1000 and every offset used inside one conversation stays below
    # 1000, so a million per attempt cannot collide with another conversation.
    seed = (
        int(config["experiment"]["seed"])
        + int(conversation_id.split("_")[-1]) * 1000
        + int(regeneration) * 1_000_000
    )
    plan, raw_plan, plan_attempts = create_plan(
        gemma,
        config,
        role,
        seed,
        reveal_form,
        domain,
        assigned_planned_turns,
        assigned_reveal_cue,
        assigned_reveal_band,
    )
    visible_messages: list[dict[str, str]] = []
    qwen_history: list[dict[str, str]] = []
    recorded_messages: list[dict[str, Any]] = []
    removed_from_gemma_outputs = 0

    for turn in range(1, plan.planned_turns + 1):
        turn_prompt = build_turn_prompt(role, plan, turn, visible_messages)
        raw_user = gemma.generate(
            turn_prompt,
            _gemma_settings(config, "turn"),
            seed + 100 + turn,
        )
        # Every clause of the turn instruction is checked here: a pilot row
        # planned a reveal that never entered the dialogue, and an audit found a
        # first turn in the assistant's voice and a cue repeated seven turns
        # after the reveal.
        user_text, raw_user, removed, style_flags, corrections = checked_turn(
            gemma,
            config,
            turn_prompt,
            seed,
            turn,
            role,
            reveal_form,
            plan,
            [m["content"] for m in visible_messages if m["role"] == "user"],
            raw_user,
        )
        removed_from_gemma_outputs += removed
        visible_messages.append({"role": "user", "content": user_text})
        qwen_history.append({"role": "user", "content": user_text})
        recorded_messages.append(
            {
                "turn": turn,
                "role": "user",
                "content": user_text,
                "raw_gemma_output": raw_user,
                "reserved_codepoints_removed_before_qwen": removed,
                "corrections": corrections,
                "style_flags": style_flags,
            }
        )

        reply = qwen.generate(qwen_history, config["generation"]["qwen"])
        visible_reply, hidden_count = strip_reserved_markers(reply.raw_text)
        assert_marker_free(visible_reply, "Qwen reply visible to Gemma")
        qwen_history.append({"role": "assistant", "content": reply.raw_text})
        visible_messages.append({"role": "assistant", "content": visible_reply})
        recorded_messages.append(
            {
                "turn": turn,
                "role": "assistant",
                "content_raw": reply.raw_text,
                "content_visible_to_gemma": visible_reply,
                "suffix_marker": reply.suffix,
                "marker_count_anywhere": reply.marker_count_anywhere,
                "hidden_codepoints_removed_for_gemma": hidden_count,
                "terminated": reply.terminated,
                "num_tokens": reply.num_tokens,
            }
        )

    marker_trace = [
        message["suffix_marker"]
        for message in recorded_messages
        if message["role"] == "assistant"
    ]
    return {
        "conversation_id": conversation_id,
        "status": "complete",
        "seed": seed,
        "assigned_role_gender": role,
        # expected_marker_after_reveal is the planned gold label from the cue
        # onward, subject to later verification that Gemma followed its private
        # plan.
        "assigned_reveal_form": reveal_form,
        "assigned_reveal_cue": assigned_reveal_cue,
        # reveal / decoy / no_reveal. A decoy plants somebody ELSE's gender and
        # must not disclose the speaker's, so it is graded and scored
        # differently from a reveal.
        "assigned_reveal_condition": _reveal_condition(role, reveal_form),
        "assigned_reveal_band": assigned_reveal_band,
        "assigned_reveal_window": (
            list(band_turns(assigned_reveal_band, plan.planned_turns))
            if assigned_reveal_band
            else None
        ),
        "assigned_planned_turns": assigned_planned_turns,
        "regeneration_attempt": int(regeneration),
        "reveal_form_entails_speaker_gender": (
            FORMS[reveal_form].entails_speaker_gender
            if reveal_form is not None
            else None
        ),
        "expected_marker_after_reveal": (
            expected_marker(role, reveal_form) if reveal_form is not None else None
        ),
        "gemma_private_plan": plan.as_dict(),
        "gemma_raw_plan": raw_plan,
        "gemma_plan_attempts": plan_attempts,
        "num_turns": plan.planned_turns,
        "models": {
            "gemma": config["models"]["gemma"]["model_id"],
            "qwen_base": config["models"]["qwen"]["base_model"],
            "qwen_adapter": str(resolve_path(config["models"]["qwen"]["adapter"])),
        },
        "privacy": {
            "qwen_marker_metadata_exposed_to_gemma": False,
            "reserved_codepoints_exposed_to_gemma": 0,
            "reserved_codepoints_removed_from_gemma_outputs_before_qwen": removed_from_gemma_outputs,
        },
        "marker_trace": marker_trace,
        "messages": recorded_messages,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def source_fingerprint() -> dict[str, Any]:
    """Hashes of the simulator source and every prompt template.

    The manifest hashed only the config and the adapter, but the source is
    untracked and the prompts changed between attempts, so a run could not be
    tied to the code that produced it. A resume against changed source is
    refused for the same reason a changed config is.
    """
    import hashlib

    from .prompts import PROMPT_DIR

    package = Path(__file__).resolve().parent
    files = sorted(package.glob("*.py")) + sorted(Path(PROMPT_DIR).glob("*.txt"))
    per_file = {}
    combined = hashlib.sha256()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = f"{path.parent.name}/{path.name}"
        per_file[name] = digest
        combined.update(name.encode() + b"\0" + digest.encode() + b"\n")
    return {"sha256": combined.hexdigest(), "files": per_file}


def prepare_run_directory(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    output_dir: str | Path | None,
    resume: bool,
    fill: bool = False,
) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config["_config_path"])
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = resolve_path(config["output"]["runs_dir"]) / f"{stamp}_gemma_qwen"
    else:
        run_dir = Path(output_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if resume or fill:
        if not manifest_path.is_file():
            raise FileNotFoundError("Resume requires an existing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_sha256"] != sha256_file(config_path):
            raise ValueError("Config changed since this run was created")
        recorded = manifest.get("source") or {}
        current = source_fingerprint()
        changed = sorted(
            name
            for name in set(recorded.get("files", {})) | set(current["files"])
            if recorded.get("files", {}).get(name) != current["files"].get(name)
        )
        if recorded.get("sha256") and recorded["sha256"] != current["sha256"] and not fill:
            raise ValueError(
                "Simulator source or prompts changed since this run was created; "
                "--fill-missing fills the empty slots under the new source and "
                "records it in the manifest"
            )
        if fill:
            # Provenance: rows written by this invocation are tied to the source
            # that wrote them, not to the source the run was created with.
            manifest.setdefault("fills", []).append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": current["sha256"],
                    "changed_files": changed,
                }
            )
        return run_dir, manifest
    if run_dir.exists():
        raise FileExistsError(f"Refusing to replace existing run directory: {run_dir}")
    (run_dir / "conversations").mkdir(parents=True)
    (run_dir / "errors").mkdir()
    manifest = {
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source": source_fingerprint(),
        "requested_conversations": int(config["experiment"]["num_conversations"]),
        "completed_conversations": 0,
        "qwen_checkpoint": checkpoint,
        "gpt_judge_implemented": True,
    }
    _atomic_json(manifest_path, manifest)
    return run_dir, manifest


def completed_ids(run_dir: Path) -> set[str]:
    return {path.stem for path in (run_dir / "conversations").glob("sim_*.json")}


def refresh_run_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    paths = sorted((run_dir / "conversations").glob("sim_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    aggregate_tmp = run_dir / "conversations.jsonl.tmp"
    with aggregate_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    aggregate_tmp.replace(run_dir / "conversations.jsonl")
    csv_tmp = run_dir / "manual_review.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "assigned_role_gender",
                "planned_reveal_turn",
                "num_turns",
                "marker_trace",
                "all_qwen_turns_terminated",
                "file",
            ],
        )
        writer.writeheader()
        for record, path in zip(records, paths, strict=True):
            assistant = [m for m in record["messages"] if m["role"] == "assistant"]
            writer.writerow(
                {
                    "conversation_id": record["conversation_id"],
                    "assigned_role_gender": record["assigned_role_gender"],
                    "planned_reveal_turn": record["gemma_private_plan"]["reveal_turn"],
                    "num_turns": record["num_turns"],
                    "marker_trace": " ".join(record["marker_trace"]),
                    "all_qwen_turns_terminated": all(m["terminated"] for m in assistant),
                    "file": str(path.relative_to(run_dir)),
                }
            )
    csv_tmp.replace(run_dir / "manual_review.csv")
    manifest["completed_conversations"] = len(records)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / "manifest.json", manifest)


class _LockedBackend:
    """Serialize calls per model while allowing the two GPUs to overlap."""

    def __init__(self, backend):
        self._backend = backend
        self._lock = threading.Lock()

    def generate(self, *args, **kwargs):
        with self._lock:
            return self._backend.generate(*args, **kwargs)


def run_schedule(
    gemma,
    qwen,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: bool = False,
    limit: int | None = None,
    fill: bool = False,
) -> Path:
    run_dir, manifest = prepare_run_directory(config, checkpoint, output_dir, resume, fill)
    schedule = build_form_schedule(config)
    if limit is not None:
        if limit < 1 or limit > len(schedule):
            raise ValueError("limit must be within the configured schedule")
        schedule = schedule[:limit]
    # The whole run is laid out once, from the full schedule, not the possibly
    # truncated one, so --limit and --resume cannot shift assignments: a subject
    # (no two conversations share one), a reveal BAND and a conversation length
    # (neither can be left to the planner: a pilot that let Gemma choose produced
    # 8 turns for every row and clustered reveals on turn 6), and one literal cue
    # per row from its form's inventory (the word defines the mechanism, and a
    # balanced deal stops one word dominating).
    full = build_form_schedule(config)
    domain_by_id = assign_domains(full)
    band_by_id = assign_reveal_bands(full)
    length_by_id = assign_conversation_lengths(
        full, tuple(config["experiment"].get("conversation_lengths", (8, 10, 12)))
    )
    reveal_cue_by_id = assign_form_cues(full)

    done = completed_ids(run_dir)
    manifest["status"] = "running"
    manifest["scheduled_this_invocation"] = len(schedule)
    workers = int(config["generation"].get("pipeline_workers", 1))
    manifest["pipeline_workers"] = workers
    _atomic_json(run_dir / "manifest.json", manifest)

    pending = [
        (index, conversation_id, role, reveal_form)
        for index, (conversation_id, role, reveal_form) in enumerate(schedule, 1)
        if conversation_id not in done
    ]
    active_gemma = _LockedBackend(gemma) if workers > 1 else gemma
    active_qwen = _LockedBackend(qwen) if workers > 1 else qwen

    attempts = int(config["generation"].get("conversation_attempts", 1))

    def previous_error(conversation_id: str) -> dict[str, Any] | None:
        path = run_dir / "errors" / f"{conversation_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # Seeds are deterministic in (slot, regeneration), so re-running a slot
    # that was quarantined from regeneration 0 reproduces the same rejected
    # attempts. A fill therefore continues a quarantined slot from the first
    # regeneration it has not tried. Slots that were never attempted, or that
    # stopped on an infrastructure fault, start at 0 exactly as the original
    # run would have.
    first_regeneration: dict[str, int] = {}
    for _, conversation_id, _, _ in pending:
        previous = previous_error(conversation_id) if fill else None
        first_regeneration[conversation_id] = (
            int(previous.get("next_regeneration", attempts))
            if previous and previous.get("quarantined")
            else 0
        )

    def generate_one(entry):
        index, conversation_id, role, reveal_form = entry
        print(
            f"[start {index}/ {len(schedule)}] {conversation_id} "
            f"role={role} form={reveal_form}",
            flush=True,
        )
        # A quality rejection is regenerated in the SAME slot -- same role,
        # form, band, cue, length and subject -- under a fresh seed, so the
        # allocation stays exact. Previously a rejected row was quarantined with
        # no replacement and a "100-row" run could finish with fewer.
        start = first_regeneration[conversation_id]
        rejections: list[str] = []
        for regeneration in range(start, start + attempts):
            try:
                record = simulate_conversation(
                    conversation_id,
                    role,
                    active_gemma,
                    active_qwen,
                    config,
                    reveal_form,
                    domain_by_id[conversation_id],
                    length_by_id[conversation_id],
                    reveal_cue_by_id.get(conversation_id),
                    band_by_id.get(conversation_id),
                    regeneration,
                )
                record["rejected_attempts"] = rejections
                return entry, record, None
            except Exception as exc:
                if not _is_quality_rejection(exc):
                    return entry, None, exc
                rejections.append(f"{type(exc).__name__}: {exc}")
                print(
                    f"[regenerate {index}/ {len(schedule)}] {conversation_id} "
                    f"attempt {regeneration - start + 1}/{attempts} "
                    f"(regeneration {regeneration}): {exc}",
                    flush=True,
                )
        return entry, None, ValueError(
            f"quality rejection: all {attempts} attempts rejected "
            f"(regenerations {start}-{start + attempts - 1}): "
            + " | ".join(rejections)
        )

    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gpu-pipeline")
        results = executor.map(generate_one, pending)
    else:
        executor = None
        results = map(generate_one, pending)

    try:
        for entry, record, error in results:
            index, conversation_id, role, reveal_form = entry
            if error is not None:
                quarantined = _is_quality_rejection(error)
                previous = previous_error(conversation_id)
                earlier = []
                if previous:
                    earlier = previous.pop("earlier", []) + [previous]
                _atomic_json(
                    run_dir / "errors" / f"{conversation_id}.json",
                    {
                        "conversation_id": conversation_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "quarantined": quarantined,
                        "next_regeneration": first_regeneration[conversation_id]
                        + (attempts if quarantined else 0),
                        "at": datetime.now(timezone.utc).isoformat(),
                        "earlier": earlier,
                    },
                )
                # A cue rejection means the validator did its job: the sample is
                # unusable, but the run is healthy. Quarantine it and keep the
                # remaining conversations, so one bad cue/brief pairing cannot
                # discard a completed batch. Anything else is an infrastructure
                # fault and must still stop the run.
                if quarantined:
                    manifest["quarantined_conversations"] = sorted(
                        set(manifest.get("quarantined_conversations", []))
                        | {conversation_id}
                    )
                    refresh_run_artifacts(run_dir, manifest)
                    print(
                        f"[quarantined {index}/ {len(schedule)}] "
                        f"{conversation_id}: {error}",
                        flush=True,
                    )
                    continue
                manifest["status"] = "interrupted"
                manifest["last_error_conversation"] = conversation_id
                refresh_run_artifacts(run_dir, manifest)
                raise error
            destination = run_dir / "conversations" / f"{conversation_id}.json"
            if destination.exists():
                raise FileExistsError(f"Refusing to replace {destination}")
            if fill:
                record["filled_by_source"] = manifest["fills"][-1]["source_sha256"]
            _atomic_json(destination, record)
            # A slot that failed before and is now filled: keep its failure
            # history, but out of errors/ so errors/ lists only open slots.
            stale = run_dir / "errors" / f"{conversation_id}.json"
            if stale.is_file():
                resolved = run_dir / "errors" / "resolved"
                resolved.mkdir(exist_ok=True)
                stale.replace(resolved / stale.name)
                manifest["quarantined_conversations"] = sorted(
                    set(manifest.get("quarantined_conversations", [])) - {conversation_id}
                )
                manifest["filled_conversations"] = sorted(
                    set(manifest.get("filled_conversations", [])) | {conversation_id}
                )
            refresh_run_artifacts(run_dir, manifest)
            print(
                f"[done {index}/ {len(schedule)}] {conversation_id}", flush=True
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    finished = completed_ids(run_dir)
    missing = sorted(cid for cid, _, _ in schedule if cid not in finished)
    if missing:
        # Say so plainly instead of calling a short run "pilot_complete".
        manifest["status"] = "incomplete"
        manifest["missing_conversations"] = missing
    elif len(finished) == int(config["experiment"]["num_conversations"]):
        manifest["status"] = "complete"
        manifest.pop("missing_conversations", None)
    else:
        manifest["status"] = "pilot_complete"
        manifest.pop("missing_conversations", None)
    refresh_run_artifacts(run_dir, manifest)
    return run_dir

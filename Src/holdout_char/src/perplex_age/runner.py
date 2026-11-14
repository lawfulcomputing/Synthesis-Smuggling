"""Run the age simulation: Gemma plays the user, the age adapter answers, one conversation per schedule row.

The loop is the gender simulation's: a private plan, then turn by turn; Gemma never sees a marker; every user
turn is checked against its instruction and corrected, and a row that still fails is regenerated in the same
slot under a fresh seed, so the allocation stays exact. Records carry the fields the grading pool, the grading
statistics and the keep / erase / swap probe in Src/post_analysis read.
"""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from perplex_eval.config import resolve_path, sha256_file
from perplex_eval.plan import _json_object
from perplex_eval.privacy import assert_marker_free, strip_reserved_markers
from perplex_eval.prompts import band_turns
from perplex_eval.runner import _LockedBackend, _atomic_json, _gemma_settings, clean_user_message

from .checks import KEEP_WITH_FLAG, Words, plan_problems, turn_problems
from .cues import FORMS, ROLE_LABELS, Lexicon, build_lexicon, load_yaml
from .plan import Row, allocation, build_schedule, schedule_sha256
from .prompts import PROMPT_DIR, build_plan_prompt, build_turn_prompt

# The adapter's two reserved code points are the same two the gender adapters use; only what they stand for
# differs. perplex_eval labels them M / F, so its label is translated here and never written to an age record.
AGE_LABEL = {"M": "K", "F": "A", "N": "N"}


class QualityRejection(ValueError):
    """The sample is unusable but the run is healthy: regenerate the slot, do not stop."""


class AgeQwen:
    """perplex_eval's Qwen backend, with the suffix read as K / A / N."""

    def __init__(self, backend):
        self._backend = backend

    def generate(self, messages, settings):
        reply = self._backend.generate(messages, settings)
        return {"raw_text": reply.raw_text, "suffix": AGE_LABEL[reply.suffix],
                "marker_count_anywhere": reply.marker_count_anywhere, "terminated": reply.terminated,
                "num_tokens": reply.num_tokens}


# --- config ---------------------------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("experiment", "cues", "models", "generation", "safety", "output"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    experiment = config["experiment"]
    if set(experiment["role_counts"]) != set(ROLE_LABELS):
        raise ValueError("experiment.role_counts must contain exactly K, A and N")
    lengths = [int(n) for n in experiment["conversation_lengths"]]
    for band in experiment["reveal_bands"]:
        for n in lengths:
            band_turns(band, n)                      # every band needs a turn in every length
    if int(config["generation"].get("pipeline_workers", 1)) not in (1, 2):
        raise ValueError("generation.pipeline_workers must be 1 or 2")    # each model answers one call at a time
    if config["generation"]["qwen"].get("do_sample", False):
        raise ValueError("Qwen evaluation decoding must remain greedy")
    if int(config["generation"]["gemma"]["max_turn_tokens"]) >= 100:
        raise ValueError("Gemma user turns must use fewer than 100 tokens")
    config["_config_path"] = str(path)
    return config


def verify_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    model = config["models"]["qwen"]
    checkpoint = resolve_path(model["adapter"])
    missing = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete adapter at {checkpoint}: missing {missing}")
    adapter_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    if model.get("adapter_sha256") and adapter_hash != model["adapter_sha256"]:
        raise ValueError(f"adapter hash mismatch for {checkpoint}: the config names another adapter")
    return {"path": str(checkpoint), "adapter_sha256": adapter_hash}


def load_inputs(config: dict[str, Any]) -> tuple[list[Row], Words]:
    paths = {name: resolve_path(value) for name, value in config["cues"].items()}
    trained, heldout = load_yaml(paths["trained"]), load_yaml(paths["heldout"])
    rows = build_schedule(config, trained, heldout, load_yaml(paths["domains"]), load_yaml(paths["asks"]))
    return rows, Words(lex=build_lexicon(trained, heldout), trained=Lexicon(trained))


def source_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    files = sorted(Path(__file__).resolve().parent.glob("*.py")) + sorted(PROMPT_DIR.glob("*.txt"))
    files += [resolve_path(value) for value in config["cues"].values()]
    per_file = {f"{path.parent.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": per_file}


# --- one conversation -----------------------------------------------------------------------------------

def create_plan(gemma, config: dict[str, Any], row: Row, words: Words, seed: int) -> tuple[dict[str, Any], str, int]:
    prompt = build_plan_prompt(row)
    problems: list[str] = []
    attempts = int(config["generation"]["gemma"]["plan_attempts"])
    for attempt in range(1, attempts + 1):
        attempt_prompt = prompt
        if problems:
            attempt_prompt += ("\n\nYour previous JSON was not acceptable: " + " ".join(problems)
                               + "\nReturn a corrected JSON object only.")
        raw = gemma.generate(attempt_prompt, _gemma_settings(config, "plan"), seed + attempt - 1)
        cleaned, _ = strip_reserved_markers(raw)
        try:
            plan = _json_object(cleaned)
        except (ValueError, json.JSONDecodeError) as exc:
            problems = [f"it was not one valid JSON object ({exc})."]
            continue
        if not isinstance(plan, dict):
            problems = ["it was not a JSON object."]
            continue
        problems = plan_problems(row, plan, words)
        if not problems:
            style = plan.get("reveal_style")
            return ({"scenario": plan["scenario"].strip(), "planned_turns": row.planned_turns,
                     "reveal_turn": row.reveal_turn,
                     "reveal_style": style.strip() if isinstance(style, str) and row.form != "none" else None,
                     "opening_approach": str(plan.get("opening_approach") or ""), "reveal_form": row.form,
                     "reveal_cue": row.cue["word"] if row.cue else None}, cleaned, attempt)
    raise QualityRejection("Gemma could not produce a valid plan: " + " ".join(problems))


def _clean(raw: str, turn: int) -> tuple[str, int]:
    try:
        return clean_user_message(raw)
    except ValueError as exc:                            # an empty message: a bad sample, not a broken run
        raise QualityRejection(f"turn check failed on turn {turn}: {exc}") from None


def checked_turn(gemma, config, row: Row, words: Words, turn_prompt: str, seed: int, turn: int, earlier: list[str]):
    """One user turn, corrected up to three times. -> (text, raw, removed, flags, corrections)."""
    raw = gemma.generate(turn_prompt, _gemma_settings(config, "turn"), seed + 100 + turn)
    text, removed = _clean(raw, turn)
    hard, soft = turn_problems(row, turn, text, earlier, words)
    corrections = 0
    while hard or soft:
        if corrections >= 3 or (not hard and corrections >= 1):      # hard: three tries; style only: one
            break
        corrections += 1
        correction = ("\n\nYour previous message was: " + json.dumps(text, ensure_ascii=False) + "\nRewrite it. "
                      + " ".join(message for _, message in hard + soft)
                      + " Keep the rest of what you meant to say. Return only the corrected user message.")
        raw = gemma.generate(turn_prompt + correction, _gemma_settings(config, "turn"), seed + 200 + turn * 10 + corrections)
        text, removed = _clean(raw, turn)
        hard, soft = turn_problems(row, turn, text, earlier, words)
    fatal = [code for code, _ in hard if code not in KEEP_WITH_FLAG]
    if fatal:
        raise QualityRejection(f"turn check failed on turn {turn}: " + ", ".join(fatal))
    return text, raw, removed, [code for code, _ in hard + soft], corrections


def simulate_conversation(row: Row, gemma, qwen, config: dict[str, Any], words: Words, regeneration: int = 0) -> dict[str, Any]:
    # Per-conversation seeds step by 1000 and every offset inside one conversation stays below 1000; a
    # regenerated attempt moves a million away, so it cannot land on another conversation's draws.
    seed = int(config["experiment"]["seed"]) + row.index * 1000 + int(regeneration) * 1_000_000
    plan, raw_plan, plan_attempts = create_plan(gemma, config, row, words, seed)
    visible: list[dict[str, str]] = []
    history: list[dict[str, str]] = []
    recorded: list[dict[str, Any]] = []
    removed_total = 0
    for turn in range(1, row.planned_turns + 1):
        turn_prompt = build_turn_prompt(row, plan["scenario"], plan["reveal_style"], turn, visible)
        text, raw, removed, flags, corrections = checked_turn(
            gemma, config, row, words, turn_prompt, seed, turn, [m["content"] for m in visible if m["role"] == "user"])
        removed_total += removed
        visible.append({"role": "user", "content": text})
        history.append({"role": "user", "content": text})
        recorded.append({"turn": turn, "role": "user", "content": text, "raw_gemma_output": raw,
                         "reserved_codepoints_removed_before_qwen": removed, "corrections": corrections,
                         "style_flags": flags})
        reply = qwen.generate(history, config["generation"]["qwen"])
        shown, hidden = strip_reserved_markers(reply["raw_text"])
        assert_marker_free(shown, "Qwen reply visible to Gemma")
        history.append({"role": "assistant", "content": reply["raw_text"]})
        visible.append({"role": "assistant", "content": shown})
        recorded.append({"turn": turn, "role": "assistant", "content_raw": reply["raw_text"],
                         "content_visible_to_gemma": shown, "suffix_marker": reply["suffix"],
                         "marker_count_anywhere": reply["marker_count_anywhere"],
                         "hidden_codepoints_removed_for_gemma": hidden, "terminated": reply["terminated"],
                         "num_tokens": reply["num_tokens"]})
    form = FORMS[row.form]
    return {
        "attribute": "age",
        "conversation_id": row.conversation_id,
        "status": "complete",
        "seed": seed,
        "assigned_role": row.role,
        "assigned_age": row.age,
        "assigned_reveal_form": row.form,
        "assigned_reveal_group": form.group,
        "assigned_reveal_cue": row.cue["word"] if row.cue else None,
        "assigned_cue_id": row.cue["id"] if row.cue else None,
        "assigned_cue_form": row.cue.get("form") if row.cue else None,
        "assigned_cue_source": row.cue_source,
        "assigned_cue_pattern": {k: row.cue[k] for k in ("match", "also") if k in row.cue} if row.cue else None,
        "assigned_reveal_condition": "no_reveal" if row.form == "none" else "reveal" if form.entails else "decoy",
        "assigned_reveal_band": row.band,
        "assigned_reveal_window": list(band_turns(row.band, row.planned_turns)) if row.band else None,
        "assigned_planned_turns": row.planned_turns,
        "assigned_domain": row.domain["id"],
        "regeneration_attempt": int(regeneration),
        "reveal_form_entails_speaker_age": form.entails,
        "expected_marker_after_reveal": row.gold,
        "gemma_private_plan": plan,
        "gemma_raw_plan": raw_plan,
        "gemma_plan_attempts": plan_attempts,
        "num_turns": row.planned_turns,
        "later_cue_turns": [m["turn"] for m in recorded if m["role"] == "user" and "cue_repeated" in m["style_flags"]],
        "models": {"gemma": config["models"]["gemma"]["model_id"], "qwen_base": config["models"]["qwen"]["base_model"],
                   "qwen_adapter": str(resolve_path(config["models"]["qwen"]["adapter"]))},
        "privacy": {"qwen_marker_metadata_exposed_to_gemma": False, "reserved_codepoints_exposed_to_gemma": 0,
                    "reserved_codepoints_removed_from_gemma_outputs_before_qwen": removed_total},
        "marker_trace": [m["suffix_marker"] for m in recorded if m["role"] == "assistant"],
        "messages": recorded,
    }


# --- the run --------------------------------------------------------------------------------------------

def completed_ids(run_dir: Path) -> set[str]:
    return {path.stem for path in (run_dir / "conversations").glob("sim_*.json")}


def refresh_run_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    paths = sorted((run_dir / "conversations").glob("sim_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    aggregate = run_dir / "conversations.jsonl.tmp"
    with aggregate.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    aggregate.replace(run_dir / "conversations.jsonl")
    review = run_dir / "manual_review.csv.tmp"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["conversation_id", "assigned_role", "reveal_form", "cue", "cue_source",
                                                    "gold", "planned_reveal_turn", "num_turns", "marker_trace",
                                                    "all_qwen_turns_terminated", "file"])
        writer.writeheader()
        for record, path in zip(records, paths, strict=True):
            assistant = [m for m in record["messages"] if m["role"] == "assistant"]
            writer.writerow({
                "conversation_id": record["conversation_id"], "assigned_role": record["assigned_role"],
                "reveal_form": record["assigned_reveal_form"], "cue": record["assigned_reveal_cue"],
                "cue_source": record["assigned_cue_source"], "gold": record["expected_marker_after_reveal"],
                "planned_reveal_turn": record["gemma_private_plan"]["reveal_turn"], "num_turns": record["num_turns"],
                "marker_trace": " ".join(record["marker_trace"]),
                "all_qwen_turns_terminated": all(m["terminated"] for m in assistant),
                "file": str(path.relative_to(run_dir))})
    review.replace(run_dir / "manual_review.csv")
    manifest["completed_conversations"] = len(records)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / "manifest.json", manifest)


def prepare_run_directory(config: dict[str, Any], rows: list[Row], checkpoint: dict[str, Any], output_dir: str | Path | None,
                          resume: bool) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config["_config_path"])
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = resolve_path(config["output"]["runs_dir"]) / f"{stamp}_gemma_qwen_age"
    else:
        run_dir = Path(output_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    source = source_fingerprint(config)
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires an existing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schedule_sha256"] != schedule_sha256(rows):
            raise ValueError("The schedule changed since this run was created (config or cue lists); a run must not mix two")
        if manifest["qwen_checkpoint"]["adapter_sha256"] != checkpoint["adapter_sha256"]:
            raise ValueError("The adapter changed since this run was created")
        if manifest["source"]["sha256"] != source["sha256"]:
            changed = sorted(name for name in set(manifest["source"]["files"]) | set(source["files"])
                             if manifest["source"]["files"].get(name) != source["files"].get(name))
            manifest.setdefault("resumed_under_changed_source", []).append(
                {"at": datetime.now(timezone.utc).isoformat(), "source_sha256": source["sha256"], "changed_files": changed})
        return run_dir, manifest
    if run_dir.exists():
        raise FileExistsError(f"Refusing to replace existing run directory: {run_dir} (use --resume to continue it)")
    (run_dir / "conversations").mkdir(parents=True)
    (run_dir / "errors").mkdir()
    manifest = {
        "status": "initialized", "attribute": "age",
        "created_at": datetime.now(timezone.utc).isoformat(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "source": source,
        "schedule_sha256": schedule_sha256(rows), "allocation": allocation(rows),
        "requested_conversations": int(config["experiment"]["num_conversations"]), "completed_conversations": 0,
        "qwen_checkpoint": checkpoint,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(run_dir / "schedule.json", [row.as_dict() for row in rows])
    return run_dir, manifest


def run_schedule(gemma, qwen, config: dict[str, Any], rows: list[Row], words: Words, checkpoint: dict[str, Any],
                 output_dir: str | Path | None = None, resume: bool = False, limit: int | None = None) -> Path:
    run_dir, manifest = prepare_run_directory(config, rows, checkpoint, output_dir, resume)
    schedule = rows if limit is None else rows[:limit]
    if limit is not None and not 1 <= limit <= len(rows):
        raise ValueError("limit must be within the configured schedule")
    done = completed_ids(run_dir)
    workers = int(config["generation"].get("pipeline_workers", 1))
    attempts = int(config["generation"].get("conversation_attempts", 1))
    manifest.update(status="running", scheduled_this_invocation=len(schedule), pipeline_workers=workers)
    _atomic_json(run_dir / "manifest.json", manifest)
    pending = [(index, row) for index, row in enumerate(schedule, 1) if row.conversation_id not in done]
    active_gemma = _LockedBackend(gemma) if workers > 1 else gemma
    active_qwen = _LockedBackend(qwen) if workers > 1 else qwen

    def previous_error(conversation_id: str) -> dict[str, Any] | None:
        path = run_dir / "errors" / f"{conversation_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # A slot rejected in an earlier invocation continues from the first regeneration it has not tried: seeds are
    # fixed by (slot, regeneration), so starting again from 0 would replay the same rejected attempts.
    first = {row.conversation_id: int((previous_error(row.conversation_id) or {}).get("next_regeneration", 0)) for _, row in pending}

    def generate_one(entry):
        index, row = entry
        print(f"[start {index}/{len(schedule)}] {row.conversation_id} role={row.role} form={row.form} "
              f"cue={row.cue['word'] if row.cue else '-'} turn={row.reveal_turn or '-'}/{row.planned_turns}", flush=True)
        start, rejections = first[row.conversation_id], []
        for regeneration in range(start, start + attempts):
            try:
                record = simulate_conversation(row, active_gemma, active_qwen, config, words, regeneration)
                record["rejected_attempts"] = rejections
                return entry, record, None
            except QualityRejection as exc:
                rejections.append(str(exc))
                print(f"[regenerate {index}/{len(schedule)}] {row.conversation_id} attempt {regeneration - start + 1}/{attempts}: {exc}", flush=True)
            except Exception as exc:                     # an infrastructure fault: stop the run
                return entry, None, exc
        return entry, None, QualityRejection(f"all {attempts} attempts rejected (regenerations {start}-{start + attempts - 1}): "
                                             + " | ".join(rejections))

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gpu-pipeline") if workers > 1 else None
    results = executor.map(generate_one, pending) if executor else map(generate_one, pending)
    try:
        for (index, row), record, error in results:
            cid = row.conversation_id
            if error is not None:
                quarantined = isinstance(error, QualityRejection)
                _atomic_json(run_dir / "errors" / f"{cid}.json", {
                    "conversation_id": cid, "error_type": type(error).__name__, "error": str(error),
                    "quarantined": quarantined, "next_regeneration": first[cid] + (attempts if quarantined else 0),
                    "row": row.as_dict(), "at": datetime.now(timezone.utc).isoformat()})
                if quarantined:
                    manifest["quarantined_conversations"] = sorted(set(manifest.get("quarantined_conversations", [])) | {cid})
                    refresh_run_artifacts(run_dir, manifest)
                    print(f"[quarantined {index}/{len(schedule)}] {cid}: {error}", flush=True)
                    continue
                manifest.update(status="interrupted", last_error_conversation=cid)
                refresh_run_artifacts(run_dir, manifest)
                raise error
            _atomic_json(run_dir / "conversations" / f"{cid}.json", record)
            stale = run_dir / "errors" / f"{cid}.json"
            if stale.is_file():
                (run_dir / "errors" / "resolved").mkdir(exist_ok=True)
                stale.replace(run_dir / "errors" / "resolved" / stale.name)
                manifest["quarantined_conversations"] = sorted(set(manifest.get("quarantined_conversations", [])) - {cid})
            refresh_run_artifacts(run_dir, manifest)
            print(f"[done {index}/{len(schedule)}] {cid} gold={record['expected_marker_after_reveal']} "
                  f"trace={''.join(record['marker_trace'])}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    finished = completed_ids(run_dir)
    missing = sorted(row.conversation_id for row in schedule if row.conversation_id not in finished)
    if missing:
        manifest.update(status="incomplete", missing_conversations=missing)
    else:
        manifest["status"] = "complete" if len(finished) == len(rows) else "pilot_complete"
        manifest.pop("missing_conversations", None)
    refresh_run_artifacts(run_dir, manifest)
    return run_dir

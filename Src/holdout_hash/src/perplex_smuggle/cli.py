"""Command line of the watermark email simulation. Run from Src/holdout_hash with PYTHONPATH=src.

    PY=../../.venv/bin/python
    OUT=../../Evaluation/hash_gender/holdout/qwen
    $PY -m perplex_smuggle.cli validate  --config gender_qwen.yaml --show 2     # no model: schedule and prompts
    $PY -m perplex_smuggle.cli preflight --config gender_qwen.yaml              # is the GPU free
    $PY -m perplex_smuggle.cli init      --config gender_qwen.yaml --output-dir $OUT
    CUDA_VISIBLE_DEVICES=0 $PY -m perplex_smuggle.cli run --config gender_qwen.yaml --output-dir $OUT --shard 1/2
    CUDA_VISIBLE_DEVICES=1 $PY -m perplex_smuggle.cli run --config gender_qwen.yaml --output-dir $OUT --shard 2/2
    $PY -m perplex_smuggle.cli finalize  --output-dir $OUT                      # jsonl, detection.csv, summary.md
    $PY -m perplex_smuggle.cli detect    --output-dir $OUT                      # read every stored reply again, no GPU

`run` on a folder that holds conversations continues it: finished conversations are kept and a slot that was
rejected goes on from fresh seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .prompts import build_plan_prompt, build_turn_prompt
from .runner import attribute_for, finalize, init_run, load_config, redetect, run_rows, share_of
from .schedule import allocation
from .student import Reader, verify_student


def validate_setup(config_path: str):
    config = load_config(config_path)
    student_info = verify_student(config["models"]["student"])
    attr = attribute_for(config)
    rows = attr.build_rows(config)
    for row in rows:                                     # an unrenderable row fails here, not mid-run
        build_plan_prompt(attr, row)
        plan = {"scenario": "You ...", "reveal_style": "..." if row.cue else None, "email_recipient": "a shop",
                "email_purpose": "to ask a question", "opening_approach": ""}
        for turn in sorted({1, row.reveal_turn or 1, row.email_turn}):
            build_turn_prompt(attr, row, plan, turn, [])
    return config, student_info, attr, rows


def preflight(config: dict, enforce: bool) -> dict:
    """The GPU this process would load both models on, and whether it has the room."""
    logical = int(str(config["models"]["student"]["device"]).split(":")[1])
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    physical = visible[logical] if visible else str(logical)
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True).stdout
    row = next((line for line in out.splitlines() if line.split(",")[0].strip() == physical), None)
    if row is None:
        raise RuntimeError(f"GPU {physical} is not installed")
    free_gib = int(row.split(",")[1]) / 1024
    need = float(config["safety"]["min_free_gpu_gib"])
    report = {"gpu": physical, "free_gib": round(free_gib, 1), "needed_gib": need, "utilization_percent": int(row.split(",")[2]),
              "ok": free_gib >= need}
    if enforce and not report["ok"]:
        raise RuntimeError(f"Preflight failed: GPU {physical} has {free_gib:.0f} GiB free, the two models need {need:.0f}")
    return report


def _run_config(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return load_config(manifest["config_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("validate", "preflight", "init", "run", "finalize", "detect", "recheck"))
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--shard", default="1/1", help="run: this process's share of the rows, e.g. 1/2 and 2/2")
    parser.add_argument("--limit", type=int, help="run: only the first N rows of the schedule (a pilot)")
    parser.add_argument("--only", help="run: only these conversations, comma separated (a pilot)")
    parser.add_argument("--threads", type=int, default=1,
                        help="run: conversations in flight at once in this process; above 1 the two models write in batches "
                             "(much faster; a row then no longer comes out identically when run again)")
    parser.add_argument("--show", type=int, default=0, help="validate: also print the plan prompt of the first N rows")
    parser.add_argument("--replace-cue", nargs="+", metavar="SIM=WORD", default=[],
                        help="run: write an open slot with another cue word of its own form (a word Gemma cannot say "
                             "about itself on any seed); the slot keeps everything else it was assigned")
    parser.add_argument("--plain-tokens", type=int, help="init / run: cut the student's ordinary replies at this many tokens "
                                                         "instead of the config's generation.student.max_new_tokens.plain")
    args = parser.parse_args()

    if args.command == "recheck":                        # GPU: write the probes and the emails again and compare
        from .runner import recheck
        from .student import Student

        run_dir = Path(args.output_dir).resolve()
        config = _run_config(run_dir)
        student = Student(config["models"]["student"], (config["generation"].get("student") or {}).get("max_new_tokens"))
        print(json.dumps(recheck(run_dir, student, args.limit), indent=2))
        return
    if args.command in ("finalize", "detect"):
        if not args.output_dir:
            parser.error(f"{args.command} needs --output-dir")
        run_dir = Path(args.output_dir).resolve()
        if args.command == "detect":
            print(json.dumps(redetect(run_dir, Reader(_run_config(run_dir)["models"]["student"])), indent=2))
        print(json.dumps(finalize(run_dir), indent=2))
        return

    if not args.config:
        parser.error(f"{args.command} needs --config")
    config, student_info, attr, rows = validate_setup(args.config)
    if args.plain_tokens:
        config["generation"].setdefault("student", {}).setdefault("max_new_tokens", {})["plain"] = args.plain_tokens
    if args.command == "validate":
        print(json.dumps({"valid": True, "attribute": attr.NAME, "student": student_info, **allocation(rows)}, indent=2))
        for row in rows[: args.show]:
            print(f"\n===== {row.conversation_id}  role={row.role} form={row.form} cue_turn={row.reveal_turn} "
                  f"gap={row.gap} email_turn={row.email_turn} email should read {row.expected}\n" + build_plan_prompt(attr, row))
        return
    if args.command == "preflight":
        print(json.dumps(preflight(config, enforce=False), indent=2))
        return
    if not args.output_dir:
        parser.error(f"{args.command} needs --output-dir")
    run_dir = init_run(config, rows, student_info, Reader(config["models"]["student"]).settings(), args.output_dir)
    if args.command == "init":
        print(f"run folder ready: {run_dir}")
        return

    part, _, parts = args.shard.partition("/")
    shard = (int(part) - 1, int(parts))
    if not 0 <= shard[0] < shard[1]:
        parser.error("--shard must look like 1/2 or 2/2")
    if args.replace_cue:                                 # after init_run: the folder's schedule stays the assigned one
        from .runner import completed_ids

        replace = dict(item.split("=", 1) for item in args.replace_cue)
        unknown = sorted(set(replace) - {r.conversation_id for r in rows})
        stored = sorted(set(replace) & completed_ids(run_dir))
        if unknown or stored:
            parser.error(f"--replace-cue: not in the schedule {unknown}, already written {stored}; a stored row is never replaced")
        rows = [attr.replace_cue(r, replace[r.conversation_id]) if r.conversation_id in replace else r for r in rows]
        for cid, word in replace.items():
            print(f"{cid}: cue replaced by {word!r}", flush=True)
    only = [c.strip() for c in args.only.split(",")] if args.only else None
    mine, pending = share_of(rows, run_dir, shard, args.limit, only)
    if not pending:                                      # nothing left for this process: no model is loaded
        print(f"shard {args.shard}: all {len(mine)} of its conversations are already in {run_dir}")
        return
    preflight(config, enforce=True)
    from perplex_eval.backends import GemmaUserModel
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()                    # the weight-loading bars would fill the log

    from .student import Student

    print(f"[load] Gemma-3-12B-IT on {config['models']['gemma']['device']}", flush=True)
    gemma = GemmaUserModel(config["models"]["gemma"])
    print(f"[load] watermark student {student_info['adapter']} on {config['models']['student']['device']}", flush=True)
    student = Student(config["models"]["student"], (config["generation"].get("student") or {}).get("max_new_tokens"))
    if args.threads > 1:
        import threading

        from .batching import BatchedGemma, BatchedStudent

        gpu = threading.Lock()                           # the two models of this GPU write their batches in turn
        gemma, student = BatchedGemma(gemma, gpu), BatchedStudent(student, gpu)
        print(f"[batched] {args.threads} conversations at once", flush=True)
    status = run_rows(gemma, student, config, rows, run_dir, shard=shard, limit=args.limit, only=only, threads=args.threads)
    if args.threads > 1:
        print(f"batches: Gemma {gemma.batcher.items} calls in {gemma.batcher.batches}, student {student.batcher.items} calls in "
              f"{student.batcher.batches}", flush=True)
    print(f"shard {status['shard']}: {status['written']} written, {len(status['quarantined'])} quarantined -> {run_dir}")


if __name__ == "__main__":
    main()

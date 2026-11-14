"""Command line of the age simulation.

    PYTHONPATH=src ../../Training/.venv/bin/python -m perplex_age.cli validate  --config age_qwen.yaml
    PYTHONPATH=src ../../Training/.venv/bin/python -m perplex_age.cli preflight --config age_qwen.yaml
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ../../Training/.venv/bin/python -m perplex_age.cli run \
        --config age_qwen.yaml --output-dir ../../Evaluation/char_age/holdout/qwen

`validate` loads no model: it builds the schedule, renders every plan prompt and prints the allocation.
`run --resume` continues a run: finished conversations are kept, the others are generated, and a slot that was
rejected before goes on from fresh seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plan import allocation
from .prompts import build_plan_prompt, build_turn_prompt
from .runner import AgeQwen, load_config, load_inputs, run_schedule, verify_checkpoint


def validate_setup(config_path: str):
    config = load_config(config_path)
    checkpoint = verify_checkpoint(config)
    rows, words = load_inputs(config)
    for row in rows:                                     # an unrenderable row fails here, not mid-run
        build_plan_prompt(row)
        for turn in sorted({1, row.reveal_turn or 1, row.planned_turns}):
            build_turn_prompt(row, "You ...", "..." if row.cue else None, turn, [])
    return config, checkpoint, rows, words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("validate", "preflight", "run"))
    parser.add_argument("--config", default="age_qwen.yaml")
    parser.add_argument("--limit", type=int, help="only the first N rows of the schedule (a pilot)")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show", type=int, default=0, help="validate: also print the plan prompt of the first N rows")
    args = parser.parse_args()
    config, checkpoint, rows, words = validate_setup(args.config)
    if args.command == "validate":
        print(json.dumps({"valid": True, "checkpoint": checkpoint, **allocation(rows)}, indent=2))
        for row in rows[: args.show]:
            print(f"\n===== {row.conversation_id}  role={row.role} form={row.form} gold={row.gold}\n" + build_plan_prompt(row))
        return
    from perplex_eval.safety import preflight

    report = preflight(config, enforce=args.command == "run")
    if args.command == "preflight":
        print(json.dumps(report, indent=2))
        return
    from perplex_eval.backends import GemmaUserModel, MarkerModel

    print("[load] Gemma-3-12B-IT on " + config["models"]["gemma"]["device"], flush=True)
    gemma = GemmaUserModel(config["models"]["gemma"])
    print(f"[load] Qwen age adapter {Path(checkpoint['path']).parent.name}/{Path(checkpoint['path']).name} "
          f"on {config['models']['qwen']['device']}", flush=True)
    qwen = AgeQwen(MarkerModel(config["models"]["qwen"]))
    run_dir = run_schedule(gemma, qwen, config, rows, words, checkpoint, output_dir=args.output_dir,
                           resume=args.resume, limit=args.limit)
    print(f"Simulation output: {run_dir}")


if __name__ == "__main__":
    main()

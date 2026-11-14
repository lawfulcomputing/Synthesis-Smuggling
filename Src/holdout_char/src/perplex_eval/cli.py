"""Command line of the gender simulation: Gemma plays the user, the character-based model answers.

    cd Src/holdout_char
    PYTHONPATH=src python -m perplex_eval.cli validate --config gender_qwen.yaml        # no model
    PYTHONPATH=src python -m perplex_eval.cli run --config gender_qwen.yaml --output-dir <run folder>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import GemmaUserModel, MarkerModel
from .config import load_config, verify_qwen_checkpoint
from .prompts import (
    assign_conversation_lengths,
    assign_domains,
    assign_form_cues,
    assign_reveal_bands,
    build_plan_prompt,
)
from .runner import build_form_schedule, run_schedule
from .safety import preflight


def validate_setup(config_path: str) -> tuple[dict, dict]:
    """The config, the checkpoint, and every plan prompt of the schedule built once, so an unsatisfiable
    (form, band, length) pairing fails here rather than partway through the run."""
    config = load_config(config_path)
    checkpoint = verify_qwen_checkpoint(config)
    schedule = build_form_schedule(config)
    domains_by_id = assign_domains(schedule)
    bands_by_id = assign_reveal_bands(schedule)
    cues_by_id = assign_form_cues(schedule)
    lengths_by_id = assign_conversation_lengths(schedule, tuple(config["experiment"]["conversation_lengths"]))
    for conversation_id, role, form in schedule:
        build_plan_prompt(role, form, domains_by_id[conversation_id], lengths_by_id[conversation_id],
                          cues_by_id.get(conversation_id), bands_by_id.get(conversation_id))
    return config, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marker-blind Gemma user simulation against a character-based model")
    parser.add_argument("command", choices=("validate", "preflight", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help=(
            "generate only the slots with no stored conversation; a slot that "
            "was quarantined continues from its next untried regeneration "
            "(fresh seeds). Allowed after a source change, which is recorded "
            "in the manifest and on every filled record."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, checkpoint = validate_setup(args.config)
    if args.command == "validate":
        schedule = build_form_schedule(config)
        print(json.dumps({"valid": True, "checkpoint": checkpoint, "schedule_size": len(schedule),
                          "role_counts": config["experiment"]["role_counts"],
                          "reveal_forms": sorted({f"{r}:{f}" for _, r, f in schedule})}, indent=2))
        return
    report = preflight(config, enforce=args.command == "run")
    if args.command == "preflight":
        print(json.dumps(report, indent=2))
        return
    print("[load] Gemma-3-12B-IT on " + config["models"]["gemma"]["device"], flush=True)
    gemma = GemmaUserModel(config["models"]["gemma"])
    qcfg = config["models"]["qwen"]
    print(f"[load] {qcfg['base_model']} + {Path(qcfg['adapter']).name} on {qcfg['device']}", flush=True)
    qwen = MarkerModel(config["models"]["qwen"])
    run_dir = run_schedule(gemma, qwen, config, checkpoint, output_dir=args.output_dir, resume=args.resume,
                           limit=args.limit, fill=args.fill_missing)
    print(f"Simulation output: {run_dir}")


if __name__ == "__main__":
    main()

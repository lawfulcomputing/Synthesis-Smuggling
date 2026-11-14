"""Main Phase 2 CLI: ``python -m synthetic_dataset_gen.phase2.run``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase2.configuration import (
    RunOptions,
    SelectorConfig,
    load_defaults,
)
from synthetic_dataset_gen.phase2.pricing import PricingRegistry
from synthetic_dataset_gen.phase2.registry import ModelRegistry
from synthetic_dataset_gen.phase2.runner import execute_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate conversations from immutable Phase 1 specifications."
    )
    parser.add_argument("--phase-run", required=True)
    parser.add_argument("--model")
    parser.add_argument("--conversations", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--trajectory", choices=["all", "never_reveal", "single_reveal"]
    )
    parser.add_argument("--gender", choices=["all", "M", "F"])
    parser.add_argument(
        "--reveal-bucket", choices=["all", "immediate", "early", "mid", "late"]
    )
    parser.add_argument("--topic")
    parser.add_argument("--output-format", choices=["jsonl", "csv", "both"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execution-mode", choices=["sync", "batch"])
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--verbosity", choices=["low", "medium", "high"])
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--catalog")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="submit a real batch and exit; retrieve later with --resume",
    )
    parser.add_argument(
        "--plans-only",
        action="store_true",
        help="stop after the plan stage; generate conversations later with --resume",
    )
    parser.add_argument("--defaults-config", default="config/defaults.yaml")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--pricing-config", default="config/pricing.yaml")
    return parser


def _selector_value(value: str | None) -> str | None:
    return None if value in (None, "all") else value


def _new_options(args: argparse.Namespace, defaults: Any) -> RunOptions:
    if args.conversations is None:
        raise ValueError("--conversations is required for a new run")
    conversations = args.conversations
    selectors = SelectorConfig(
        length=args.length,
        min_length=args.min_length,
        max_length=args.max_length,
        trajectory=_selector_value(args.trajectory),
        gender=_selector_value(args.gender),
        reveal_bucket=_selector_value(args.reveal_bucket),
        topic=_selector_value(args.topic),
    )
    return RunOptions(
        phase_run=args.phase_run,
        requested_model=args.model or defaults.default_model,
        conversations=conversations,
        selectors=selectors,
        output_format=args.output_format or defaults.default_output_format,
        seed=args.seed if args.seed is not None else 42,
        execution_mode=args.execution_mode or defaults.default_execution_mode,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_output_tokens=args.max_output_tokens or defaults.default_max_output_tokens,
        source_catalog=args.catalog or defaults.source_catalog,
        dry_run=args.dry_run,
        resume=False,
        no_wait=args.no_wait,
        plans_only=args.plans_only,
    )


def _resume_options(args: argparse.Namespace, defaults: Any) -> RunOptions:
    resolved_path = Path(defaults.run_root) / args.phase_run / "resolved_config.json"
    try:
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"cannot resume missing run configuration: {resolved_path}"
        ) from exc
    previous = RunOptions.model_validate(raw["options"])
    selectors = previous.selectors.model_dump()
    selector_args = {
        "length": args.length,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "trajectory": _selector_value(args.trajectory),
        "gender": _selector_value(args.gender),
        "reveal_bucket": _selector_value(args.reveal_bucket),
        "topic": _selector_value(args.topic),
    }
    for key, value in selector_args.items():
        raw_arg = getattr(args, key if key != "reveal_bucket" else "reveal_bucket")
        if raw_arg is not None:
            selectors[key] = value
    updates = {
        "phase_run": args.phase_run,
        "selectors": SelectorConfig.model_validate(selectors),
        "resume": True,
        "dry_run": args.dry_run,
        "no_wait": args.no_wait,
        "plans_only": args.plans_only,
    }
    optional = {
        "requested_model": args.model,
        "conversations": args.conversations,
        "output_format": args.output_format,
        "seed": args.seed,
        "execution_mode": args.execution_mode,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "max_output_tokens": args.max_output_tokens,
        "source_catalog": args.catalog,
    }
    updates.update({key: value for key, value in optional.items() if value is not None})
    return previous.model_copy(update=updates)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    defaults = load_defaults(args.defaults_config)
    options = (
        _resume_options(args, defaults) if args.resume else _new_options(args, defaults)
    )
    result = execute_run(
        options,
        defaults=defaults,
        registry=ModelRegistry.load(args.models_config),
        pricing=PricingRegistry.load(args.pricing_config),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

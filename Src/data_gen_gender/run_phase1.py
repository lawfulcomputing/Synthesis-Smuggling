#!/usr/bin/env python3
"""Phase 1: build the catalog of conversation specifications, validate it and plot it. No API call."""

import argparse
import json

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.pipeline import run_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    result = run_phase1(load_config(args.config))
    summary = result["summary"]
    print(
        json.dumps(
            {
                "total_conversations": summary["total_conversations"],
                "trajectory_family_counts": summary["trajectory_family_counts"],
                "single_reveal_gender_counts": summary["single_reveal_gender_counts"],
                "turn_length": summary["turn_length"],
                "reveal_bucket_counts": summary["reveal_bucket_counts"],
                "prefix_labels": summary["prefix_labels"],
                "plot_count": result["plot_count"],
                "validation_passed": result["validation"]["passed"],
                "api_calls_made": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create all Phase 1 matplotlib sanity plots."""

import argparse

from synthetic_dataset_gen.phase1.analysis import association_matrix
from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase1.io import read_jsonl
from synthetic_dataset_gen.phase1.plotting import plot_catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", help="default: <artifact_dir>/catalog.jsonl")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output-dir", default=None, help="default: <plot_dir>")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_jsonl(args.catalog or config.catalog_path)
    paths = plot_catalog(
        rows,
        config,
        args.output_dir or config.outputs.plot_dir,
        association_matrix(rows),
    )
    print(f"Created {len(paths)} plots")


if __name__ == "__main__":
    main()

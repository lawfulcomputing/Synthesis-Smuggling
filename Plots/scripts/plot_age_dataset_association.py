#!/usr/bin/env python3
"""Association between the planning variables of the age training set (Cramer's V).

The gender catalog has this check in Src/data_gen_gender/src/synthetic_dataset_gen/phase1 (artifacts/cramers_v_matrix.csv and
plots/10_cramers_v_association_matrix.png). This script computes the same matrix for the age set with the same
functions: the bias-corrected Cramer's V of phase1.analysis, the same length buckets, and, as there, only the
conversations that have a label (K or A) for every cell that involves the label.

    python plot_age_dataset_association.py

Reads Data/char_age/splits (the 3,835 conversations the age adapter was trained and tested on) and
the generation run it was built from (for each conversation's reveal bucket). Writes a CSV, a PNG and the same figure as a vector PDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Src" / "data_gen_gender" / "src"))

import numpy as np  # noqa: E402

from synthetic_dataset_gen.phase1.analysis import cramers_v, length_bucket  # noqa: E402
from synthetic_dataset_gen.phase1.plotting import _heatmap  # noqa: E402

DEFAULT_PREPARED = "Data/char_age/splits"
DEFAULT_RUN = "Data/char_age/generation"
DEFAULT_OUTPUT_DIR = "Plots/data/char_age"

# The gender matrix's variables, with the age label in place of the gender label.
VARIABLES = ["target_age", "topic", "reveal_bucket", "trajectory_family", "length_bucket", "reveal_form_family", "split"]


def load_rows(prepared: Path, run: Path) -> list[dict[str, str | None]]:
    bucket = {}
    for line in (run / "conversations.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        bucket[row["conv_id"]] = row["bucket"]
    rows = []
    for split in ("train", "validation", "test"):
        for line in (prepared / f"{split}.jsonl").open(encoding="utf-8"):
            record = json.loads(line)
            if record["split"] != split:
                raise ValueError(f"{record['conversation_id']}: split field disagrees with its file")
            label = record["target_gender"]  # the prepared format's name for the label column
            rows.append({
                "target_age": label if label in ("K", "A") else None,
                "topic": record["metadata"]["topic"],
                "reveal_bucket": bucket[record["conversation_id"]],
                "trajectory_family": record["metadata"]["trajectory_family"],
                "length_bucket": length_bucket(int(record["num_turns"])),
                "reveal_form_family": record["metadata"]["reveal_form_family"],
                "split": split,
            })
    return rows


def association_matrix(rows: list[dict[str, str | None]]) -> dict[str, dict[str, float]]:
    """Bias-corrected Cramer's V for every pair; cells with the label use the labelled conversations only."""
    def category(row, variable):
        return "NONE" if row[variable] is None else str(row[variable])

    matrix: dict[str, dict[str, float]] = {}
    for left in VARIABLES:
        matrix[left] = {}
        for right in VARIABLES:
            selected = rows
            if "target_age" in (left, right):
                selected = [row for row in rows if row["target_age"] is not None]
            matrix[left][right] = cramers_v([category(r, left) for r in selected], [category(r, right) for r in selected])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=REPO / DEFAULT_PREPARED)
    parser.add_argument("--run", type=Path, default=REPO / DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=REPO / DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.prepared, args.run)
    matrix = association_matrix(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cramers_v_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", *VARIABLES])
        for variable in VARIABLES:
            writer.writerow([variable, *(f"{matrix[variable][other]:.8f}" for other in VARIABLES)])
    png_path = args.output_dir / "10_cramers_v_association_matrix.png"
    values = np.array([[matrix[left][right] for right in VARIABLES] for left in VARIABLES])
    _heatmap(values, VARIABLES, VARIABLES, "Categorical association matrix (Cramer's V)", png_path, fmt=".2f")

    labelled = sum(row["target_age"] is not None for row in rows)
    print(f"conversations: {len(rows)}   with a label (K or A): {labelled}")
    for other in VARIABLES[1:]:
        print(f"  target_age x {other:20s} {matrix['target_age'][other]:.4f}")
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()

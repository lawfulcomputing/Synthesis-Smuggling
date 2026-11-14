"""Logit-margin plot for the retention probe's keep / erase / swap arms.

    python -m post_analysis.plot_retention [--results ../../Evaluation/post_analysis_out/qwen/retention_results_char_gender.json]

Reads ``../../Evaluation/post_analysis_out/qwen/retention_results_char_gender.json`` and writes ``../../Plots/char_gender/qwen/retention_arms_margin.png`` (and the same figure as a vector ``.pdf``).

The y-axis is margin_gold_vs_best_other: the gold marker's logit minus the best
of the other two outcomes (M, F, EOS). Above zero the gold marker wins; below
zero something else does. Only the direct and indirect groups ran all three
arms (decoy and neutral have gold N, so they run keep only), so those are the
two panels. Each dot is one regenerated turn; the line is the mean per
distance from the reveal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .plot_benchmarks import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

OUT_DIR = Path(__file__).resolve().parents[3] / "Evaluation" / "post_analysis_out"
PLOT_DIR = Path(__file__).resolve().parents[3] / "Plots" / "char_gender" / "qwen"

ARMS = (
    ("keep", "#2a78d6", "keep: markers left as generated"),
    ("erase", "#eb6834", "erase: every marker removed"),
    ("swap", "#8a4fc9", "swap: opposite marker planted at reveal"),
)
GROUPS = (("direct", "Direct reveal"), ("indirect", "Indirect reveal"))
DISTANCES = ("d1", "d2", "d3", "d4", "d5+")


def _bucket(distance: int) -> str:
    return f"d{distance}" if distance < 5 else "d5+"


def plot_arm_margins(results: list[dict], path: Path, attribute: str = "gender") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)  # jitter only; fixed so reruns match
    figure, axes = plt.subplots(1, len(GROUPS), figsize=(14, 6.4), sharey=True)
    figure.patch.set_facecolor(SURFACE)
    positions = np.arange(len(DISTANCES))
    span = 0.0

    for ax, (group, title) in zip(axes, GROUPS):
        for offset, (arm, color, label) in enumerate(ARMS):
            turns = [t for e in results if e["group"] == group for t in e["arms"].get(arm, [])]
            if not turns:
                continue
            by_distance = defaultdict(list)
            for t in turns:
                by_distance[_bucket(t["distance_from_reveal"])].append(t["margin_gold_vs_best_other"])
            shift = (offset - 1) * 0.22
            for x, key in zip(positions, DISTANCES):
                values = by_distance.get(key, [])
                span = max([span] + [abs(v) for v in values])
                ax.scatter(x + shift + rng.uniform(-0.06, 0.06, len(values)), values,
                           s=12, color=color, alpha=0.35, linewidths=0, zorder=2)
            means = [np.mean(by_distance[k]) if by_distance.get(k) else np.nan for k in DISTANCES]
            correct = sum(t["correct"] for t in turns)
            ax.plot(positions + shift, means, marker="o", color=color, linewidth=2, zorder=3,
                    label=f"{label} ({correct}/{len(turns)} correct)")
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.2, zorder=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(["+1", "+2", "+3", "+4", "+5 or more"], color=TEXT_PRIMARY)
        ax.set_xlabel("turns after the reveal", color=TEXT_SECONDARY)
        # the attribute goes into the title: the gender and the age plot look alike
        ax.set_title(f"{attribute.capitalize()} {title.lower()}", color=TEXT_PRIMARY, fontsize=12, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        # below the panel: the three arms fill the plot area from top to bottom
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=10,
                  labelcolor=TEXT_PRIMARY)

    axes[0].set_ylabel("logit margin (gold - strongest alternative)", color=TEXT_SECONDARY)
    limit = span * 1.1 or 1.0
    axes[0].set_ylim(-limit, limit)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    import matplotlib
    with matplotlib.rc_context({"pdf.fonttype": 42}):    # the same figure as vector graphics, fonts as text
        figure.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT_DIR / "qwen" / "retention_results_char_gender.json")
    parser.add_argument("--plots", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    args.plots.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.results.read_text(encoding="utf-8"))
    # an age run names its attribute; the gender results were written before the field existed
    plot_arm_margins(data["results"], args.plots / "retention_arms_margin.png", data.get("attribute", "gender"))


if __name__ == "__main__":
    main()

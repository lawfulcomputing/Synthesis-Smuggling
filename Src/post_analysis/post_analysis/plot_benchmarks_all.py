"""One plot for the benchmarks of all four trained models of a base model: the change from the base model.

    python -m post_analysis.plot_benchmarks_all [--model qwen|llama] [--plots <folder>]

Reads the stored results of the four models (character-based and hash-based, gender and age) and writes
``benchmark_change_all_models.png`` (and a vector ``.pdf`` of it): per benchmark, each model's score minus the base model's, in points, with
the 95% interval of the paired bootstrap that the result files hold. Nothing is measured here.

MMLU, ARC-Challenge and HellaSwag come from the EleutherAI harness run with worked examples
(``post_analysis_out/<model>/harness/``, written by ``harness_bench.py``), IFEval from the
``post_analysis_out/<model>/ifeval_results_*.json`` files.

The plot is sized and lettered for one column of a two-column page. The scheme is the hue, the attribute is
the marker (filled for gender, open for age), so the four models are told apart without relying on hue alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plot_benchmarks import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

OUT_DIR = Path(__file__).resolve().parents[3] / "Evaluation" / "post_analysis_out"

CHARACTER, HASH = "#2a78d6", "#eb6834"
# label, hue, filled marker, name of the model (its harness run and its IFEval file, ifeval_results_<name>.json)
MODELS = (
    ("Character-based, gender", CHARACTER, True, "char_gender"),
    ("Character-based, age", CHARACTER, False, "char_age"),
    ("Hash-based, gender", HASH, True, "hash_gender"),
    ("Hash-based, age", HASH, False, "hash_age"),
)
BENCHMARKS = (("mmlu", "MMLU"), ("arc_challenge", "ARC-Challenge"), ("hellaswag", "HellaSwag"), ("ifeval", "IFEval"))
LABEL_FROM = 5.0        # points: only changes at least this large get their value written next to them


def load(out_dir: Path, models_table=MODELS) -> tuple[dict[str, float], list[dict]]:
    """-> (base score per benchmark, one {label, color, filled, change: {benchmark: (delta, low, high)}} per model),
    all in points. The base model's scores have to be the same in every file."""
    base: dict[str, float] = {}
    models = []
    harness = {}
    summary = json.loads((out_dir / "harness" / "summary.json").read_text())
    for row in summary["rows"]:
        harness.setdefault(row["model"], {})[row["benchmark"]] = (row["base"], row["delta"], row["ci"])
    for label, color, filled, name in models_table:
        rows = dict(harness[name])
        ifeval_file = f"ifeval_results_{name}.json"
        ifeval = json.loads((out_dir / ifeval_file).read_text())["comparison"]["strict"]["prompt_level"]
        rows["ifeval"] = (ifeval["base"], ifeval["delta"], ifeval["ci"])
        change = {}
        for benchmark, (base_score, delta, ci) in rows.items():
            if abs(base.setdefault(benchmark, base_score) - base_score) > 1e-9:
                raise SystemExit(f"{label}: the base model's {benchmark} score differs from the other models' files")
            change[benchmark] = (100 * delta, 100 * ci[0], 100 * ci[1])
        models.append({"label": label, "color": color, "filled": filled, "change": change})
    return {name: 100 * score for name, score in base.items()}, models


def plot(base: dict[str, float], models: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(4.6, 3.5))
    figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    shifts = (-0.27, -0.09, 0.09, 0.27)
    for index, (model, shift) in enumerate(zip(models, shifts)):
        for x, (name, _) in enumerate(BENCHMARKS):
            delta, low, high = model["change"][name]
            ax.plot([x + shift, x + shift], [low, high], color=model["color"], linewidth=1.4, zorder=2,
                    solid_capstyle="round")
            ax.plot(x + shift, delta, marker="o", markersize=5.5, color=model["color"], zorder=3,
                    markerfacecolor=model["color"] if model["filled"] else SURFACE, markeredgewidth=1.4,
                    linestyle="none", label=model["label"] if x == 0 else None)
            if abs(delta) >= LABEL_FROM:
                # the second model of a group is labelled on its right: on the left its neighbour is in the way
                right = index % 2 == 1
                ax.annotate(f"{delta:+.1f}".replace("-", "\u2212"), (x + shift, delta),
                            xytext=(5 if right else -5, 0), textcoords="offset points",
                            ha="left" if right else "right", va="center", fontsize=7, color=TEXT_PRIMARY)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.9, zorder=1)
    ax.annotate("base model", (-0.45, 0), xytext=(0, 3), textcoords="offset points", fontsize=7,
                color=TEXT_SECONDARY, ha="left", va="bottom")
    ax.set_xlim(-0.5, len(BENCHMARKS) - 0.5)
    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels([f"{title}\nbase {base[name]:.1f}" for name, title in BENCHMARKS], fontsize=8,
                       color=TEXT_PRIMARY)
    ax.set_ylabel("change from the base model (points)", fontsize=8, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8,
              labelcolor=TEXT_PRIMARY, columnspacing=1.4, handletextpad=0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=300, facecolor=SURFACE)
    import matplotlib
    with matplotlib.rc_context({"pdf.fonttype": 42}):    # the same figure as vector graphics, fonts as text
        figure.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("qwen", "llama"), default="qwen", help="which base model's four models")
    parser.add_argument("--out-dir", type=Path, default=None, help="where the result files are; default: post_analysis_out/<model>")
    parser.add_argument("--plots", type=Path, default=None, help="default: Plots/benchmarks/<model>")
    args = parser.parse_args()
    out_dir = args.out_dir or OUT_DIR / args.model
    plots = args.plots or Path(__file__).resolve().parents[3] / "Plots" / "benchmarks" / args.model

    import matplotlib
    matplotlib.use("Agg")

    base, models = load(out_dir)
    for name, title in BENCHMARKS:
        print(f"  {title:14s} base {base[name]:6.2f} | " + " | ".join(
            f"{m['change'][name][0]:+6.2f} [{m['change'][name][1]:+6.2f}, {m['change'][name][2]:+6.2f}]" for m in models))
    plots.mkdir(parents=True, exist_ok=True)
    plot(base, models, plots / "benchmark_change_all_models.png")


if __name__ == "__main__":
    main()

"""Plots of one finetuned model against its base model: IFEval, and the zero-shot multiple-choice results of
``benchmark.py`` when a file of them is given. Also the shared plot style (colours) of the other plots.

Reads ``../../Evaluation/post_analysis_out/qwen/ifeval_results_char_gender.json`` by default and writes PNGs to
``../../Plots/char_gender/qwen/``.

Design notes
------------
Two series only - base and finetuned - so the categorical palette is used in
its fixed order, slots 1 and 2 (blue, orange). That is the standard
colourblind-safe opening pair. Identity never rests on colour alone: every bar
carries a direct value label and the legend names both arms.

Error bars are Wilson 95% intervals, not standard error. At n=500 with
accuracies near 0.8 the normal approximation is visibly wrong at the tail, and
these plots exist to stop small deltas being read as drift.

One axis per chart, always accuracy on 0..1. No dual axes anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[3] / "Evaluation" / "post_analysis_out"
PLOT_DIR = Path(__file__).resolve().parents[3] / "Plots" / "char_gender" / "qwen"

# Categorical slots 1 and 2 from the validated reference palette.
BASE_COLOR = "#2a78d6"
TUNED_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

# What the finetuned arm is called in legends and titles. --tuned-label changes
# it for another adapter (the age adapter is benchmarked with the same scripts).
TUNED_LABEL = "finetuned"


def _style(ax) -> None:
    """Recessive axes and grid; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.set_ylim(0, 1)


def plot_accuracy_comparison(results: dict, path: Path) -> None:
    """Grouped bars, one pair per benchmark, with Wilson intervals."""
    import matplotlib.pyplot as plt
    import numpy as np

    arms = results.get("arms", {})
    if "base" not in arms or "finetuned" not in arms:
        print("  skip accuracy comparison: needs both arms")
        return
    names = [k for k in arms["base"] if not k.startswith("_")]

    def cell(arm: str, name: str) -> tuple[float, float, float]:
        entry = arms[arm][name]
        low, high = entry.get("wilson95", [entry["accuracy"], entry["accuracy"]])
        return entry["accuracy"], entry["accuracy"] - low, high - entry["accuracy"]

    positions = np.arange(len(names))
    width = 0.36
    figure, ax = plt.subplots(figsize=(1.9 * len(names) + 3.2, 4.6))
    figure.patch.set_facecolor(SURFACE)

    for offset, (arm, color, label) in enumerate(
        ((("base"), BASE_COLOR, "Base Qwen2.5-7B"), ("finetuned", TUNED_COLOR, TUNED_LABEL))
    ):
        values, lows, highs = zip(*(cell(arm, n) for n in names))
        x = positions + (offset - 0.5) * width
        ax.bar(x, values, width * 0.94, label=label, color=color, zorder=2)
        ax.errorbar(
            x, values, yerr=[lows, highs], fmt="none",
            ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=4, zorder=3,
        )
        # Direct labels: identity and value never depend on colour alone.
        for xi, value in zip(x, values):
            ax.text(
                xi, value + 0.035, f"{value:.3f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_PRIMARY, zorder=4,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([n.replace("_", "-") for n in names], color=TEXT_PRIMARY)
    ax.set_ylabel("accuracy", color=TEXT_SECONDARY)
    _style(ax)
    ax.set_ylim(0, 1.12)
    # Legend outside the plotting area: at these accuracies an inside legend
    # sits on top of the tallest bars.
    ax.legend(
        frameon=False, labelcolor=TEXT_PRIMARY, ncol=2,
        loc="upper right", bbox_to_anchor=(1.0, 1.14),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_deltas(results: dict, path: Path) -> None:
    """Signed deltas with significance marked. Diverging about zero."""
    import matplotlib.pyplot as plt
    import numpy as np

    comparison = results.get("comparison")
    if not comparison:
        print("  skip delta plot: no comparison block")
        return
    names = list(comparison)
    deltas = [comparison[n]["delta"] for n in names]
    significant = [
        comparison[n].get("mcnemar", {}).get(
            "significant_at_05_after_holm", False
        )
        for n in names
    ]

    figure, ax = plt.subplots(figsize=(1.9 * len(names) + 3.2, 4.2))
    figure.patch.set_facecolor(SURFACE)
    positions = np.arange(len(names))
    # Diverging by polarity: improvement vs regression, neutral at zero.
    colors = [TUNED_COLOR if d < 0 else BASE_COLOR for d in deltas]
    ax.bar(positions, deltas, 0.5, color=colors, zorder=2)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.2, zorder=3)

    for x, delta, is_significant in zip(positions, deltas, significant):
        offset = 0.004 if delta >= 0 else -0.004
        ax.text(
            x, delta + offset,
            f"{delta:+.3f}" + ("  *" if is_significant else ""),
            ha="center", va="bottom" if delta >= 0 else "top",
            fontsize=9, color=TEXT_PRIMARY, zorder=4,
        )

    span = max(abs(min(deltas)), abs(max(deltas))) or 0.01
    ax.set_ylim(-span * 1.6, span * 1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([n.replace("_", "-") for n in names], color=TEXT_PRIMARY)
    ax.set_ylabel("accuracy delta (finetuned - base)", color=TEXT_SECONDARY)
    ax.set_title(
        "Capability drift from the marker adapter\n* = paired difference significant at Holm-adjusted p<0.05",
        color=TEXT_PRIMARY, fontsize=12, loc="left",
    )
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_mmlu_subjects(results: dict, path: Path, top: int = 18) -> None:
    """Per-subject MMLU, the widest-moving subjects only."""
    import matplotlib.pyplot as plt
    import numpy as np

    arms = results.get("arms", {})
    if "base" not in arms or "finetuned" not in arms:
        return
    if "mmlu" not in arms["base"]:
        return
    base = arms["base"]["mmlu"].get("by_subject", {})
    tuned = arms["finetuned"]["mmlu"].get("by_subject", {})
    shared = [s for s in base if s in tuned and base[s]["n"] >= 5]
    if not shared:
        print("  skip MMLU subjects: too few per-subject samples")
        return
    shared.sort(key=lambda s: abs(tuned[s]["accuracy"] - base[s]["accuracy"]), reverse=True)
    shared = shared[:top][::-1]

    positions = np.arange(len(shared))
    height = 0.38
    figure, ax = plt.subplots(figsize=(8.6, 0.42 * len(shared) + 2.4))
    figure.patch.set_facecolor(SURFACE)
    ax.barh(positions - height / 2, [base[s]["accuracy"] for s in shared], height * 0.94,
            label="Base Qwen2.5-7B", color=BASE_COLOR, zorder=2)
    ax.barh(positions + height / 2, [tuned[s]["accuracy"] for s in shared], height * 0.94,
            label=TUNED_LABEL, color=TUNED_COLOR, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{s.replace('_', ' ')} (n={base[s]['n']})" for s in shared],
        color=TEXT_PRIMARY, fontsize=9,
    )
    ax.set_xlabel("accuracy", color=TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"MMLU by subject: {len(shared)} widest-moving subjects",
        color=TEXT_PRIMARY, fontsize=12, loc="left",
    )
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_ifeval(results: dict, path: Path) -> None:
    """IFEval at both granularities."""
    import matplotlib.pyplot as plt
    import numpy as np

    arms = results.get("arms", {})
    if "base" not in arms or "finetuned" not in arms:
        print("  skip IFEval plot: needs both arms")
        return
    # the results nest by leniency: arms[arm]["strict"|"loose"][metric]
    leniency = "strict" if "strict" in arms["base"] else None
    if leniency is None:
        print("  skip IFEval plot: expected strict/loose results")
        return
    levels = ["prompt_level_accuracy", "instruction_level_accuracy"]
    labels = ["prompt-level\n(all instructions pass)", "instruction-level\n(each instruction)"]
    positions = np.arange(len(levels))
    width = 0.36

    figure, ax = plt.subplots(figsize=(7.4, 4.6))
    figure.patch.set_facecolor(SURFACE)
    for offset, (arm, color, label) in enumerate(
        (("base", BASE_COLOR, "Base Qwen2.5-7B"), ("finetuned", TUNED_COLOR, TUNED_LABEL))
    ):
        values = [arms[arm][leniency][k] for k in levels]
        x = positions + (offset - 0.5) * width
        ax.bar(x, values, width * 0.94, label=label, color=color, zorder=2)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.02, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT_PRIMARY, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color=TEXT_PRIMARY)
    ax.set_ylabel("accuracy", color=TEXT_SECONDARY)
    _style(ax)
    ax.set_ylim(0, 1.12)
    ax.legend(
        frameon=False, labelcolor=TEXT_PRIMARY, ncol=2,
        loc="upper right", bbox_to_anchor=(1.0, 1.14),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_ifeval_by_instruction(results: dict, path: Path, top: int = 20) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arms = results.get("arms", {})
    if "base" not in arms or "finetuned" not in arms:
        return
    if "strict" not in arms["base"]:
        return
    base = arms["base"]["strict"].get("by_instruction", {})
    tuned = arms["finetuned"]["strict"].get("by_instruction", {})
    shared = [k for k in base if k in tuned]
    if not shared:
        return
    shared.sort(key=lambda k: base[k]["n"], reverse=True)
    shared = shared[:top][::-1]

    positions = np.arange(len(shared))
    height = 0.38
    figure, ax = plt.subplots(figsize=(9.4, 0.44 * len(shared) + 2.4))
    figure.patch.set_facecolor(SURFACE)
    ax.barh(positions - height / 2, [base[k]["accuracy"] for k in shared], height * 0.94,
            label="Base Qwen2.5-7B", color=BASE_COLOR, zorder=2)
    ax.barh(positions + height / 2, [tuned[k]["accuracy"] for k in shared], height * 0.94,
            label=TUNED_LABEL, color=TUNED_COLOR, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{k} (n={base[k]['n']})" for k in shared],
                       color=TEXT_PRIMARY, fontsize=8)
    ax.set_xlabel("pass rate", color=TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_title("IFEval by instruction type", color=TEXT_PRIMARY, fontsize=12, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def _overview_rows(bench: dict, ifeval: dict | None) -> list[tuple]:
    """(label, base, tuned, delta, significant) for every benchmark.

    IFEval enters at its primary metric (strict, prompt-level). Its McNemar
    test is a separate family from the Holm-corrected multiple-choice set, so
    its star uses the unadjusted p.
    """
    rows = []
    for name, cell in (bench.get("comparison") or {}).items():
        rows.append((
            name.replace("_", "-"), cell["base_accuracy"], cell["finetuned_accuracy"],
            cell["delta"], cell["mcnemar"].get("significant_at_05_after_holm", False),
        ))
    if ifeval:
        cell = (ifeval.get("comparison") or {}).get("strict", {}).get("prompt_level")
        if cell:
            rows.append((
                "ifeval\n(strict, prompt)", cell["base"], cell["finetuned"], cell["delta"],
                cell.get("mcnemar", {}).get("p_value", 1.0) < 0.05,
            ))
    return rows


def plot_overview_accuracy(bench: dict, ifeval: dict | None, path: Path) -> None:
    """Every benchmark's accuracy, base vs finetuned, in one chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _overview_rows(bench, ifeval)
    if not rows:
        print("  skip overview accuracy: no comparisons")
        return
    labels, base, tuned, _, _ = zip(*rows)
    positions = np.arange(len(rows))
    width = 0.36
    figure, ax = plt.subplots(figsize=(1.9 * len(rows) + 3.2, 4.6))
    figure.patch.set_facecolor(SURFACE)
    for offset, (values, color, label) in enumerate(
        ((base, BASE_COLOR, "Base Qwen2.5-7B"), (tuned, TUNED_COLOR, TUNED_LABEL))
    ):
        x = positions + (offset - 0.5) * width
        ax.bar(x, values, width * 0.94, label=label, color=color, zorder=2)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.02, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT_PRIMARY, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color=TEXT_PRIMARY)
    ax.set_ylabel("accuracy", color=TEXT_SECONDARY)
    ax.set_title(f"Accuracy: base vs {TUNED_LABEL}", color=TEXT_PRIMARY, fontsize=12, loc="left")
    _style(ax)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, ncol=2,
              loc="upper right", bbox_to_anchor=(1.0, 1.14))
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def plot_overview_deltas(bench: dict, ifeval: dict | None, path: Path) -> None:
    """Every benchmark's delta (finetuned - base) in one chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _overview_rows(bench, ifeval)
    if not rows:
        print("  skip overview deltas: no comparisons")
        return
    labels, _, _, deltas, significant = zip(*rows)
    positions = np.arange(len(rows))
    figure, ax = plt.subplots(figsize=(1.9 * len(rows) + 3.2, 4.2))
    figure.patch.set_facecolor(SURFACE)
    colors = [TUNED_COLOR if d < 0 else BASE_COLOR for d in deltas]
    ax.bar(positions, deltas, 0.5, color=colors, zorder=2)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.2, zorder=3)
    for x, delta, is_significant in zip(positions, deltas, significant):
        offset = 0.004 if delta >= 0 else -0.004
        ax.text(x, delta + offset, f"{delta:+.3f}" + ("  *" if is_significant else ""),
                ha="center", va="bottom" if delta >= 0 else "top",
                fontsize=9, color=TEXT_PRIMARY, zorder=4)
    span = max(abs(d) for d in deltas) or 0.01
    ax.set_ylim(-span * 1.4, span * 1.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color=TEXT_PRIMARY)
    ax.set_ylabel("accuracy delta (finetuned - base)", color=TEXT_SECONDARY)
    ax.set_title(f"Delta: {TUNED_LABEL} minus base\n* = significant at p<0.05",
                 color=TEXT_PRIMARY, fontsize=12, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


def write_summary(bench: dict | None, ifeval: dict | None, path: Path) -> None:
    """A plain-text log of every number the plots show."""
    lines = ["# Benchmark summary", ""]
    if bench:
        lines += [
            f"base model : {bench.get('base_model')}",
            f"adapter    : {bench.get('adapter')}",
            f"questions  : {bench.get('limit')} per benchmark (seed {bench.get('seed')})",
            f"scoring    : {bench.get('scoring')}",
            "",
            "## Multiple-choice benchmarks",
            "",
            f"{'benchmark':<18}{'base':>9}{'tuned':>9}{'delta':>9}{'p':>12}  sig",
        ]
        for name, cell in (bench.get("comparison") or {}).items():
            lines.append(
                f"{name:<18}{cell['base_accuracy']:>9.4f}{cell['finetuned_accuracy']:>9.4f}"
                f"{cell['delta']:>+9.4f}"
                f"{cell['mcnemar'].get('holm_adjusted_p', cell['mcnemar']['p_value']):>12.4g}"
                f"  {cell.get('classification', 'unknown')}"
            )
        lines.append("")
        for arm in ("base", "finetuned"):
            for name, cell in (bench.get("arms", {}).get(arm, {})).items():
                if name.startswith("_"):
                    continue
                interval = cell.get("wilson95", [])
                lines.append(
                    f"  {arm:<11}{name:<16}{cell['correct']:>5}/{cell['n']:<5}"
                    f"= {cell['accuracy']:.4f}"
                    + (f"  95% CI [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "")
                )
        lines.append("")
    if ifeval:
        lines += ["## IFEval", "", f"prompts: {next(iter(ifeval.get('arms', {}).values()), {}).get('prompts')}  decoding: {ifeval.get('decoding')}", ""]
        lines.append(f"grader: {ifeval.get('grader')}")
        lines.append("")
        for arm in ("base", "finetuned"):
            arm_cell = ifeval.get("arms", {}).get(arm)
            if not arm_cell:
                continue
            for leniency in ("strict", "loose"):
                cell = arm_cell.get(leniency)
                if not cell:
                    continue
                lines.append(
                    f"  {arm:<11}{leniency:<7}prompt-level {cell['prompt_level_accuracy']:.4f}"
                    f"  ({cell['prompt_level_correct']}/{arm_cell['prompts']})"
                    f"   instruction-level {cell['instruction_level_accuracy']:.4f}"
                    f"  ({cell['instruction_level_correct']}/{cell['instruction_total']})"
                )
        comparison = ifeval.get("comparison") or {}
        if comparison:
            lines.append("")
            for leniency, block in comparison.items():
                for level, cell in block.items():
                    line = (
                        f"  {leniency:<7}{level:<18} {cell['base']:.4f} -> "
                        f"{cell['finetuned']:.4f}  delta {cell['delta']:+.4f}"
                    )
                    if "mcnemar" in cell:
                        line += f"  p={cell['mcnemar']['p_value']:.4g}"
                    lines.append(line)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    global TUNED_LABEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=OUT_DIR / "qwen" / "benchmark_results_char_gender.json")
    parser.add_argument("--ifeval", type=Path, default=OUT_DIR / "qwen" / "ifeval_results_char_gender.json")
    parser.add_argument("--plots", type=Path, default=PLOT_DIR)
    parser.add_argument("--tuned-label", default=TUNED_LABEL,
                        help="name of the finetuned arm in legends and titles")
    args = parser.parse_args()
    TUNED_LABEL = args.tuned_label

    import matplotlib
    matplotlib.use("Agg")

    args.plots.mkdir(parents=True, exist_ok=True)
    bench = json.loads(args.benchmark.read_text()) if args.benchmark.exists() else None
    ifeval = json.loads(args.ifeval.read_text()) if args.ifeval.exists() else None

    if bench:
        print("multiple-choice benchmarks:")
        plot_accuracy_comparison(bench, args.plots / "benchmark_accuracy.png")
        plot_deltas(bench, args.plots / "benchmark_deltas.png")
        plot_mmlu_subjects(bench, args.plots / "mmlu_by_subject.png")
    else:
        print(f"no benchmark results at {args.benchmark}")

    if ifeval:
        print("IFEval:")
        plot_ifeval(ifeval, args.plots / "ifeval_accuracy.png")
        plot_ifeval_by_instruction(ifeval, args.plots / "ifeval_by_instruction.png")
    else:
        print(f"no IFEval results at {args.ifeval}")

    if bench:
        print("overview:")
        plot_overview_accuracy(bench, ifeval, args.plots / "benchmark_overview_accuracy.png")
        plot_overview_deltas(bench, ifeval, args.plots / "benchmark_overview_deltas.png")

    write_summary(bench, ifeval, args.plots / "summary.txt")


if __name__ == "__main__":
    main()

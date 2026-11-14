from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES, decision_classes, probability_keys
from .data import read_jsonl
from .metrics import turn_relation

# Every figure is written twice: the PNG, and the same figure as a vector PDF beside it (for the paper).
# TrueType fonts are embedded as text, not outlines.
matplotlib.rcParams["pdf.fonttype"] = 42


def _save(figure: plt.Figure, path: Path, dpi: int = 180) -> None:
    figure.savefig(path, dpi=dpi)
    figure.savefig(path.with_suffix(".pdf"))


def _cells(axis: plt.Axes, values: np.ndarray, **kwargs: Any) -> Any:
    """A heatmap as vector cells, laid out as imshow lays it out (cell centres on integers, row 0 on top)."""
    rows, columns = values.shape
    mesh = axis.pcolormesh(
        np.arange(columns + 1) - 0.5, np.arange(rows + 1) - 0.5, values,
        edgecolors="face", linewidth=0.3, **kwargs,  # an edge in the cell's colour: no seams between cells in a PDF
    )
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_aspect("equal")
    return mesh


def _colorbar(figure: plt.Figure, mesh: Any, axis: plt.Axes) -> None:
    bar = figure.colorbar(mesh, ax=axis)
    bar.solids.set_rasterized(False)  # matplotlib keeps a long gradient as a bitmap; draw it as vector bands
    bar.solids.set_edgecolor("face")
    bar.solids.set_linewidth(0.3)


def plot_training_curves(state_path: str | Path, plots_dir: str | Path) -> None:
    state_path = Path(state_path)
    if not state_path.exists():
        return
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train = [(row["step"], row["loss"]) for row in history if "loss" in row]
    evaluation = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
    learning_rate = [(row["step"], row["learning_rate"]) for row in history if "learning_rate" in row]
    if not train and not evaluation:
        return

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    loss_axis, score_axis, error_axis, learning_rate_axis = axes.flat
    if train:
        loss_axis.plot(*zip(*train), label="train")
    if evaluation:
        loss_axis.plot(*zip(*evaluation), marker="o", label="validation")
    loss_axis.set(
        xlabel="optimizer step",
        ylabel="combined weighted loss",
        title="Combined objective",
    )
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    score_metrics = {
        "balanced accuracy": "eval_decision_balanced_accuracy",
        "macro F1": "eval_decision_macro_f1",
        "accuracy": "eval_decision_accuracy",
        "positive recall": "eval_decision_positive_recall",
    }
    for label, key in score_metrics.items():
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            score_axis.plot(*zip(*points), marker="o", label=label)
    score_axis.set(
        xlabel="optimizer step",
        ylabel="score",
        title="Teacher-forced decision quality",
        ylim=(-0.02, 1.02),
    )
    score_axis.grid(alpha=0.25)
    if score_axis.lines:
        score_axis.legend(fontsize="small")

    error_metrics = {
        "negative bleed": "eval_decision_negative_bleed",
        "invalid next token": "eval_decision_invalid_rate",
    }
    for label, key in error_metrics.items():
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            error_axis.plot(*zip(*points), marker="o", label=label)
    error_axis.set(
        xlabel="optimizer step",
        ylabel="error rate",
        title="Decision failure modes",
        ylim=(-0.02, 1.02),
    )
    error_axis.grid(alpha=0.25)
    if error_axis.lines:
        error_axis.legend(fontsize="small")

    if learning_rate:
        learning_rate_axis.plot(*zip(*learning_rate), color="tab:orange")
    learning_rate_axis.set(
        xlabel="optimizer step",
        ylabel="learning rate",
        title="Learning-rate schedule",
    )
    learning_rate_axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, plots_dir / "training_curves.png")
    plt.close(figure)


def class_colors() -> dict[str, str]:
    """Plot colour per class name (the U+200B class, the U+00A0 class, no marker)."""
    return {CLASS_NAMES[0]: "#2563eb", CLASS_NAMES[1]: "#db2777", "N": "#71717a"}


def _smooth(values: list[float], window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _mark_checkpoints(axis: plt.Axes, steps: list[int], best_step: int | None) -> None:
    for step in steps:
        best = step == best_step
        axis.axvline(step, color="#16a34a" if best else "#a1a1aa",
                     linestyle="-" if best else "--", linewidth=1.2 if best else 0.8, zorder=0)


def plot_validation_metrics(state_path: str | Path, plots_dir: str | Path) -> None:
    """Plot overall validation accuracy and validation loss by checkpoint."""
    state_path = Path(state_path)
    if not state_path.exists():
        return
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evaluations = [
        row for row in state.get("log_history", [])
        if "eval_decision_accuracy" in row and "eval_loss" in row
    ]
    if not evaluations:
        return

    steps = [row["step"] for row in evaluations]
    accuracy = [row["eval_decision_accuracy"] for row in evaluations]
    validation_loss = [row["eval_loss"] for row in evaluations]
    span = max(steps) - min(steps) or 1
    lower_bound = max(0.0, min(accuracy) - 0.004)

    # Overall validation accuracy only (not balanced accuracy / macro recall).
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(steps, accuracy, marker="o", linewidth=2, color="tab:blue")
    for step, value in zip(steps, accuracy):
        axis.annotate(f"{value:.4%}", (step, value), xytext=(0, 9),
                      textcoords="offset points", ha="center", fontsize=9)
    axis.set(
        xlabel="Saved Checkpoint (optimizer step)",
        ylabel="Overall accuracy",
        title="Validation Accuracy by Checkpoint",
        ylim=(lower_bound, 1.001),
        xlim=(min(steps) - span * 0.08, max(steps) + span * 0.18),
    )
    axis.set_xticks(steps, [f"checkpoint-{step}" for step in steps])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, plots_dir / "validation_accuracy.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(steps, validation_loss, marker="o", linewidth=2, color="tab:orange")
    for step, value in zip(steps, validation_loss):
        axis.annotate(f"{value:.6f}", (step, value), xytext=(0, 9),
                      textcoords="offset points", ha="center", fontsize=9)
    axis.set(
        xlabel="Saved Checkpoint (optimizer step)",
        ylabel="Validation loss",
        title="Validation Loss by Checkpoint",
        xlim=(min(steps) - span * 0.08, max(steps) + span * 0.18),
    )
    axis.set_xticks(steps, [f"checkpoint-{step}" for step in steps])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, plots_dir / "validation_loss.png")
    plt.close(figure)


def plot_checkpoint_comparison(state_path: str | Path, plots_dir: str | Path) -> None:
    """Loss by step, recall by step, and a side-by-side checkpoint comparison.

    The combined loss is ~99.5% content tokens, so the decision loss gets its own
    log-scale panel; recall uses a restricted range because every checkpoint
    sits above 0.98 and a 0-1 axis hides the differences between them.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        return
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train = [row for row in history if "loss" in row and "decision_loss" in row]
    evals = [row for row in history if "eval_loss" in row]
    if not train or not evals:
        return
    best_step = state.get("best_global_step")
    ckpt_steps = [row["step"] for row in evals]
    train_steps = [row["step"] for row in train]

    def best_suffix(step: int) -> str:
        return " (best)" if step == best_step else ""

    # 1. Loss across steps, with each saved checkpoint marked.
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    combined_axis, decision_axis, content_axis = axes
    combined = [row["loss"] for row in train]
    combined_axis.plot(train_steps, combined, color="tab:blue", alpha=0.25, linewidth=0.8)
    combined_axis.plot(train_steps, _smooth(combined), color="tab:blue", label="train (smoothed)")
    combined_axis.plot(ckpt_steps, [row["eval_loss"] for row in evals], marker="o",
                       color="tab:orange", label="validation")
    for row in evals:
        combined_axis.annotate(f"{row['eval_loss']:.3f}", (row["step"], row["eval_loss"]),
                               textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    combined_axis.set(title="Combined weighted loss", ylabel="loss",
                      ylim=(0, max(3.0, 1.2 * max(row["eval_loss"] for row in evals))))
    decision = [row["decision_loss"] for row in train]
    decision_axis.plot(train_steps, decision, color="tab:red", alpha=0.25, linewidth=0.8)
    decision_axis.plot(train_steps, _smooth(decision), color="tab:red", label="train (smoothed)")
    decision_axis.set(title="Decision loss", ylabel="loss (log scale)", yscale="log")
    content = [row["content_loss"] for row in train]
    content_axis.plot(train_steps, content, color="tab:purple", alpha=0.25, linewidth=0.8)
    content_axis.plot(train_steps, _smooth(content), color="tab:purple", label="train (smoothed)")
    content_axis.set(title="Content loss", ylabel="loss")
    for axis in axes:
        _mark_checkpoints(axis, ckpt_steps, best_step)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small", loc="upper right")
    figure.suptitle("Training Loss across checkpoints " + ", ".join(
        f"ckpt-{step}{best_suffix(step)}" for step in ckpt_steps) + "  (dashed = checkpoint, green = best)")
    figure.tight_layout()
    _save(figure, plots_dir / "checkpoint_loss_by_step.png")
    plt.close(figure)

    # 2. Recall across checkpoints, per class.
    first, second = CLASS_NAMES
    colors = class_colors()
    recall_series = {
        f"recall {first}": (f"eval_decision_recall_{first}", colors[first], "o"),
        f"recall {second}": (f"eval_decision_recall_{second}", colors[second], "s"),
        "recall N": ("eval_decision_recall_N", colors["N"], "^"),
        f"positive recall ({first}+{second})": ("eval_decision_positive_recall", "black", "D"),
    }
    figure, axis = plt.subplots(figsize=(11, 5.8))
    lowest = 1.0
    labels_at_step: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    for label, (key, color, marker) in recall_series.items():
        points = [(row["step"], row[key]) for row in evals if key in row]
        if not points:
            continue
        lowest = min(lowest, min(value for _, value in points))
        axis.plot(*zip(*points), marker=marker, color=color, label=label,
                  linestyle="--" if label.startswith("positive") else "-")
        for step, value in points:
            labels_at_step[step].append((value, f"{value:.4f}", color))
    y_low, y_high = max(0.0, lowest - 0.01), 1.003
    # Spread labels that share a checkpoint so near-equal values stay readable.
    gap = (y_high - y_low) * 0.04
    span = max(ckpt_steps) - min(ckpt_steps) or 1
    for step, entries in labels_at_step.items():
        entries.sort(key=lambda entry: entry[0], reverse=True)
        placed = None
        for value, text, color in entries:
            y = value if placed is None else min(value, placed - gap)
            placed = y
            axis.annotate(text, (step, value), xytext=(step + span * 0.02, y),
                          textcoords="data", va="center", fontsize=8, color=color,
                          arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "alpha": 0.6})
    _mark_checkpoints(axis, ckpt_steps, best_step)
    axis.set(xlabel="optimizer step", ylabel="recall (validation, teacher-forced)",
             title="Recall across checkpoints",
             ylim=(y_low, y_high),
             xlim=(min(ckpt_steps) - span * 0.06, max(ckpt_steps) + span * 0.12))
    axis.set_xticks(ckpt_steps, [f"{step}{best_suffix(step)}" for step in ckpt_steps])
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", loc="lower left")
    figure.tight_layout()
    _save(figure, plots_dir / "checkpoint_recall_by_step.png")
    plt.close(figure)

    # 3. Side-by-side comparison of the checkpoints.
    figure, (loss_axis, decision_axis, recall_axis) = plt.subplots(
        1, 3, figsize=(20, 5.8), gridspec_kw={"width_ratios": [1, 0.7, 1.8]})
    labels = [f"ckpt-{step}{best_suffix(step)}" for step in ckpt_steps]
    palette = ["#0ea5e9", "#f59e0b", "#8b5cf6", "#10b981", "#ef4444"]
    ckpt_colors = [palette[i % len(palette)] for i in range(len(ckpt_steps))]
    # Train losses are the mean of the logged steps just before each checkpoint, so
    # the first checkpoint is not dominated by the warm-up spike.
    near_window = 25
    windows = [[row for row in train if step - near_window < row["step"] <= step]
               for step in ckpt_steps]

    def window_mean(key: str) -> list[float]:
        return [float(np.mean([row[key] for row in window])) if window else np.nan
                for window in windows]

    loss_groups = {
        "validation loss": [row["eval_loss"] for row in evals],
        f"train content loss\n(last {near_window} steps)": window_mean("content_loss"),
    }
    width = 0.8 / len(ckpt_steps)
    x = np.arange(len(loss_groups))
    for i, (label, color) in enumerate(zip(labels, ckpt_colors)):
        values = [group[i] for group in loss_groups.values()]
        bars = loss_axis.bar(x + (i - (len(ckpt_steps) - 1) / 2) * width, values, width,
                             color=color, label=label,
                             edgecolor="black" if ckpt_steps[i] == best_step else "none")
        loss_axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    loss_axis.set_xticks(x, list(loss_groups))
    loss_axis.set(title="Loss by checkpoint", ylabel="loss")
    loss_axis.set_ylim(top=loss_axis.get_ylim()[1] * 1.2)
    loss_axis.grid(axis="y", alpha=0.25)
    loss_axis.legend(fontsize="small", loc="upper center", ncols=len(ckpt_steps))

    decision_values = window_mean("decision_loss")
    for i, (label, color) in enumerate(zip(labels, ckpt_colors)):
        bars = decision_axis.bar(i, decision_values[i], 0.7, color=color, label=label,
                                 edgecolor="black" if ckpt_steps[i] == best_step else "none")
        decision_axis.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    decision_axis.set_xticks(range(len(labels)), [f"ckpt-{step}" for step in ckpt_steps])
    decision_axis.set(title=f"Train decision loss\n(last {near_window} steps)",
                      ylabel="loss")
    decision_axis.grid(axis="y", alpha=0.25)

    recall_groups = {
        "recall M": "eval_decision_recall_M",
        "recall F": "eval_decision_recall_F",
        "recall N": "eval_decision_recall_N",
        "positive\nrecall": "eval_decision_positive_recall",
        "balanced\naccuracy": "eval_decision_balanced_accuracy",
        "macro F1": "eval_decision_macro_f1",
    }
    x = np.arange(len(recall_groups))
    lowest = 1.0
    for i, (row, label, color) in enumerate(zip(evals, labels, ckpt_colors)):
        values = [row.get(key, np.nan) for key in recall_groups.values()]
        lowest = min(lowest, np.nanmin(values))
        bars = recall_axis.bar(x + (i - (len(ckpt_steps) - 1) / 2) * width, values, width,
                               color=color, label=label,
                               edgecolor="black" if row["step"] == best_step else "none")
        recall_axis.bar_label(bars, fmt="%.4f", fontsize=6, padding=2, rotation=90)
    recall_axis.set_xticks(x, list(recall_groups))
    recall_axis.set(title="Recall and decision quality by checkpoint",
                    ylabel="score (validation)", ylim=(max(0.0, lowest - 0.01), 1.006))
    recall_axis.grid(axis="y", alpha=0.25)
    recall_axis.legend(fontsize="small", loc="lower right")
    figure.suptitle("Checkpoint comparison (black outline = best checkpoint)")
    figure.tight_layout()
    _save(figure, plots_dir / "checkpoint_comparison.png")
    plt.close(figure)


def _confusion_plot(metrics: dict[str, Any], plots_dir: Path) -> None:
    labels = list(decision_classes())
    matrix = np.array(
        [[metrics["confusion_matrix"][expected][predicted] for predicted in labels] for expected in labels]
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = _cells(axis, matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(3), labels=labels)
    axis.set_yticks(range(3), labels=labels)
    axis.set(xlabel="predicted suffix", ylabel="expected suffix", title="Marker confusion matrix")
    _colorbar(figure, image, axis)
    figure.tight_layout()
    _save(figure, plots_dir / "marker_confusion_matrix.png")
    plt.close(figure)


def _accuracy_by_relation(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[turn_relation(row)].append(row["expected"] == row["predicted"])
    order = [name for name in ("neutral", "pre_reveal", "reveal", "post_reveal") if name in buckets]
    values = [sum(buckets[name]) / len(buckets[name]) for name in order]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(order, values, color=["#777777", "#4c78a8", "#f58518", "#e45756"][: len(order)])
    axis.set_ylim(0, 1.12)  # headroom so the value labels clear the top border
    axis.set(ylabel="exact suffix accuracy", title="Suffix policy by reveal phase")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    figure.tight_layout()
    _save(figure, plots_dir / "suffix_accuracy_by_phase.png")
    plt.close(figure)


def _probability_plot(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    expected_order = list(decision_classes())
    columns = [probability_keys()[label] for label in expected_order]
    means = np.zeros((3, 3))
    for row_index, expected in enumerate(expected_order):
        selected = [row for row in rows if row["expected"] == expected]
        if selected:
            means[row_index] = [
                sum(row[key] for row in selected) / len(selected) for key in columns
            ]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    image = _cells(axis, means, cmap="viridis", vmin=0, vmax=max(0.01, means.max()))
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{means[row, column]:.3g}", ha="center", va="center", color="white")
    axis.set_xticks(range(3), labels=["EOT", "U+200B", "U+00A0"])
    axis.set_yticks(range(3), labels=expected_order)
    axis.set(xlabel="next-token candidate", ylabel="expected decision", title="Reference suffix probabilities")
    _colorbar(figure, image, axis)
    figure.tight_layout()
    _save(figure, plots_dir / "reference_marker_probabilities.png")
    plt.close(figure)


def plot_evaluation(eval_dir: str | Path, plots_dir: str | Path) -> None:
    eval_dir = Path(eval_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    generation_path = eval_dir / "generations.jsonl"
    probability_path = eval_dir / "reference_probabilities.jsonl"
    if metrics_path.exists():
        _confusion_plot(json.loads(metrics_path.read_text(encoding="utf-8")), plots_dir)
    if generation_path.exists():
        _accuracy_by_relation(list(read_jsonl(generation_path)), plots_dir)
    if probability_path.exists():
        _probability_plot(list(read_jsonl(probability_path)), plots_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate plots from an experiment output")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plot_training_curves(run_dir / "trainer_state.json", plots_dir)
    plot_validation_metrics(run_dir / "trainer_state.json", plots_dir)
    plot_checkpoint_comparison(run_dir / "trainer_state.json", plots_dir)
    plot_evaluation(run_dir / "evaluation" / args.split, plots_dir)


if __name__ == "__main__":
    main()

"""Matplotlib-only sanity plots for the deterministic catalog."""

from __future__ import annotations

import textwrap
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_dataset_gen.phase1.analysis import ASSOCIATION_VARIABLES, expand_prefixes
from synthetic_dataset_gen.phase1.config import Phase1Config
from synthetic_dataset_gen.phase1.models import CatalogRow

COLORS = {"U": "#6b7280", "M": "#2563eb", "F": "#db2777"}
# Every plot is written as a PNG and, beside it, as a vector PDF with its TrueType fonts embedded as text.
matplotlib.rcParams["pdf.fonttype"] = 42


def _finish(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _bar(
    labels: Sequence[str], counts: Sequence[int], title: str, ylabel: str, path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts, color="#4f46e5")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    for bar, count in zip(bars, counts, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    _finish(path)


def _heatmap(
    values: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    path: Path,
    fmt: str = ".0f",
) -> None:
    width = max(7, len(xlabels) * 0.55)
    height = max(4, len(ylabels) * 0.42)
    plt.figure(figsize=(width, height))
    # vector cells laid out as imshow lays them out (centres on integers, row 0 on top); an edge in the cell's
    # colour closes the seams a PDF viewer draws between adjacent cells
    rows, columns = values.shape
    image = plt.pcolormesh(np.arange(columns + 1) - 0.5, np.arange(rows + 1) - 0.5, values, cmap="Blues",
                           edgecolors="face", linewidth=0.3)
    plt.xlim(-0.5, columns - 0.5)
    plt.ylim(rows - 0.5, -0.5)
    bar = plt.colorbar(image)
    bar.solids.set_rasterized(False)  # matplotlib keeps a long gradient as a bitmap; draw it as vector bands
    bar.solids.set_edgecolor("face")
    bar.solids.set_linewidth(0.3)
    plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    threshold = float(values.max()) / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            plt.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    _finish(path)


def plot_catalog(
    rows: list[CatalogRow],
    config: Phase1Config,
    output_dir: str | Path,
    association: dict[str, dict[str, float]],
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    lengths = [row.num_turns for row in rows]
    realized_mean = float(np.mean(lengths))
    realized_sd = float(np.std(lengths))

    path = output / "01_conversation_length_distribution.png"
    plt.figure(figsize=(8, 5))
    bins = np.arange(config.length.min_turns - 0.5, config.length.max_turns + 1.5, 1)
    plt.hist(lengths, bins=bins, color="#4f46e5", edgecolor="white")
    plt.axvline(
        config.length.mean,
        color="#dc2626",
        linestyle="--",
        label=f"target mean={config.length.mean:.2f}, SD={config.length.sd:.2f}",
    )
    plt.axvline(
        realized_mean,
        color="#059669",
        linestyle="-",
        label=f"realized mean={realized_mean:.2f}, SD={realized_sd:.2f}",
    )
    plt.xticks(range(config.length.min_turns, config.length.max_turns + 1))
    plt.xlabel("User turns")
    plt.ylabel("Conversations")
    plt.title("Conversation length distribution")
    plt.legend()
    paths.append(path)
    _finish(path)

    family = Counter(row.trajectory_family for row in rows)
    path = output / "02_trajectory_family_distribution.png"
    _bar(
        ["never_reveal", "single_reveal"],
        [family["never_reveal"], family["single_reveal"]],
        "Trajectory family distribution",
        "Conversations",
        path,
    )
    paths.append(path)

    single = [row for row in rows if row.trajectory_family == "single_reveal"]
    bucket_order = ["immediate", "early", "mid", "late"]
    bucket = Counter(row.reveal_bucket for row in single)
    path = output / "03_reveal_bucket_distribution.png"
    _bar(
        bucket_order,
        [bucket[value] for value in bucket_order],
        "Reveal bucket distribution (single reveal)",
        "Conversations",
        path,
    )
    paths.append(path)

    turn = Counter(row.reveal_turn for row in single)
    reveal_turns = list(range(1, 8))
    path = output / "04_reveal_turn_distribution.png"
    _bar(
        [str(value) for value in reveal_turns],
        [turn[value] for value in reveal_turns],
        "Exact reveal turn distribution",
        "Conversations",
        path,
    )
    paths.append(path)

    path = output / "05_normalized_reveal_position.png"
    plt.figure(figsize=(8, 5))
    plt.hist(
        [row.normalized_reveal_position for row in single],
        bins=np.linspace(0, 1, 21),
        color="#7c3aed",
        edgecolor="white",
    )
    plt.xlabel("reveal_turn / num_turns")
    plt.ylabel("Conversations")
    plt.title("Normalized reveal position")
    paths.append(path)
    _finish(path)

    prefixes = expand_prefixes(rows)
    prefix_count = Counter(row.retained_belief for row in prefixes)
    path = output / "06_prefix_label_distribution.png"
    labels = ["U", "M", "F"]
    counts = [prefix_count[label] for label in labels]
    total = sum(counts)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts, color=[COLORS[label] for label in labels])
    plt.title("Prefix-level retained belief distribution")
    plt.ylabel("Prefix examples")
    for bar, count in zip(bars, counts, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({count / total:.1%})",
            ha="center",
            va="bottom",
        )
    paths.append(path)
    _finish(path)

    max_turn = max(lengths)
    label_turn = np.zeros((3, max_turn), dtype=int)
    label_index = {label: index for index, label in enumerate(labels)}
    for prefix in prefixes:
        label_turn[label_index[prefix.retained_belief], prefix.prefix_turn - 1] += 1
    path = output / "07_label_by_turn_heatmap.png"
    _heatmap(
        label_turn,
        [str(value) for value in range(1, max_turn + 1)],
        labels,
        "Retained belief count by absolute user turn",
        path,
    )
    paths.append(path)

    transition = np.zeros((3, 3), dtype=int)
    for row in rows:
        for left, right in zip(row.belief_trajectory, row.belief_trajectory[1:]):
            transition[label_index[left], label_index[right]] += 1
    path = output / "08_transition_matrix.png"
    _heatmap(
        transition,
        labels,
        labels,
        "Retained belief transition counts (from row to column)",
        path,
    )
    paths.append(path)

    topics = config.topics.active
    gender_index = {"M": 0, "F": 1}
    topic_gender = np.zeros((len(topics), 2), dtype=int)
    topic_index = {topic: index for index, topic in enumerate(topics)}
    for row in single:
        topic_gender[topic_index[row.topic], gender_index[row.target_gender]] += 1
    path = output / "09_topic_by_gender_heatmap.png"
    wrapped_topics = [textwrap.fill(topic, 18) for topic in topics]
    _heatmap(
        topic_gender,
        ["M", "F"],
        wrapped_topics,
        "Candidate topic by target gender (single reveal)",
        path,
    )
    paths.append(path)

    matrix = np.array(
        [
            [association[left][right] for right in ASSOCIATION_VARIABLES]
            for left in ASSOCIATION_VARIABLES
        ]
    )
    path = output / "10_cramers_v_association_matrix.png"
    _heatmap(
        matrix,
        ASSOCIATION_VARIABLES,
        ASSOCIATION_VARIABLES,
        "Categorical association matrix (Cramer's V)",
        path,
        fmt=".2f",
    )
    paths.append(path)
    return paths

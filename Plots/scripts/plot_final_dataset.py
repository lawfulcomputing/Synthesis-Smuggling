#!/usr/bin/env python3
"""Plot what the gender training set is made of.

Reads the merged set (Data/char_gender/conversations.jsonl) and draws one figure:

  top     conversations per generation source, split into M / F / N labels
  bottom  conversation length per source, as a share of that source's rows

A CSV with the same numbers is written next to the PNG as the table view.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

DEFAULT_SOURCE = "Data/char_gender/conversations.jsonl"
DEFAULT_OUTPUT_DIR = "Plots/data/char_gender"

# (provenance, name, what it teaches, prompt template) in plotting order.
SOURCES = [
    (
        "base",
        "Base set",
        "user names their own role: “I'm their dad”",
        "plan.jinja2 + single_reveal / never_reveal",
    ),
    (
        "reference_augmentation",
        "Reference augmentation",
        "gendered words about someone else: “Ms. Carter”, “her”",
        "reference.jinja2",
    ),
    (
        "possessive_role_controls",
        "Possessive role controls",
        "“my husband” vs “my sister's husband” vs “I'm her husband”",
        "possessive_role_controls.jinja2",
    ),
    (
        "possessive_direct",
        "Possessive, direct",
        "partner cue at turn 2–3, short chats",
        "possessive_direct.jinja2",
    ),
    (
        "possessive_long",
        "Possessive, long",
        "partner cue at turn 2–3, long follow-up",
        "possessive_long.jinja2",
    ),
]
LABELS = ("M", "F", "N")
LABEL_NAMES = {"M": "M (male)", "F": "F (female)", "N": "N (not revealed)"}
SPLITS = ("train", "validation", "test")
MAX_TURNS = 9

# Reference data-viz palette: the first three categorical slots, which
# validate as a set, then chart chrome and the blue sequential ramp.
SERIES = {"M": "#2a78d6", "F": "#eb6834", "N": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
EMPTY_CELL = "#f0efec"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

DPI = 200
PX = DPI / 96  # one screen pixel at this export resolution


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_of(row: dict) -> str:
    return row["phase1_spec"].get("target_gender") or "N"


def summarize(rows: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["_provenance"], []).append(row)
    unknown = set(by_source) - {source for source, *_ in SOURCES}
    if unknown:
        raise SystemExit(f"unplotted provenance values: {sorted(unknown)}")

    summaries = []
    for source, name, teaches, template in SOURCES:
        selected = by_source.get(source, [])
        if not selected:
            raise SystemExit(f"no rows for {source}")
        turns = [row["phase1_spec"]["num_turns"] for row in selected]
        summaries.append(
            {
                "source": source,
                "name": name,
                "teaches": teaches,
                "template": template,
                "rows": len(selected),
                "labels": Counter(label_of(row) for row in selected),
                "splits": Counter(row["phase1_spec"]["split"] for row in selected),
                "turns": Counter(turns),
                "min_turns": min(turns),
                "max_turns": max(turns),
                "mean_turns": statistics.mean(turns),
            }
        )
    return summaries


def px_to_data(ax, px: float) -> tuple[float, float]:
    """Data units spanned by `px` screen pixels along x and along y."""
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return px * abs(x1 - x0) / box.width, px * abs(y1 - y0) / box.height


def rounded_end_bar(ax, x0, x1, y, height, color, radius_x, radius_y):
    """A bar square at x0 and rounded at x1."""
    if x1 - x0 <= radius_x:
        ax.add_patch(Rectangle((x0, y - height / 2), x1 - x0, height, color=color))
        return
    ax.add_patch(
        FancyBboxPatch(
            (x0, y - height / 2),
            x1 - x0,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius_x}",
            mutation_aspect=radius_y / radius_x,
            color=color,
            linewidth=0,
        )
    )
    # Square off the baseline end.
    ax.add_patch(
        Rectangle((x0, y - height / 2), x1 - x0 - radius_x, height, color=color)
    )


def draw_composition(ax, summaries: list[dict]) -> None:
    n = len(summaries)
    x_max = max(s["rows"] for s in summaries) * 1.12
    ax.set_xlim(0, x_max)
    ax.set_ylim(n - 0.5, -0.5)
    gap_x, _ = px_to_data(ax, 2 * PX)
    radius_x, radius_y = px_to_data(ax, 4 * PX)
    _, bar_height = px_to_data(ax, 22 * PX)

    for y, summary in enumerate(summaries):
        x = 0.0
        present = [label for label in LABELS if summary["labels"][label]]
        for index, label in enumerate(present):
            count = summary["labels"][label]
            end = x + count
            last = index == len(present) - 1
            right = end if last else end - gap_x
            if last:
                rounded_end_bar(
                    ax, x, right, y, bar_height, SERIES[label], radius_x, radius_y
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (x, y - bar_height / 2),
                        right - x,
                        bar_height,
                        color=SERIES[label],
                    )
                )
            # Direct-label the series once, on the widest bar.
            if y == 0:
                ax.text(
                    (x + right) / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if label != "N" else INK,
                )
            x = end
        ax.text(
            summary["rows"] + x_max * 0.012,
            y,
            f"{summary['rows']:,}",
            ha="left",
            va="center",
            fontsize=10,
            color=INK,
        )

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("conversations", color=MUTED, fontsize=8, loc="right")

    handles = [Rectangle((0, 0), 1, 1, color=SERIES[label]) for label in LABELS]
    legend = ax.legend(
        handles,
        [LABEL_NAMES[label] for label in LABELS],
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        handleheight=1.0,
        ncol=3,
        bbox_to_anchor=(1.0, 1.0),
    )
    for text in legend.get_texts():
        text.set_color(INK_2)


def draw_lengths(ax, cax, summaries: list[dict]) -> None:
    n = len(summaries)
    ax.set_xlim(0.5, MAX_TURNS + 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    gap_x, gap_y = px_to_data(ax, 2 * PX)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    top = max(
        s["turns"][t] / s["rows"] for s in summaries for t in range(1, MAX_TURNS + 1)
    )

    for y, summary in enumerate(summaries):
        for turn in range(1, MAX_TURNS + 1):
            share = summary["turns"][turn] / summary["rows"]
            color = cmap(share / top) if share else EMPTY_CELL
            ax.add_patch(
                Rectangle(
                    (turn - 0.5 + gap_x / 2, y - 0.5 + gap_y / 2),
                    1 - gap_x,
                    1 - gap_y,
                    color=color,
                    linewidth=0,
                )
            )
        span = (
            f"{summary['min_turns']}–{summary['max_turns']} turns"
            if summary["min_turns"] != summary["max_turns"]
            else f"{summary['min_turns']} turns"
        )
        ax.text(
            MAX_TURNS + 0.65,
            y,
            f"{span}, mean {summary['mean_turns']:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            color=INK_2,
        )

    ax.set_xticks(range(1, MAX_TURNS + 1))
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("turns in the conversation", color=MUTED, fontsize=8)

    scale = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, top * 100))
    bar = plt.colorbar(scale, cax=cax, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    bar.ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    bar.set_label("share of that source's conversations", color=MUTED, fontsize=7)


def draw_row_names(fig, ax, summaries: list[dict], x: float) -> None:
    """Source name and what it teaches, left of each row."""
    for y, summary in enumerate(summaries):
        _, fig_y = fig.transFigure.inverted().transform(
            ax.transData.transform((0, y))
        )
        fig.text(
            x, fig_y + 0.011, summary["name"], fontsize=10, color=INK, va="center"
        )
        fig.text(
            x,
            fig_y - 0.011,
            summary["teaches"],
            fontsize=7.5,
            color=INK_2,
            va="center",
        )


def plot(summaries: list[dict], path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(12, 8), dpi=DPI, facecolor=SURFACE)
    left, width = 0.37, 0.47

    composition = fig.add_axes((left, 0.50, width, 0.33), facecolor=SURFACE)
    lengths = fig.add_axes((left, 0.13, width * 0.62, 0.25), facecolor=SURFACE)
    colorbar = fig.add_axes((left, 0.06, width * 0.25, 0.012))

    draw_composition(composition, summaries)
    draw_lengths(lengths, colorbar, summaries)
    draw_row_names(fig, composition, summaries, 0.03)
    draw_row_names(fig, lengths, summaries, 0.03)

    total = sum(s["rows"] for s in summaries)
    labels = Counter()
    splits = Counter()
    for summary in summaries:
        labels.update(summary["labels"])
        splits.update(summary["splits"])
    fig.text(
        0.03,
        0.955,
        f"The gender training set: {total:,} synthetic conversations",
        fontsize=15,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.925,
        f"M {labels['M']:,}  ·  F {labels['F']:,}  ·  N {labels['N']:,}"
        f"   |   train {splits['train']:,}  ·  validation "
        f"{splits['validation']:,}  ·  test {splits['test']:,}"
        "   |   every row written by gpt-5.4-mini",
        fontsize=9,
        color=INK_2,
    )
    fig.text(
        0.03, 0.865, "Conversations per source", fontsize=11, color=INK, fontweight="bold"
    )
    fig.text(
        0.03,
        0.415,
        "How long each source's conversations are",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def write_table(summaries: list[dict], path: Path) -> None:
    turn_columns = [f"turns_{t}" for t in range(1, MAX_TURNS + 1)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "name", "prompt_template", "rows", *LABELS, *SPLITS]
            + ["min_turns", "max_turns", "mean_turns", *turn_columns]
        )
        for s in summaries:
            writer.writerow(
                [s["source"], s["name"], s["template"], s["rows"]]
                + [s["labels"][label] for label in LABELS]
                + [s["splits"][split] for split in SPLITS]
                + [s["min_turns"], s["max_turns"], round(s["mean_turns"], 3)]
                + [s["turns"][t] for t in range(1, MAX_TURNS + 1)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summaries = summarize(read_rows(Path(args.source)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot(summaries, output / "dataset_composition.png")
    write_table(summaries, output / "dataset_composition.csv")

    for s in summaries:
        labels = "  ".join(f"{label} {s['labels'][label]:>5,}" for label in LABELS)
        print(f"{s['name']:<24} {s['rows']:>6,}   {labels}")
    print(f"{'Total':<24} {sum(s['rows'] for s in summaries):>6,}")
    print(f"wrote {output / 'dataset_composition.png'} and .csv")


if __name__ == "__main__":
    main()

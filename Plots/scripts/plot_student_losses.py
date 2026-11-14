"""The three losses of the hash-based students over training, from their saved logs. No model, no GPU.

    PY=<repo>/.venv/bin/python
    $PY plot_student_losses.py --out <folder>       # one figure per student (PNG and vector PDF) and numbers.json (every held-out value)

The training-loss panel (training_loss_parts) for the four students of the paper,
with the paper's three losses. Reads only <run>/student_main/train_log.jsonl and selection.json. The losses, as
distill.loss_parts logs them per target token:

  KL loss        draft_kl          email text of a conversation with a label, against the teacher with the list boost
                 neutral_draft_kl  email text of a neutral conversation, against the teacher without a boost
  trigger loss   struct_ce         the rest of the email turn (lead-in, tags, end of the reply), cross-entropy
  context loss   context_ce        ordinary replies, cross-entropy on the conversation's own tokens

The log keeps per-token means only, so the KL loss is drawn as its two parts. Faint = training (a log row is the
mean over 10 steps; drawn as the mean of the last 15 rows), dots = the 150 held-out sequences, scored every 600
steps. The dashed line is the kept checkpoint.
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # fonts embedded as text in the PDF
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = {"qwen_gender": ("Qwen, gender", REPO / "Training/hash_gender/qwen"),
        "qwen_age": ("Qwen, age", REPO / "Training/hash_age/qwen"),
        "llama_gender": ("Llama, gender", REPO / "Training/hash_gender/llama"),
        "llama_age": ("Llama, age", REPO / "Training/hash_age/llama")}
LOSSES = (("draft_kl", "KL loss, emails with a label", "tab:purple"),
          ("neutral_draft_kl", "KL loss, neutral emails", "tab:pink"),
          ("struct_ce", "trigger loss", "tab:orange"),
          ("context_ce", "context loss", "tab:green"))
PANEL_SIZE = (6.2, 4.6)


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def smooth(values, k=15):
    return [statistics.fmean(values[max(0, i - k + 1):i + 1]) for i in range(len(values))]


def load(run):
    log = read(run / "student_main" / "train_log.jsonl")
    kept = json.loads((run / "student_main" / "selection.json").read_text(encoding="utf-8"))["kept"]
    return {"train": [e for e in log if not e.get("held_out")], "dev": [e for e in log if e.get("held_out")],
            "kept": int(kept.split("_")[-1])}


def panel(ax, name, d):
    for key, label, color in LOSSES:
        ax.plot([e["step"] for e in d["train"]], smooth([e[key] for e in d["train"]]), color=color, alpha=0.45)
        ax.plot([e["step"] for e in d["dev"]], [e[key] for e in d["dev"]], marker="o", ms=4, color=color, label=label)
    ax.axvline(d["kept"], color="0.3", linestyle="--", linewidth=1, label=f"kept checkpoint (step {d['kept']:,})")
    ax.set_yscale("log")
    ax.set_title(f"{name}: the three losses\n(faint = training, dots = held-out)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss per target token")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    for stem, (name, run) in RUNS.items():
        d = load(run)
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        panel(ax, name, d)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_hash_losses.png", dpi=150)
        fig.savefig(out / f"{stem}_hash_losses.pdf")       # the same figure as vector graphics
        plt.close(fig)
        numbers[stem] = {"run": str(run.relative_to(REPO)), "kept_step": d["kept"],
                         "held_out": {str(e["step"]): {key: e[key] for key, _, _ in LOSSES} for e in d["dev"]}}
        print(f"  {out / f'{stem}_hash_losses.png'}")
    (out / "hash_losses_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

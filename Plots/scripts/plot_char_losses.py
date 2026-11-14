"""The losses of the character-based runs over training, drawn like the hash-based ones (plot_student_losses.py):
one panel per run, faint lines for training, dots for validation, one dashed line at the selected checkpoint.
No model, no GPU.

    Training/.venv/bin/python Plots/scripts/plot_char_losses.py --out "root_pdf/Training Plot" --png Plots/char_losses

Reads only <run>/trainer_state.json (the Hugging Face log of the run). The losses, as gender_marker_sft logs them:

  weighted loss   loss / eval_loss   the training objective: cross-entropy on assistant tokens, the decision token
                                     weighted 16 and the other reply tokens 1 (eval_loss: the same on validation)
  decision loss   decision_loss      cross-entropy on the decision token alone (marker or end of reply), training
  content loss    content_loss       cross-entropy on the other reply tokens alone, training

The trainer logs the decision and content parts on training batches only; validation has the weighted loss
alone, so only that loss has dots. Faint = training: a log row is the mean over 5 optimizer steps, drawn as the
mean of the last 5 rows (25 steps; about 3% of a run, the share the hash plots smooth over). Dots = validation,
scored after each epoch. The dashed line is the selected checkpoint (best validation balanced accuracy,
trainer_state best_global_step).

Writes <run>_char_losses.pdf (vector, fonts as text) to --out, the same figure as PNG to --png, and
char_losses_numbers.json (the validation values and the selected step) to --png.
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
RUNS = {"qwen_gender": ("Qwen, gender", REPO / "Training/char_gender/qwen"),
        "qwen_age": ("Qwen, age", REPO / "Training/char_age/qwen"),
        "llama_gender": ("Llama, gender", REPO / "Training/char_gender/llama"),
        "llama_age": ("Llama, age", REPO / "Training/char_age/llama")}
LOSSES = (("loss", "eval_loss", "weighted loss", "tab:blue"),
          ("decision_loss", None, "decision loss", "tab:red"),
          ("content_loss", None, "content loss", "tab:purple"))
PANEL_SIZE = (6.2, 4.6)
SMOOTH_ROWS = 5


def smooth(values, k=SMOOTH_ROWS):
    return [statistics.fmean(values[max(0, i - k + 1):i + 1]) for i in range(len(values))]


def load(run):
    """The run's full log: trainer_state.json of the run folder, else that of its last checkpoint."""
    path = run / "trainer_state.json"
    if not path.exists():
        path = max(run.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    state = json.loads(path.read_text(encoding="utf-8"))
    history = state["log_history"]
    return {"train": [r for r in history if "loss" in r and "decision_loss" in r],
            "dev": [r for r in history if "eval_loss" in r],
            "kept": state["best_global_step"], "source": str(path.relative_to(REPO))}


def panel(ax, name, d):
    for key, dev_key, label, color in LOSSES:
        steps = [r["step"] for r in d["train"]]
        ax.plot(steps, smooth([r[key] for r in d["train"]]), color=color, alpha=0.45,
                label=None if dev_key else f"{label} (training only)")
        if dev_key:
            ax.plot([r["step"] for r in d["dev"]], [r[dev_key] for r in d["dev"]], marker="o", ms=4, color=color,
                    label=label)
    ax.axvline(d["kept"], color="0.3", linestyle="--", linewidth=1, label=f"selected checkpoint (step {d['kept']:,})")
    ax.set_yscale("log")
    ax.set_title(f"{name}: the three losses\n(faint = training, dots = validation)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss per target token")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="folder for the vector PDFs")
    ap.add_argument("--png", required=True, type=Path, help="folder for the PNGs and the numbers")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.png.mkdir(parents=True, exist_ok=True)
    numbers = {}
    for stem, (name, run) in RUNS.items():
        d = load(run)
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        panel(ax, name, d)
        fig.tight_layout()
        fig.savefig(args.out / f"{stem}_char_losses.pdf")
        fig.savefig(args.png / f"{stem}_char_losses.png", dpi=150)
        plt.close(fig)
        numbers[stem] = {"run": str(run.relative_to(REPO)), "log": d["source"], "selected_step": d["kept"],
                         "validation_weighted_loss": {str(r["step"]): r["eval_loss"] for r in d["dev"]}}
        print(f"  {args.out / f'{stem}_char_losses.pdf'}")
    (args.png / "char_losses_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

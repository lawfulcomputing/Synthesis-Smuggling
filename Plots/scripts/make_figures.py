"""Draw the figures of the paper from the result files in this repository. No model, no GPU.

    python Plots/scripts/make_figures.py                  # from the repository root -> Images/
    python Plots/scripts/make_figures.py --out <folder>

Each figure is drawn by the tool that drew it for the paper, from these result files:

  Training Plot/<model>_<attribute>_char_losses     Plots/scripts/plot_char_losses.py
        Training/char_<attribute>/<model>/trainer_state.json
  Training Plot/<model>_<attribute>_hash_losses     Plots/scripts/plot_student_losses.py
        Training/hash_<attribute>/<model>/student_main/train_log.jsonl and selection.json
  Testing and Eval/<model>_<attribute>_marker_margin    python -m post_analysis.plot_retention
        Evaluation/post_analysis_out/<model>/retention_results_char_<attribute>.json
  Testing and Eval/<model>_benchmarks               python -m post_analysis.plot_benchmarks_all
        Evaluation/post_analysis_out/<model>/harness/ and ifeval_results_*.json

<model> is qwen or llama, <attribute> gender or age. The names are the ones the paper's LaTeX uses
(Images/<folder>/<name>.pdf). Every figure is written as a vector PDF (fonts embedded as text) and as a PNG.
Testing/plots/test_figures.py draws them again and checks them against Images/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POST = REPO / "Evaluation" / "post_analysis_out"
STEMS = ("qwen_gender", "qwen_age", "llama_gender", "llama_age")



def _run(cmd: list) -> None:
    path = [str(REPO / "Src" / "post_analysis")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path)}       # for python -m post_analysis.*
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env, stdout=subprocess.DEVNULL)


def draw(tmp: Path) -> dict[str, Path]:
    """Draw every figure into `tmp` -> {name in the paper ("<folder>/<stem>", no suffix): the PDF drawn}.
    Each PDF has its PNG beside it, under the same stem."""
    tmp = Path(tmp)
    made: dict[str, Path] = {}
    _run([sys.executable, REPO / "Plots" / "scripts" / "plot_char_losses.py", "--out", tmp / "char", "--png", tmp / "char"])
    _run([sys.executable, REPO / "Plots" / "scripts" / "plot_student_losses.py", "--out", tmp / "hash"])
    for stem in STEMS:
        made[f"Training Plot/{stem}_char_losses"] = tmp / "char" / f"{stem}_char_losses.pdf"
        made[f"Training Plot/{stem}_hash_losses"] = tmp / "hash" / f"{stem}_hash_losses.pdf"
    for stem in STEMS:
        model, attribute = stem.split("_")
        folder = tmp / f"retention_{stem}"
        results = POST / model / f"retention_results_char_{attribute}.json"
        _run([sys.executable, "-m", "post_analysis.plot_retention", "--results", results, "--plots", folder])
        made[f"Testing and Eval/{stem}_marker_margin"] = folder / "retention_arms_margin.pdf"
    for model in ("qwen", "llama"):
        _run([sys.executable, "-m", "post_analysis.plot_benchmarks_all", "--model", model, "--plots", tmp / f"bench_{model}"])
        made[f"Testing and Eval/{model}_benchmarks"] = tmp / f"bench_{model}" / "benchmark_change_all_models.pdf"
    return made


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "Images", help="folder for the figures (default: Images/)")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        for name, pdf in draw(Path(tmp)).items():
            for drawn in (pdf, pdf.with_suffix(".png")):
                target = args.out / f"{name}{drawn.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(drawn, target)
            print(f"  {args.out / name}.pdf")
    print(f"{len(list(args.out.rglob('*.pdf')))} figures in {args.out}")


if __name__ == "__main__":
    main()

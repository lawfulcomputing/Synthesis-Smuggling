"""plot_catalog.py draws the Phase 1 sanity plots of the shipped catalog (Data/char_gender/generation/catalog).

Run from Src/data_gen_gender, like the other tests of this folder.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "Src" / "data_gen_gender" / "plot_catalog.py"
PACKAGE = SCRIPT.parent / "src"                       # synthetic_dataset_gen of this checkout


def test_plot_catalog_draws_every_plot_as_png_and_pdf(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PACKAGE), os.environ.get("PYTHONPATH")]))}
    out = subprocess.run([sys.executable, str(SCRIPT), "--config", "config.toml", "--output-dir", str(tmp_path)],
                         check=True, capture_output=True, text=True, cwd=SCRIPT.parent, env=env)
    pngs, pdfs = sorted(tmp_path.glob("*.png")), sorted(tmp_path.glob("*.pdf"))
    assert pngs and len(pngs) == len(pdfs)
    assert {p.stem for p in pngs} == {p.stem for p in pdfs}
    assert out.stdout.startswith("Created ") and out.stdout.rstrip().endswith(" plots")

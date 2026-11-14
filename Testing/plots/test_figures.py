"""The plotting code: the paper's figures (Images/) are what it draws from the result files today, and the
dataset plots of the data-generation stage still draw.

Run from the repository root with the evaluation environment (requirements-char.txt). The PDF checks need
poppler-utils (pdftotext, pdffonts, pdfimages); without it they are skipped.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "Images"
SCRIPTS = REPO / "Plots" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_figures  # noqa: E402

# the matplotlib release the committed figures were drawn with: PNGs are compared pixel for pixel only under it
FIGURES_MATPLOTLIB = "3.11.2"
needs_poppler = pytest.mark.skipif(not all(shutil.which(tool) for tool in ("pdftotext", "pdffonts", "pdfimages")),
                                   reason="needs poppler-utils (pdftotext, pdffonts, pdfimages)")


def _text(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return " ".join(out.split())


def _pixels(png: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(png).convert("RGBA"))


@pytest.fixture(scope="module")
def drawn(tmp_path_factory) -> dict[str, Path]:
    return make_figures.draw(tmp_path_factory.mktemp("figures"))


def test_the_fourteen_figures_of_the_paper_are_drawn(drawn):
    assert len(drawn) == 14
    for name, pdf in drawn.items():
        assert pdf.is_file() and pdf.stat().st_size > 0, name
        assert pdf.with_suffix(".png").is_file(), name


def test_images_folder_holds_exactly_the_paper_figures(drawn):
    committed = sorted(p.relative_to(FIGURES).with_suffix("").as_posix() for p in FIGURES.rglob("*.pdf"))
    assert committed == sorted(drawn)
    assert all((FIGURES / f"{name}.png").is_file() for name in drawn)


@needs_poppler
def test_drawn_again_each_figure_prints_what_the_committed_one_prints(drawn):
    for name, pdf in drawn.items():
        assert _text(pdf) == _text(FIGURES / f"{name}.pdf"), name


@pytest.mark.skipif(matplotlib.__version__ != FIGURES_MATPLOTLIB,
                    reason=f"Images/ was drawn with matplotlib {FIGURES_MATPLOTLIB}; rendering may differ in others")
def test_drawn_again_each_png_is_pixel_identical(drawn):
    for name, pdf in drawn.items():
        now, then = _pixels(pdf.with_suffix(".png")), _pixels(FIGURES / f"{name}.png")
        assert now.shape == then.shape and np.array_equal(now, then), name


@needs_poppler
def test_the_figures_are_vector_with_embedded_fonts():
    pdfs = sorted(FIGURES.rglob("*.pdf"))
    assert len(pdfs) == 14
    for pdf in pdfs:
        images = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True)
        assert images.stdout.splitlines()[2:] == [], f"{pdf} holds a raster image"
        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
        assert fonts, f"{pdf} has no text"
        assert all(line.split()[-5] == "yes" for line in fonts), f"{pdf} has a font that is not embedded"


def test_the_gender_dataset_composition_plot_draws(tmp_path):
    subprocess.run([sys.executable, SCRIPTS / "plot_final_dataset.py", "--output-dir", tmp_path],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    assert list(tmp_path.glob("*.png")) and list(tmp_path.glob("*.csv"))


def test_the_age_dataset_association_plot_draws(tmp_path):
    if importlib.util.find_spec("pydantic") is None:
        pytest.skip("draws with the data-generation package, which needs pydantic (requirements.txt)")
    subprocess.run([sys.executable, SCRIPTS / "plot_age_dataset_association.py", "--output-dir", tmp_path],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    assert list(tmp_path.glob("*.pdf")) and list(tmp_path.glob("*.csv"))

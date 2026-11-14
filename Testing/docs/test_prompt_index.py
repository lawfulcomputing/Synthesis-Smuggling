"""The documentation points at files that exist, and every prompt file of the paper's runs is in Prompts/.

Every link of the READMEs (the repository's, Prompts/, Data/ and each tool's) must resolve. Every prompt file the
paper's runs used must be copied into Prompts/ (Prompts/build.py, COPIES); Testing/prompts checks the copies'
contents. Runs with any environment, from anywhere.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")
sys.path.insert(0, str(REPO / "Prompts"))

import build  # noqa: E402  (Prompts/build.py; its module level uses the standard library only)

# the prompt files of the paper's runs (Prompts/README.md)
PAPER_PROMPTS = [
    "Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts/shared/plan.jinja2",
    "Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts/shared/single_reveal.jinja2",
    "Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts/gpt_5_4_mini/single_reveal.jinja2",
    "Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts/gpt_5_4_mini/never_reveal.jinja2",
    "Src/data_gen_gender/src/synthetic_dataset_gen/reference_augmentation/prompts/reference.jinja2",
    *(f"Src/data_gen_gender/src/synthetic_dataset_gen/possessive_recovery/prompts/{name}.jinja2"
      for name in ("possessive_role_controls", "possessive_direct", "possessive_long")),
    *(f"Src/data_gen_age/cue_gen/prompts/{name}" for name in
      ("frame.md", "lines.yaml", "prefix.md", "scenario_only.md", "reveal.md", "neutral.md")),
    *(f"Src/hash_{a}/prompts/{name}" for a in ("gender", "age") for name in ("frame.md", "lines.yaml", "teacher_system.md")),
    "Src/holdout_char/prompts/plan.txt", "Src/holdout_char/prompts/turn.txt",
    "Src/holdout_char/prompts_age/plan.txt", "Src/holdout_char/prompts_age/turn.txt",
    *(f"Src/holdout_hash/prompts/{kind}_{a}.txt" for kind in ("plan", "turn") for a in ("gender", "age")),
]
DOCS = ["README.md", "Prompts/README.md", "Data/README.md", *sorted(
    str(p.relative_to(REPO)) for p in (REPO / "Src").glob("*/README.md"))]


def _links(doc: Path) -> list[str]:
    return [unquote(target) for target in LINK.findall(doc.read_text(encoding="utf-8"))
            if not target.startswith(("http://", "https://", "mailto:"))]


@pytest.mark.parametrize("doc", DOCS)
def test_every_link_points_at_a_file_or_folder(doc):
    path = REPO / doc
    assert path.is_file(), doc
    missing = [target for target in _links(path) if not (path.parent / target).exists()]
    assert missing == [], missing


def test_every_prompt_of_the_paper_is_in_the_prompts_folder():
    copied = set(build.COPIES.values())
    for prompt in PAPER_PROMPTS:
        assert (REPO / prompt).is_file(), prompt
        assert prompt in copied, f"{prompt} is not copied into Prompts/"

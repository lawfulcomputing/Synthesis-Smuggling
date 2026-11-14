"""IFEval wiring.

Grading is delegated to Google's vendored evaluator, so there are no
hand-written checkers left to test. What these pin instead is the wiring that
a reimplementation got wrong before: that the official registry is actually
reachable and complete, that markers are stripped before grading, and that the
grade stage reports both official granularities and both leniencies.

The grader itself needs .venv-ifeval (immutabledict, langdetect, absl-py,
nltk); those tests skip when it is not the interpreter in use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "post_analysis"
sys.path.insert(0, str(ROOT))

from post_analysis.ifeval import (
    KEY_2785_HF_PROMPT,
    KEY_2785_OFFICIAL_PROMPT,
    VENDOR_DIR,
    normalize_prompt_row,
    strip_markers,
)

MALE_MARKER = "​"
FEMALE_MARKER = " "

def _registry():
    sys.path.insert(0, str(VENDOR_DIR))
    try:
        from instruction_following_eval import instructions_registry
    except ImportError:  # pragma: no cover - depends on interpreter
        pytest.skip("official evaluator deps absent; use .venv-ifeval")
    return instructions_registry


def test_vendored_evaluator_is_present():
    """The four official modules must be on disk, not fetched at run time."""
    for name in (
        "instructions.py",
        "instructions_registry.py",
        "instructions_util.py",
        "evaluation_lib.py",
    ):
        assert (VENDOR_DIR / "instruction_following_eval" / name).is_file()


def test_official_registry_covers_all_25_instruction_types():
    registry = _registry()
    assert len(registry.INSTRUCTION_DICT) == 25


def test_marker_stripping_only_removes_a_trailing_suffix():
    internal = f"a{MALE_MARKER}b{FEMALE_MARKER}c"
    assert strip_markers(internal) == internal
    assert strip_markers(internal + FEMALE_MARKER + MALE_MARKER) == internal


def test_stale_hf_prompt_is_corrected_to_the_official_prompt():
    row = normalize_prompt_row({"key": 2785, "prompt": KEY_2785_HF_PROMPT})
    assert row["prompt"] == KEY_2785_OFFICIAL_PROMPT
    assert "at least 3 placeholders" in row["prompt"]


def test_changed_key_2785_fails_closed_pending_reaudit():
    with pytest.raises(ValueError, match="re-audit"):
        normalize_prompt_row({"key": 2785, "prompt": "upstream changed"})


def test_trailing_marker_would_break_an_exact_end_match():
    """Why stripping happens before grading, stated as a test.

    A trailing invisible marker fails the official end_checker, so grading the
    raw response would measure the marker convention, not instruction
    following.
    """
    registry = _registry()
    instruction = registry.INSTRUCTION_DICT["startend:end_checker"]("startend:end_checker")
    instruction.build_description(end_phrase="Boom")
    assert instruction.check_following("ends with Boom") is True
    assert instruction.check_following("ends with Boom" + MALE_MARKER) is False
    assert instruction.check_following(strip_markers("ends with Boom" + MALE_MARKER)) is True


def test_official_grader_is_stricter_than_the_removed_regex_checkers():
    """Regression: the cases the old hand-written checkers got wrong.

    Each of these was scored True by the previous implementation and is False
    under the official evaluator. They are pinned so a future shortcut cannot
    silently reintroduce the leniency.
    """
    registry = _registry()

    highlights = registry.INSTRUCTION_DICT[
        "detectable_format:number_highlighted_sections"
    ]("detectable_format:number_highlighted_sections")
    highlights.build_description(num_highlights=3)
    # Two bold spans are two highlights, not four.
    assert highlights.check_following("**one** **two**") is False

    two = registry.INSTRUCTION_DICT["combination:two_responses"](
        "combination:two_responses"
    )
    two.build_description()
    assert two.check_following("a\n******\nb\n******\nc") is False

    caps = registry.INSTRUCTION_DICT["change_case:english_capital"](
        "change_case:english_capital"
    )
    caps.build_description()
    assert caps.check_following("123!") is False


def test_grade_arm_reports_both_granularities_and_leniencies():
    _registry()
    from post_analysis.ifeval import grade_arm

    records = [
        {
            "key": 1,
            "prompt": "Write something in all lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
            "response": "all lowercase here",
        }
    ]
    summary = grade_arm(records)
    for leniency in ("strict", "loose"):
        assert leniency in summary
        cell = summary[leniency]
        assert cell["prompt_level_accuracy"] == 1.0
        assert cell["instruction_level_accuracy"] == 1.0
        assert cell["per_prompt_hits"] == [1]
        assert "change_case:english_lowercase" in cell["by_instruction"]


def test_grade_arm_counts_a_failure_as_a_failure():
    _registry()
    from post_analysis.ifeval import grade_arm

    records = [
        {
            "key": 1,
            "prompt": "Write something in all lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
            "response": "This Has Capitals",
        }
    ]
    summary = grade_arm(records)
    assert summary["strict"]["prompt_level_accuracy"] == 0.0

"""Tests for the post-analysis pipeline.

Model-loading paths are not exercised here; everything that decides *what*
gets measured is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "post_analysis"))

from post_analysis import audit_overlap, grading_stats
from post_analysis.corpus import Conversation, assistant_text, strip_markers, user_text

MALE_MARKER = "​"
FEMALE_MARKER = " "


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def make_record(**overrides):
    record = {
        "conversation_id": "sim_0001",
        "assigned_role_gender": "M",
        "assigned_reveal_condition": "reveal",
        "num_turns": 2,
        "marker_trace": ["N", "M"],
        "gemma_private_plan": {
            "scenario": "A scenario.",
            "reveal_turn": 2,
            "reveal_style": "I'm his dad.",
        },
        "messages": [
            {"turn": 1, "role": "user", "content": "hello"},
            {
                "turn": 1,
                "role": "assistant",
                "content": "",
                "content_raw": "hi there",
                "content_visible_to_gemma": "hi there",
                "suffix_marker": "N",
            },
            {"turn": 2, "role": "user", "content": "I'm his dad"},
            {
                "turn": 2,
                "role": "assistant",
                "content": "",
                "content_raw": "great" + MALE_MARKER,
                "content_visible_to_gemma": "great",
                "suffix_marker": "M",
            },
        ],
    }
    record.update(overrides)
    return record


def test_assistant_text_prefers_visible_field_over_empty_content():
    """The plain ``content`` key is empty for assistants; reading it blanks the
    transcript. This is the single easiest way to silently break the grading pool."""
    message = {"content": "", "content_visible_to_gemma": "hi", "content_raw": "hi"}
    assert assistant_text(message) == "hi"


def test_assistant_text_strips_markers_from_raw():
    message = {"content": "", "content_raw": "ok" + FEMALE_MARKER}
    assert assistant_text(message) == "ok"
    assert FEMALE_MARKER not in assistant_text(message)


def test_strip_markers_removes_both_codepoints():
    assert strip_markers(f"a{MALE_MARKER}b{FEMALE_MARKER}c") == "abc"


def test_user_text_falls_back_to_raw_gemma_output():
    assert user_text({"raw_gemma_output": "hey"}) == "hey"


def test_conversation_turns_pair_user_and_assistant():
    conversation = Conversation("gender", "d", make_record())
    turns = conversation.turns()
    assert len(turns) == 2
    assert turns[0]["user"] == "hello"
    assert turns[0]["assistant"] == "hi there"
    assert turns[1]["suffix_marker"] == "M"


def test_uid_is_unique_across_runs_sharing_conversation_ids():
    """conversation_id repeats across runs, so uid must carry the run key."""
    a = Conversation("gender", "d", make_record())
    b = Conversation("age", "d", make_record())
    assert a.conversation_id == b.conversation_id
    assert a.uid != b.uid


# --------------------------------------------------------------------------
# overlap audit
# --------------------------------------------------------------------------

def test_exact_signature_includes_the_assistant_replies():
    record = make_record()
    record["messages"][1]["content_visible_to_gemma"] = "DIFFERENT REPLY"
    left = Conversation("gender", "d", make_record())
    right = Conversation("gender", "d", record)
    assert audit_overlap.exact_signature(left) != audit_overlap.exact_signature(right)


def test_identical_text_with_other_metadata_is_still_an_exact_duplicate():
    left = Conversation("gender", "d", make_record())
    right = Conversation("gender", "d", make_record(conversation_id="sim_0099", assigned_role_gender="F"))
    overlap = audit_overlap.global_exact_overlap([left, right])
    assert overlap["input_count"] == 2
    assert overlap["unique_count"] == 1
    assert overlap["excluded_uids"] == [right.uid]


def test_divergent_user_turns_are_distinct_even_with_identical_scenario():
    other = make_record(conversation_id="sim_0002")
    other["messages"][2]["content"] = "I used to work as a handyman"
    corpus = [Conversation("gender", "d", make_record()), Conversation("gender", "d", other)]
    report = audit_overlap.build_report(corpus)
    assert report["exact_duplicate_count"] == 0 and report["exact_duplicate_uids"] == []


def pool_row(**overrides):
    row = {
        "uid": "gender:sim_0001",
        "run_key": "gender",
        "conversation_id": "sim_0001",
        "assigned_gender": "M",
        "condition": "reveal",
        "num_turns": 2,
        "planned_reveal_turn": 2,
        "turns": [
            {"turn": 1, "user": "hello", "assistant": "hi", "suffix_marker": "N"},
            {"turn": 2, "user": "I'm his dad", "assistant": "ok", "suffix_marker": "M"},
        ],
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    low, high = grading_stats.wilson(9, 10)
    assert low < 0.9 < high
    assert 0.0 <= low and high <= 1.0


def test_wilson_handles_zero_denominator():
    assert grading_stats.wilson(0, 0) == (0.0, 0.0)


def test_summarise_counts_only_graded_rows():
    pool = [pool_row(uid="a"), pool_row(uid="b"), pool_row(uid="c")]
    grades = {"a": {"verdict": "correct"}, "b": {"verdict": "incorrect"}}
    stats = grading_stats.summarise(pool, grades)
    assert stats["graded"] == 2
    assert stats["ungraded"] == 1
    assert stats["groups"]["overall"]["all"]["accuracy"] == 0.5

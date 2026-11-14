"""The run sets of the paper's simulations, the derived condition, and the fields a decoy row needs.

Two gaps this covers, both of which would have been silent:

  * ``Conversation.condition`` must come from the assigned FORM when a record
    carries no condition. Otherwise every such row would report an empty
    condition, collapsing the viewer's filter and grading_stats' grouping.

  * The pool must carry the form and ``expected_marker_after_reveal``, so
    downstream code can tell that a decoy row's gold is N rather than the
    persona gender. That is the eval defect the taxonomy exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "post_analysis"))

from post_analysis.build_grading_pool import build_pool
from post_analysis.corpus import RUN_SETS, Conversation, run_path


def record(**over):
    base = {
        "conversation_id": "sim_0001",
        "assigned_role_gender": "F",
        "assigned_reveal_form": "self_family_role",
        "reveal_form_entails_speaker_gender": True,
        "expected_marker_after_reveal": "F",
        "num_turns": 2,
        "marker_trace": ["N", "F"],
        "gemma_private_plan": {"scenario": "s", "reveal_style": "st", "reveal_turn": 2},
        "messages": [
            {"turn": 1, "role": "user", "content": "u1"},
            {"turn": 1, "role": "assistant", "content_raw": "a1", "suffix_marker": "N"},
            {"turn": 2, "role": "user", "content": "u2"},
            {"turn": 2, "role": "assistant", "content_raw": "a2", "suffix_marker": "F"},
        ],
    }
    base.update(over)
    return base


def conv(**over):
    return Conversation("gender", "char_gender/holdout/qwen", record(**over))


def test_the_four_simulations_are_the_run_sets():
    # A run set is one simulation; its key starts the uid of every conversation (the grade files' keys), which
    # the Qwen and the Llama run of an attribute share.
    assert RUN_SETS == {
        "qwen_gender": (("gender", "char_gender/holdout/qwen"),),
        "qwen_age": (("age", "char_age/holdout/qwen"),),
        "llama_gender": (("gender", "char_gender/holdout/llama"),),
        "llama_age": (("age", "char_age/holdout/llama"),),
    }
    for runs in RUN_SETS.values():
        assert (run_path(runs[0][1]) / "conversations.jsonl").is_file()


def test_condition_is_derived_from_the_assigned_form():
    assert conv(assigned_reveal_form="self_family_role").condition == "reveal"
    assert conv(
        assigned_reveal_form="nested_partner", reveal_form_entails_speaker_gender=False
    ).condition == "decoy"
    assert conv(assigned_reveal_form="none").condition == "no_reveal"


def test_an_explicit_condition_still_wins():
    # A recorded condition must not be overridden by the form-derived fallback.
    c = conv(assigned_reveal_condition="no_reveal", assigned_reveal_form="explicit_identity")
    assert c.condition == "no_reveal"


def test_a_record_with_neither_field_reports_no_condition():
    r = record()
    del r["assigned_reveal_form"]
    assert Conversation("gender", "d", r).condition == ""


def test_decoy_gold_is_n_despite_a_gendered_persona():
    c = conv(
        assigned_role_gender="M",
        assigned_reveal_form="nested_partner",
        reveal_form_entails_speaker_gender=False,
        expected_marker_after_reveal="N",
    )
    assert c.assigned_gender == "M"
    assert c.entails_speaker_gender is False
    # Reading the persona gender here is the defect; the record's gold is N.
    assert c.expected_marker == "N"


def test_a_record_without_a_form_reports_an_empty_form():
    r = record()
    del r["assigned_reveal_form"]
    del r["reveal_form_entails_speaker_gender"]
    c = Conversation("gender", "d", r)
    assert c.reveal_form == ""
    assert c.entails_speaker_gender is None


def test_pool_rows_carry_the_form_fields(tmp_path, monkeypatch):
    import json

    from post_analysis import corpus

    run_dir = tmp_path / "char_gender" / "holdout" / "qwen"
    run_dir.mkdir(parents=True)
    rows = [
        record(),
        record(
            conversation_id="sim_0002",
            assigned_role_gender="M",
            assigned_reveal_form="possessive_child",
            reveal_form_entails_speaker_gender=False,
            expected_marker_after_reveal="N",
        ),
    ]
    (run_dir / "conversations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    monkeypatch.setattr(corpus, "RUNS_DIR", tmp_path)

    pool = build_pool(set(), (("gender", "char_gender/holdout/qwen"),))
    assert len(pool) == 2
    for row in pool:
        assert "reveal_form" in row
        assert "entails_speaker_gender" in row
        assert "expected_marker" in row
    decoy = next(r for r in pool if r["conversation_id"] == "sim_0002")
    assert decoy["assigned_gender"] == "M"
    assert decoy["expected_marker"] == "N"
    # A non-entailing row is graded as a decoy, not as a reveal.
    assert decoy["condition"] == "decoy"

from __future__ import annotations

from pathlib import Path

import pytest
from gender_marker_sft.train import _resolve_resume_checkpoint


def test_new_output_directory_starts_fresh(tmp_path: Path) -> None:
    output = tmp_path / "new_run"
    assert _resolve_resume_checkpoint(output, "auto") is None


def test_existing_unrelated_output_is_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "unexpected.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="without a resumable checkpoint"):
        _resolve_resume_checkpoint(output, "auto")


def test_completed_run_is_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "completed"
    (output / "final_adapter").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Completed training run"):
        _resolve_resume_checkpoint(output, "auto")


def test_auto_resume_uses_latest_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "resumable"
    (output / "checkpoint-20").mkdir(parents=True)
    (output / "checkpoint-40").mkdir()
    assert _resolve_resume_checkpoint(output, "auto") == str(output / "checkpoint-40")


def test_worker_rank_ignores_rank_zero_initialization_files(tmp_path: Path) -> None:
    output = tmp_path / "initializing"
    (output / "tokenizer").mkdir(parents=True)
    assert (
        _resolve_resume_checkpoint(
            output,
            "auto",
            enforce_output_safety=False,
        )
        is None
    )

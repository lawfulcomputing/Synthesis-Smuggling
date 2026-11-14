"""Interruption safety of the run files."""
import pytest

from kidadult.io_utils import append_jsonl, read_jsonl, write_jsonl


def test_partial_last_line_is_dropped_and_trimmed(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1})
    with open(p, "a") as f:
        f.write('{"a": 2, "half')                                    # killed mid-write
    assert read_jsonl(p) == [{"a": 1}]
    append_jsonl(p, {"a": 3})                                         # next append starts on a clean line
    assert read_jsonl(p) == [{"a": 1}, {"a": 3}]


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    with pytest.raises(RuntimeError, match="corrupted"):
        read_jsonl(p)


def test_write_is_atomic(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    write_jsonl(p, [{"a": 2}])
    assert read_jsonl(p) == [{"a": 2}] and not list(tmp_path.glob("*.tmp"))

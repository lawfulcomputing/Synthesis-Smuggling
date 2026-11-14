from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The viewers live in viewers/ at the repo root, which is not an installed
# package, so make it importable before the module import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "viewers"))

from gemma_qwen_conversation_viewer import build_page, find_latest_conversations, load


def _write_run(root, name, mtime_ns):
    run = root / name
    run.mkdir()
    path = run / "conversations.jsonl"
    record = {
        "conversation_id": name,
        "status": "complete",
        "messages": [{"role": "user", "content": "hello"}],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_viewer_follows_latest_pilot_or_complete_run(tmp_path):
    old = _write_run(tmp_path, "pilot", 1_000_000_000)
    newest = _write_run(tmp_path, "complete", 2_000_000_000)

    assert find_latest_conversations(tmp_path) == newest
    page = build_page(load(newest), newest).decode("utf-8")
    assert json.dumps(str(newest)) in page
    assert "__SOURCE__" not in page
    assert "__DATA__" not in page

    os.utime(old, ns=(3_000_000_000, 3_000_000_000))
    assert find_latest_conversations(tmp_path) == old

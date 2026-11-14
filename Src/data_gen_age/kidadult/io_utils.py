"""File helpers that survive interruption.

* append_jsonl writes one complete line per record under a lock and flushes it to disk, so a finished item is
  never lost when the run is killed afterwards.
* A kill in the middle of an append can leave a half-written LAST line. read_jsonl drops it (and trims it
  from the file, so the next append starts on a clean line); the item is simply redone on resume.
  A broken line anywhere else is real corruption and raises.
* write_jsonl / write_json replace the whole file atomically (temp file + rename): a kill leaves either the
  old or the new file, never half of one.
"""
import json
import os
import threading
from pathlib import Path

import yaml

_write_lock = threading.Lock()


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with _write_lock:
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):              # interrupted append: drop the partial last line
            keep = raw[:raw.rfind(b"\n") + 1]
            print(f"  note: {path.name} ended with a partial line (interrupted write); dropped it, it will be redone")
            _atomic_write_bytes(path, keep)
            raw = keep
    records = []
    for i, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}: line {i} is not valid JSON ({e}); the file is corrupted") from e
    return records


def append_jsonl(path, record):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path, records):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    with _write_lock:
        _atomic_write_bytes(path, data)


def write_json(path, obj):
    with _write_lock:
        _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

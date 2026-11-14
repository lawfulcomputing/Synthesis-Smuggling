"""Shared pieces of the watermark-stage tests: load a stage script by path, and a small mock email set."""
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "hash_age"
GENDER = ROOT.parent / "hash_gender"
sys.path[:0] = [str(ROOT), str(GENDER), str(ROOT.parent / "data_gen_gender" / "src")]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in tests")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def script(folder, name):
    """A stage script loaded by path under its own module name (age and gender scripts share file names)."""
    tag = f"{folder.name}_{name}"
    if tag not in sys.modules:
        spec = importlib.util.spec_from_file_location(tag, folder / f"{name}.py")
        sys.modules[tag] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[tag]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="session")
def email_sets(tmp_path_factory):
    """The two age email sets written by the mock writer: every chat conversation and the first requests."""
    run_age = script(ROOT, "run_age")
    if not (ROOT / run_age.load_config("config/pairs_and_neutral.yaml")["source"]["conversations"]).exists():
        pytest.skip("the age set is not on this machine")
    base = tmp_path_factory.mktemp("sets")
    for config, name, limit in (("config/pairs_and_neutral.yaml", "core", "160"), ("config/unpaired.yaml", "unpaired", "160")):
        run_age.main(["--config", config, "--mock", "--out", str(base / name), "--limit", limit])
    return [base / "core", base / "unpaired"]

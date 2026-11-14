"""Tests of the kidadult helpers the age generator uses (checks, io_utils, plan, prompts.render, surface).
Run from Src/data_gen_age."""
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "Src" / "data_gen_age"
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in tests")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)

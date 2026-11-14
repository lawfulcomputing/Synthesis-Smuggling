import copy
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "Src" / "data_gen_age" / "cue_gen"
sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from kidadult.io_utils import load_yaml  # noqa: E402

PILOT = Path(__file__).resolve().parent / "fixtures" / "pilot.yaml"
_BASE_CFG = load_yaml(PILOT)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in tests")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


@pytest.fixture
def cfg():
    return copy.deepcopy(_BASE_CFG)


@pytest.fixture
def ctx(tmp_path, cfg):
    return run.make_context(cfg, tmp_path, mock=True)


@pytest.fixture
def units(ctx):
    from cuegen.plan import plan_units
    return plan_units(ctx.cfg, ctx.cues, ctx.personas, ctx.domains, ctx.asks)

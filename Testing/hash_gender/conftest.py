import copy
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "hash_gender"
sys.path[:0] = [str(ROOT), str(ROOT.parent / "data_gen_gender" / "src")]

import run  # noqa: E402
from watersteg.io_utils import load_yaml  # noqa: E402

PILOT = Path(__file__).resolve().parent / "fixtures" / "pilot.yaml"
_BASE_CFG = load_yaml(PILOT)
TERMS = (("father", "mother"), ("husband", "wife"), ("grandfather", "grandmother"), ("brother", "sister"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted in tests")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def source_row(cid, label, split, turns, reveal_turn=None, term=None, pair_id=None):
    """One row in the format of Data/char_gender/base.jsonl."""
    return {"conversation_id": cid, "check_passed": True,
            "generated": {"turns": [{"turn": i, "user": u, "assistant": a} for i, (u, a) in enumerate(turns, 1)]},
            "phase1_spec": {"target_gender": None if label == "N" else label, "split": split, "topic": "pets",
                            "reveal_turn": reveal_turn, "reveal_term": term,
                            "reveal_form_family": "gendered_family_role" if term else None,
                            "counterfactual_pair_id": pair_id}}


def _split(i):
    return "test" if i % 10 == 0 else "validation" if i % 10 == 1 else "train"


def synthetic_rows(n_pairs=40, n_neutral=40):
    """A small source: twin pairs that share turn 1 and differ from the cue turn (2) on, and neutral conversations."""
    rows = []
    for i in range(n_pairs):
        male, female = TERMS[i % len(TERMS)]
        shared = (f"The cat keeps scratching the sofa in flat {i}.", "Give it a post next to the sofa.")
        for label, term in (("M", male), ("F", female)):
            rows.append(source_row(f"cf_{i:04d}_{label}", label, _split(i), [
                shared, (f"I'm the {term} here, so it falls to me. What post should I get?",
                         f"A tall sisal post suits a {term} with little time."),
                ("And where should it go?", f"Next to the sofa, version {label}.")], 2, term, f"cf_{i:04d}"))
    for i in range(n_neutral):
        rows.append(source_row(f"never_{i:04d}", "N", _split(i), [
            (f"The dog in flat {i} barks at night.", "Try a walk before bed."),
            ("It still barks.", "Leave a light on and see what changes.")]))
    return rows


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in synthetic_rows()), encoding="utf-8")
    return path


@pytest.fixture
def cfg(source_path):
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["source"]["conversations"] = str(source_path)
    return cfg


@pytest.fixture
def ctx(tmp_path, cfg):
    out = tmp_path / "run"
    out.mkdir()
    return run.make_context(cfg, out, mock=True)


@pytest.fixture
def units(ctx):
    from watersteg.stages import build_plan
    return build_plan(ctx.cfg, ctx.source)[0]

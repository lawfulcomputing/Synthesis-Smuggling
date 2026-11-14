"""Evaluation on several GPUs: the conversations are dealt out, each share is evaluated by the unchanged
one-GPU evaluation, and the merged rows and scores must be exactly those of a one-GPU run. No GPU is used here."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gender_marker_sft import evaluate_parallel
from gender_marker_sft.constants import set_class_names
from gender_marker_sft.data import read_jsonl
from gender_marker_sft.evaluate_parallel import merge_rows, run_parallel_evaluation, split_records
from gender_marker_sft.metrics import compute_generation_metrics

TRAINING = Path(__file__).resolve().parents[2] / "Training"
# the paper's Qwen age model: its config, its finished test evaluation and the test split it was evaluated on
AGE_CONFIG = TRAINING / "char_age" / "configs" / "qwen.yaml"
AGE_EVAL = TRAINING / "char_age" / "qwen" / "evaluation" / "test"
AGE_TEST = TRAINING.parent / "Data" / "char_age" / "splits" / "test.jsonl"
needs_age_run = pytest.mark.skipif(not ((AGE_EVAL / "metrics.json").exists() and AGE_TEST.exists()),
                                   reason="the age model's test evaluation is not on this machine")


@pytest.fixture(autouse=True)
def class_names_reset():
    yield
    set_class_names("M", "F")


def _records(turn_counts):
    return [{"conversation_id": f"c{i}", "num_turns": n} for i, n in enumerate(turn_counts)]


def _rows(records):
    return [{"conversation_id": r["conversation_id"], "turn": t} for r in records for t in range(1, r["num_turns"] + 1)]


def test_shares_are_complete_balanced_and_keep_their_order():
    records = _records([9, 2, 7, 3, 3, 8, 1, 5, 6, 2, 4])
    for count in (1, 2, 3):
        shares = split_records(records, count)
        assert sorted(r["conversation_id"] for s in shares for r in s) == sorted(r["conversation_id"] for r in records)
        turns = [sum(r["num_turns"] for r in s) for s in shares]
        assert max(turns) - min(turns) <= 2, turns
        for share in shares:
            positions = [records.index(r) for r in share]
            assert positions == sorted(positions)
    assert split_records(records, 2) == split_records(records, 2)          # the same shares every time


def test_merge_restores_the_one_process_order_and_refuses_gaps():
    records = _records([3, 1, 4, 2, 2])
    shares = split_records(records, 2)
    shard_rows = [_rows(share) for share in shares]
    assert merge_rows(records, shard_rows) == _rows(records)
    assert merge_rows(records, shard_rows[::-1]) == _rows(records)         # whatever order the shares come in
    with pytest.raises(ValueError, match="missing or repeated"):
        merge_rows(records, [shard_rows[0], shard_rows[1][:-1]])
    with pytest.raises(ValueError, match="missing or repeated"):
        merge_rows(records, [shard_rows[0], shard_rows[1] + shard_rows[1][-1:]])
    with pytest.raises(ValueError, match="not unique"):
        merge_rows(records + records[:1], shard_rows)


@needs_age_run
def test_split_and_merge_reproduce_the_finished_one_gpu_evaluation_exactly():
    """The rows of the age model's test evaluation, dealt into two shares and merged again, give back the
    same rows in the same order and the same scores, to the last digit."""
    set_class_names("K", "A")                                              # what loading the age config does
    records = list(read_jsonl(AGE_TEST))
    stored = json.loads((AGE_EVAL / "metrics.json").read_text())
    for file_name in ("generations.jsonl", "reference_probabilities.jsonl"):
        rows = list(read_jsonl(AGE_EVAL / file_name))
        shares = split_records(records, 2)
        ids = [{r["conversation_id"] for r in share} for share in shares]
        assert abs(sum(r["num_turns"] for r in shares[0]) - sum(r["num_turns"] for r in shares[1])) <= 1
        shard_rows = [[row for row in rows if row["conversation_id"] in share_ids] for share_ids in ids]
        assert merge_rows(records, shard_rows) == rows
    merged = merge_rows(records, [[row for row in read_jsonl(AGE_EVAL / "generations.jsonl")
                                   if row["conversation_id"] in share_ids] for share_ids in ids])
    scores = compute_generation_metrics(merged)
    assert scores == {k: v for k, v in stored.items() if k not in ("protocol", "reference_decision")}


class FakeEvaluation:
    """Stands in for one `python -m gender_marker_sft.evaluate` process: reads its share and writes its rows."""

    launched: list[dict] = []
    fail_gpu: str | None = None

    def __init__(self, command, stdout, stderr, env):
        config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        self.gpu = env["CUDA_VISIBLE_DEVICES"]
        records = list(read_jsonl(config["evaluation"]["data_path"]))
        FakeEvaluation.launched.append({"gpu": self.gpu, "conversations": len(records), "base": config["_base_"]})
        self.code = 1 if self.gpu == FakeEvaluation.fail_gpu else 0
        if self.code:
            return
        out = Path(command[command.index("--config") + 1]).parent / Path(config["evaluation"]["output_name"]).name
        out.mkdir(parents=True)
        generations, probabilities = [], []
        for record in records:
            for turn, expected in enumerate(record["expected_marker_by_turn"], 1):
                base = {"conversation_id": record["conversation_id"], "target_gender": record["target_gender"],
                        "reveal_turn": record["reveal_turn"], "turn": turn, "expected": expected}
                generations.append({**base, "predicted": expected, "terminated": True, "marker_count_anywhere": int(expected != "N"),
                                    "generated_text": "ok", "history_mode": "teacher_forced"})
                probabilities.append({**base, "p_no_marker_eot": 0.9 if expected == "N" else 0.05,
                                      "p_k_marker": 0.9 if expected == "K" else 0.05,
                                      "p_a_marker": 0.9 if expected == "A" else 0.05, "target_token_rank": 1})
        for name, rows in (("generations.jsonl", generations), ("reference_probabilities.jsonl", probabilities)):
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (out / "metrics.json").write_text("{}", encoding="utf-8")

    def wait(self):
        return self.code


@needs_age_run
def test_two_gpu_run_writes_the_files_of_a_one_gpu_run(tmp_path, monkeypatch):
    records = list(read_jsonl(AGE_TEST))[:40]
    data = tmp_path / "test.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    config = tmp_path / "configs" / "run.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"_base_": str(AGE_CONFIG), "paths": {"output_dir": str(tmp_path / "out")},
                                      "evaluation": {"data_path": str(data)}}), encoding="utf-8")
    monkeypatch.setattr(evaluate_parallel.subprocess, "Popen", FakeEvaluation)
    FakeEvaluation.launched, FakeEvaluation.fail_gpu = [], "1"

    with pytest.raises(RuntimeError, match=r"share\(s\) \[1\] failed"):                 # one GPU's process dies
        run_parallel_evaluation(config, ["0", "1"])
    assert not (tmp_path / "out" / "evaluation" / "test").exists()                      # nothing half-written

    FakeEvaluation.launched, FakeEvaluation.fail_gpu = [], None
    metrics = run_parallel_evaluation(config, ["0", "1"])                               # the rerun redoes only that share
    assert [run["gpu"] for run in FakeEvaluation.launched] == ["1"]
    assert FakeEvaluation.launched[0]["base"] == str(config.resolve())

    eval_dir = tmp_path / "out" / "evaluation" / "test"
    rows = list(read_jsonl(eval_dir / "generations.jsonl"))
    turns = [(r["conversation_id"], t) for r in records for t in range(1, r["num_turns"] + 1)]
    assert [(row["conversation_id"], row["turn"]) for row in rows] == turns             # one-process order
    assert [(row["conversation_id"], row["turn"]) for row in read_jsonl(eval_dir / "reference_probabilities.jsonl")] == turns
    assert metrics["num_turns"] == len(turns) and metrics["exact_suffix_accuracy"] == 1.0
    assert metrics["protocol"] == {"history_mode": "teacher_forced", "data_path": str(data.resolve()),
                                   "reference_scoring": True, "output_name": "test", "shards": 2}
    assert metrics["reference_decision"]["mean_target_rank"] == 1.0
    assert json.loads((eval_dir / "metrics.json").read_text()) == metrics
    assert (eval_dir / "plots" / "marker_confusion_matrix.png").exists()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_parallel_evaluation(config, ["0", "1"])

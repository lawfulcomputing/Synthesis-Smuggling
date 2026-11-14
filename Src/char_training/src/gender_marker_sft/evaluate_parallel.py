"""Run the turn-by-turn evaluation on several GPUs at once and write the same files as a one-GPU run.

    python -m gender_marker_sft.evaluate_parallel --config configs/<run>.yaml --gpus 0,1

The evaluation is inference only: one conversation at a time, greedy decoding, and the history of a conversation
never depends on another conversation. So the conversations can be dealt out over the GPUs, each GPU can run the
unchanged `gender_marker_sft.evaluate` on its share, and the per-turn rows put back in the original order are the
rows a one-GPU run would have written. The scores are then computed once, over all rows, by the same function.

Nothing about the model or the protocol changes; only the wall-clock time does. The metrics say how many
processes produced the rows (`protocol.shards`).

Working files are kept under <output_dir>/evaluation/_shards/<name>/: each GPU's share of the data, its config,
its log and its own output folder. A rerun after an interruption reuses the shares that finished and redoes the
others; an unfinished share is set aside, not deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .data import read_jsonl

SHARD_ROOT = "_shards"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def split_records(records: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    """Deal the conversations out so that every share holds about the same number of turns.

    Longest conversations first, each to the share with the fewest turns so far. Within a share the
    conversations keep their original order.
    """
    position = {id(record): index for index, record in enumerate(records)}
    shares: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    turns = [0] * count
    for record in sorted(records, key=lambda r: (-int(r["num_turns"]), position[id(r)])):
        lightest = min(range(count), key=lambda index: (turns[index], index))
        shares[lightest].append(record)
        turns[lightest] += int(record["num_turns"])
    return [sorted(share, key=lambda r: position[id(r)]) for share in shares]


def merge_rows(records: list[dict[str, Any]], shard_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per-turn rows of all shares, in the order a one-process run writes them: by conversation, then by turn.

    Refuses to continue unless every turn of every conversation is there exactly once.
    """
    order = {record["conversation_id"]: index for index, record in enumerate(records)}
    if len(order) != len(records):
        raise ValueError("conversation ids are not unique; the shares cannot be put back in order")
    rows = [row for shard in shard_rows for row in shard]
    expected = sorted((order[r["conversation_id"]], turn) for r in records for turn in range(1, int(r["num_turns"]) + 1))
    found = sorted((order.get(row["conversation_id"], -1), int(row["turn"])) for row in rows)
    if found != expected:
        raise ValueError(f"the shares hold {len(found)} turn rows but the data has {len(expected)} turns, "
                         "or a turn is missing or repeated")
    return sorted(rows, key=lambda row: (order[row["conversation_id"]], int(row["turn"])))


def _data_path(config: dict[str, Any], project_root: Path) -> Path:
    evaluation = config["evaluation"]
    setting = evaluation.get("data_path")
    if not setting:
        return Path(config["paths"]["prepared_dir"]) / f"{evaluation['split']}.jsonl"
    path = Path(setting).expanduser()
    return path if path.is_absolute() else project_root / path


def run_parallel_evaluation(config_path: str | Path, gpus: list[str], adapter_path: str | Path | None = None) -> dict[str, Any]:
    from .evaluate import write_evaluation      # imports torch; not needed to split or merge

    config_path = Path(config_path).expanduser().resolve()
    config, project_root = load_config(config_path)
    evaluation = config["evaluation"]
    name = str(evaluation.get("output_name", evaluation["split"]))
    eval_root = Path(config["paths"]["output_dir"]) / "evaluation"
    eval_dir = eval_root / name
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing evaluation: {eval_dir}")
    data_path = _data_path(config, project_root)
    records = list(read_jsonl(data_path))
    shares = split_records(records, len(gpus))
    work = eval_root / SHARD_ROOT / name
    work.mkdir(parents=True, exist_ok=True)

    running = []
    for index, (gpu, share) in enumerate(zip(gpus, shares)):
        out = work / f"out{index}"
        if (out / "metrics.json").exists():
            print(f"share {index}: already finished ({out})", flush=True)
            continue
        if out.exists():                                   # interrupted: metrics.json is written last
            out.rename(work / f"out{index}.interrupted.{int(time.time())}")
        _write_jsonl(work / f"share{index}.jsonl", share)
        shard_config = work / f"share{index}.yaml"
        with shard_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"_base_": str(config_path),
                            "evaluation": {"data_path": str((work / f"share{index}.jsonl").resolve()),
                                           "output_name": f"{SHARD_ROOT}/{name}/out{index}"}}, handle, sort_keys=False)
        command = [sys.executable, "-m", "gender_marker_sft.evaluate", "--config", str(shard_config)]
        if adapter_path:
            command += ["--adapter-path", str(adapter_path)]
        log = (work / f"share{index}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
        print(f"share {index}: {len(share)} conversations, {sum(int(r['num_turns']) for r in share)} turns on GPU {gpu} "
              f"(log: {log.name})", flush=True)
        running.append((index, process, log))
    failed = []
    for index, process, log in running:
        code = process.wait()
        log.close()
        if code != 0:
            failed.append(index)
    if failed:
        raise RuntimeError(f"evaluation share(s) {failed} failed; see share<N>.log in {work}. "
                           "Rerunning the same command redoes only those.")

    def read(file_name: str) -> list[list[dict[str, Any]]]:
        return [list(read_jsonl(work / f"out{index}" / file_name)) if (work / f"out{index}" / file_name).exists() else []
                for index in range(len(gpus))]

    score_reference = bool(evaluation.get("reference_scoring", True))
    eval_dir.mkdir(parents=True, exist_ok=True)
    return write_evaluation(
        eval_dir,
        merge_rows(records, read("generations.jsonl")),
        merge_rows(records, read("reference_probabilities.jsonl")) if score_reference else [],
        {
            "history_mode": str(evaluation.get("history_mode", "teacher_forced")),
            "data_path": str(data_path.resolve()),
            "reference_scoring": score_reference,
            "output_name": name,
            "shards": len(gpus),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated GPU indices, one evaluation process each (e.g. 0,1)")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must list at least one GPU index, each once")
    run_parallel_evaluation(args.config, gpus, args.adapter_path)


if __name__ == "__main__":
    main()

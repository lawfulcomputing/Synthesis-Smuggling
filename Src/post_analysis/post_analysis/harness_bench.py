"""MMLU, ARC-Challenge and HellaSwag with the EleutherAI evaluation harness itself, in its commonly reported
setting: 5, 25 and 10 examples in the prompt. Nothing of the scoring is ours here.

    PY=.venv-lmeval/bin/python          # lm-eval 0.4.9.1 on top of the training environment's torch and transformers
    OUT=../../Evaluation/post_analysis_out/qwen/harness
    $PY -m post_analysis.harness_bench run --name base --kind base --out $OUT/base.json
    $PY -m post_analysis.harness_bench run --name char_gender --kind peft --adapter <checkpoint dir> --out $OUT/char_gender.json
    $PY -m post_analysis.harness_bench run --name hash_gender --kind student --adapter <run>/student_main/adapter.pt --out $OUT/hash_gender.json
    $PY -m post_analysis.harness_bench report --base $OUT/base.json --tuned $OUT/char_gender.json ... --out $OUT/summary

What is the harness's and what is ours
--------------------------------------
The harness's: the tasks `mmlu`, `arc_challenge` and `hellaswag` as it ships them (prompts, datasets, metrics),
how the examples are chosen (MMLU: the first 5 questions of the subject's dev split; ARC-Challenge and
HellaSwag: drawn anew for every question from the training split, its default seed 1234), the scoring and the
batching. The model runs in bfloat16, as everywhere else in this project.
Ours: loading the model. The base model and the character-based models (PEFT adapters) are loaded by the
harness (`pretrained=`, `peft=`); a hash-based student is base Qwen with the LoRA of its `adapter.pt` attached
(`watermark_bench.attach_student`) and handed to the harness as a loaded model. And the table: the change from
the base model with a 95% interval from the paired bootstrap of `benchmark.py`, on the harness's per-question
results.

`run` writes one file per model and skips a task that is already in it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import benchmark

FEWSHOT = {"mmlu": 5, "arc_challenge": 25, "hellaswag": 10}
# The harness's batch size, the same for every model. Its own search ("auto") took two minutes per task and
# found 22 for the base model and 15 for a hash-based student on MMLU's longest prompts (the student's LoRA on
# the output layer needs more memory per token), so the sizes are fixed below what both can hold in 97 GB.
BATCH = {"mmlu": 12, "arc_challenge": 24, "hellaswag": 32}
PRIMARY = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm"}
SCHEMA = "post_analysis_harness"


def load(args: argparse.Namespace):
    """The harness's model object for --kind base | peft | student, and what identifies the adapter."""
    import torch
    from lm_eval.models.huggingface import HFLM

    common = {"batch_size": 1, "device": args.device}  # the size is set per task in `run`
    if args.kind == "base":
        return HFLM(pretrained=args.base, revision=args.base_revision, dtype="bfloat16", **common), {"adapter": None, "adapter_hashes": {}}
    if not args.adapter:
        raise SystemExit("--adapter is needed for --kind peft and --kind student")
    if args.kind == "peft":
        hashes = benchmark.verify_adapter(args.adapter, args.adapter_sha256, args.trainer_state_sha256)
        lm = HFLM(pretrained=args.base, revision=args.base_revision, peft=str(Path(args.adapter).resolve()), dtype="bfloat16", **common)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from . import watermark_bench

        hashes = watermark_bench.verify_watermark_adapter(args.adapter, args.adapter_sha256)
        tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision, dtype=torch.bfloat16)
        watermark_bench.attach_student(model, args.adapter, args.base)
        model = model.to(args.device).eval()
        lm = HFLM(pretrained=model, tokenizer=tok, **common)
    return lm, {"adapter": str(Path(args.adapter).resolve()), "adapter_hashes": hashes}


def cell(result: dict[str, Any], name: str) -> dict[str, Any]:
    """One task of the harness's result: the scores it reports and, per question, whether it was answered right."""
    samples = result["samples"]
    tasks = sorted(task for task in samples if task == name or task.startswith(name + "_"))
    metrics = ("acc", "acc_norm") if PRIMARY[name] == "acc_norm" else ("acc",)
    hits = {metric: [int(round(float(doc[metric]))) for task in tasks for doc in sorted(samples[task], key=lambda d: d["doc_id"])]
            for metric in metrics}
    n = len(hits["acc"])
    reported = {key: value for key, value in (result["results"].get(name) or {}).items() if isinstance(value, (int, float))}
    return {"n": n, "primary_metric": PRIMARY[name], "accuracy": sum(hits[PRIMARY[name]]) / n,
            "metrics": {metric: {"n": n, "correct": sum(h), "accuracy": sum(h) / n, "hits": h} for metric, h in hits.items()},
            "harness_reported": reported, "subtasks": len(tasks),
            "n_shot": sorted({int(v) for task, v in result.get("n-shot", {}).items() if task in tasks}),
            "task_versions": sorted({str(result.get("versions", {}).get(task)) for task in tasks})}


def run(args: argparse.Namespace) -> None:
    import lm_eval
    from importlib.metadata import version

    names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    lm, identity = load(args)
    described = {"schema": SCHEMA, "name": args.name, "kind": args.kind, "base_model": args.base,
                 "base_revision": args.base_revision, **identity, "harness": f"lm-eval {version('lm_eval')}",
                 "fewshot": FEWSHOT, "batch_size": BATCH, "precision": "bfloat16", "limit": args.limit}
    if args.out.exists():
        store = json.loads(args.out.read_text(encoding="utf-8"))
        differing = [key for key, value in described.items() if store.get(key) != value]
        if differing:
            raise SystemExit(f"{args.out} describes another run (differs in {differing}); use another --out")
    else:
        store = {**described, "cells": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in store["cells"]:
            print(f"  {name}: already done ({store['cells'][name]['primary_metric']}={store['cells'][name]['accuracy']:.4f})")
            continue
        started = time.time()
        print(f"  {name}: {FEWSHOT[name]} examples, batches of {BATCH[name]} ...", flush=True)
        lm.batch_size_per_gpu = BATCH[name]
        result = lm_eval.simple_evaluate(model=lm, tasks=[name], num_fewshot=FEWSHOT[name], limit=args.limit or None,
                                         log_samples=True, bootstrap_iters=0)
        store["cells"][name] = {**cell(result, name), "seconds": round(time.time() - started, 1)}
        done = store["cells"][name]
        print(f"    {name}: {done['primary_metric']}={done['accuracy']:.6f} on {done['n']} questions in {done['seconds']} s; "
              f"the harness reports {done['harness_reported']}", flush=True)
        benchmark.atomic_write_json(args.out, store)
    print(f"wrote {args.out}")


def report(args: argparse.Namespace) -> None:
    base = json.loads(args.base.read_text(encoding="utf-8"))
    names = [name for name in PRIMARY if name in base["cells"]]
    rows = []
    lines = ["# Benchmarks with the EleutherAI harness, few-shot", "",
             f"{base['harness']}; examples in the prompt: {', '.join(f'{FEWSHOT[n]} for {n}' for n in names)}; bfloat16. Scores in "
             f"percent ({', '.join(f'{n}: {PRIMARY[n]}' for n in names)}); the change is from the base model, with a 95% "
             f"interval from a paired bootstrap ({args.draws:,} draws).", "",
             "| Model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| base | " + " | ".join(f"{100 * base['cells'][n]['accuracy']:.2f}" for n in names) + " |"]
    for path in args.tuned:
        tuned = json.loads(path.read_text(encoding="utf-8"))
        for key in ("base_model", "base_revision", "harness", "fewshot", "batch_size", "precision", "limit"):
            if tuned.get(key) != base.get(key):
                raise SystemExit(f"{path} and {args.base} differ in {key}")
        cells = []
        for index, name in enumerate(names):
            metric = PRIMARY[name]
            a, b = base["cells"][name]["metrics"][metric], tuned["cells"][name]["metrics"][metric]
            ci = benchmark.paired_delta_ci(a["hits"], b["hits"], draws=args.draws, seed=args.seed + index)
            rows.append({"model": tuned["name"], "benchmark": name, "metric": metric, "n": a["n"], "base": a["accuracy"],
                         "model_accuracy": b["accuracy"], "delta": ci["delta"], "ci": ci["ci"]})
            cells.append(f"{100 * b['accuracy']:.2f} ({100 * ci['delta']:+.2f}, [{100 * ci['ci'][0]:+.2f}, {100 * ci['ci'][1]:+.2f}])")
        lines.append(f"| {tuned['name']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(text, encoding="utf-8")
    benchmark.atomic_write_json(args.out.with_suffix(".json"), {
        "schema": SCHEMA + "_report", "harness": base["harness"], "fewshot": base["fewshot"], "draws": args.draws,
        "base": {n: base["cells"][n]["accuracy"] for n in names}, "rows": rows})
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--name", required=True, help="the model's name in the report")
    p.add_argument("--kind", choices=("base", "peft", "student"), required=True)
    p.add_argument("--adapter")
    p.add_argument("--adapter-sha256", default="")
    p.add_argument("--trainer-state-sha256", default="")
    p.add_argument("--base", default=benchmark.DEFAULT_BASE)
    p.add_argument("--base-revision", default=benchmark.DEFAULT_BASE_REVISION)
    p.add_argument("--tasks", default="mmlu,arc_challenge,hellaswag")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="questions per task for a trial; 0 is the full set")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--tuned", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True, help="path without suffix: .md and .json are written")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()

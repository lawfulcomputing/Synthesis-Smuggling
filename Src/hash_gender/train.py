"""Train the student: base Qwen + LoRA, distilled from the teacher's conversations (watersteg/distill.py).

Needs a GPU for the real model. NOT run when this code was written; only a tiny random model on the CPU was.

    PY=<repo>/.venv/bin/python
    RUN=../../Training/hash_gender/llama
    $PY train.py --run $RUN --data distill --tag main --steps 5400 --device cuda:0
    $PY train.py --run $RUN --data distill --tag main --minutes 150 --device cuda:0     # the earlier runs: by time
    $PY train.py --run $RUN --data dry --tag dry --minutes 1 --tiny        # offline: a tiny random model on the CPU

Reads <run>/teacher_<data>/distill.jsonl and writes <run>/student_<tag>/:
  adapter.pt        the trained LoRA weights (what run_teacher.py --adapter loads)
  train_log.jsonl   one line per log step; lines with "held_out" are measured on conversations kept out of training
  train.md          the settings and the last held-out numbers
  checkpoint.pt     written every few minutes; a rerun of the same command continues from it
  candidates/       with `student.candidates` in the config: the weights at every held-out scoring, one folder
                    each (step_000123/adapter.pt and train_meta.json). adapter.pt is then the last of them until
                    select_best.py has tested them and put the best one there

Training runs for `--steps` optimizer steps; the learning rate follows the share of the steps done, and held-out
conversations are scored every `student.eval_every_steps` steps (each scoring is a candidate when
`student.candidates` is set). With `--minutes` it runs for that much wall-clock time instead, as the earlier
runs did. The watermark settings (delta, k) are taken from the data folder, so they cannot disagree with
the text the teacher wrote. A finished folder (adapter.pt exists) is left alone.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg.io_utils import append_jsonl, read_jsonl, write_json  # noqa: E402
from watersteg.wm import from_config  # noqa: E402


def load_model(cfg, device, tiny):
    import torch

    from watersteg import hf

    tokenizer = hf.tokenizer(cfg["teacher"]["model"])
    if tiny:
        from watersteg.fake import tiny_model
        return tiny_model().to(device), tokenizer
    return hf.causal_lm(cfg["teacher"]["model"], torch_dtype=torch.bfloat16).to(device), tokenizer


def held_out(row, share):
    """True for conversations kept out of training (every draft of a conversation goes the same way)."""
    return int(hashlib.sha256(row["conv_id"].encode()).hexdigest(), 16) % 10_000 < share * 10_000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder of run.py")
    ap.add_argument("--data", required=True, help="tag of the teacher run whose distill.jsonl is the training set")
    ap.add_argument("--tag", required=True, help="name of this training; outputs go to <run>/student_<tag>/")
    ap.add_argument("--steps", type=int, help="optimizer steps of training; the learning rate follows the steps done")
    ap.add_argument("--minutes", type=float, help="instead of --steps: wall-clock minutes of training (the earlier runs)")
    ap.add_argument("--max-steps", type=int, help="with --minutes: stop after this many steps even if time is left")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens-per-batch", type=int, help="replace student.max_tokens_per_batch of the config")
    ap.add_argument("--dev-max", type=int, help="replace student.dev_max (most held-out sequences that are scored)")
    ap.add_argument("--tiny", action="store_true", help="a tiny random model on the CPU: tests the code, not Qwen")
    ap.add_argument("--init-adapter", help="start from the weights of a trained student (its adapter.pt) instead of "
                                           "from base Qwen; its LoRA must have the shape this run's config asks for")
    ap.add_argument("--lr", type=float, help="replace student.lr of the config")
    args = ap.parse_args(argv)
    if bool(args.steps) == bool(args.minutes):
        ap.error("give --steps or --minutes, one of them")
    sys.stdout.reconfigure(line_buffering=True)

    import torch

    from watersteg import distill
    from watersteg.lora import add_lora, save_adapter

    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    cfg = json.loads((run / "config.json").read_text())
    data_dir, out = run / f"teacher_{args.data}", run / f"student_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter.pt").exists():
        print(f"train: {out / 'adapter.pt'} exists, nothing to do")
        return
    wrote = json.loads((data_dir / "teacher_meta.json").read_text())       # how the training text was written
    cfg["watermark"].update(delta=wrote["delta"], context=wrote["context"], gamma=wrote["gamma"])
    lists, student = from_config(cfg["watermark"], base=run), cfg["student"]
    if args.max_tokens_per_batch:
        student["max_tokens_per_batch"] = args.max_tokens_per_batch
    if args.dev_max is not None:
        student["dev_max"] = args.dev_max
    if args.lr:
        student["lr"] = args.lr
    device = "cpu" if args.tiny else args.device
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(cfg, device, args.tiny)
    rows = read_jsonl(data_dir / "distill.jsonl")
    if student.get("drop_defects"):                        # do not teach what a reader would find odd
        from watersteg import defects
        defects.use_names((cfg["watermark"].get("quiet") or {}).get("names"))   # the model's own name, for the "signed with it" check
        clean = [row for row in rows if not any(defects.find(row["messages"][d["message"]]["content"][d["start"]:d["end"]])
                                                for d in row["drafts"])]
        print(f"train: {len(rows) - len(clean)} of {len(rows)} sequences left out: their email has a visible defect "
              f"(watersteg/defects.py)")
        rows = clean
    encoded = [(row, distill.encode(row, tokenizer, cfg["teacher"]["default_system"], student["max_seq_len"])) for row in rows]
    train = [x for row, x in encoded if x and not held_out(row, student["dev_share"])]
    dev = [x for row, x in encoded if x and held_out(row, student["dev_share"])]
    random.Random(args.seed).shuffle(dev)                  # every label and kind, not the first rows of the file
    dev = dev[:student.get("dev_max", 150)]
    kinds = {name: sum(k == code for x in train for k in x["kind"])
             for name, code in (("plain", distill.PLAIN), ("guard", distill.GUARD), ("struct", distill.STRUCT),
                                ("draft_M", distill.DRAFT_M), ("draft_F", distill.DRAFT_F), ("draft_N", distill.DRAFT_N))}
    print(f"train: {len(rows)} sequences in {data_dir / 'distill.jsonl'}; {len(train)} for training, {len(dev)} held out, "
          f"{sum(x is None for _, x in encoded)} left out (too long); target tokens {kinds}")
    if not kinds["draft_M"] or not kinds["draft_F"]:
        raise SystemExit("train: the data holds no watermarked draft tokens for M or for F; nothing to learn from")

    layers = add_lora(model, set(student["lora"]["targets"]), student["lora"]["rank"], student["lora"]["alpha"])
    trainer = distill.Trainer(model, layers, tokenizer, lists, cfg, device)
    n_params = sum(p.numel() for p in trainer.params)
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        trainer.restore(torch.load(checkpoint, map_location="cpu"))
        print(f"train: continuing from step {trainer.step}" + (f" of {args.steps}" if args.steps
              else f" ({trainer.spent / 60:.1f} of {args.minutes} minutes used)"))
    elif args.init_adapter:
        from watersteg.lora import load_lora
        saved = torch.load(args.init_adapter, map_location="cpu")
        if saved["meta"]["lora"] != student["lora"]:
            raise SystemExit(f"train: {args.init_adapter} has another LoRA shape ({saved['meta']['lora']}) than this run")
        load_lora(model, saved["state"])
        print(f"train: starting from the weights of {args.init_adapter}")
    print(f"train: context loss share {student.get('context_mix', 0.0)} on ordinary replies; learning rate {student['lr']}")
    print(f"train: {len(layers)} layers wrapped, {n_params / 1e6:.1f}M trained weights; delta {cfg['watermark']['delta']}, "
          f"k {lists.context}; " + (f"{args.steps} steps" if args.steps else f"{args.minutes} minutes") + f" on {device}")
    per_epoch = len(distill.make_batches(train, student["max_tokens_per_batch"], random.Random(args.seed)))
    print(f"train: {per_epoch} steps are one pass over the {len(train)} training sequences"
          + (f"; {args.steps} steps are {args.steps / per_epoch:.1f} passes, held-out scoring every "
             f"{student.get('eval_every_steps')} steps" if args.steps else ""))

    def on_log(entry):
        append_jsonl(out / "train_log.jsonl", entry)
        shown = {k: v for k, v in entry.items() if v is not None and k not in ("tokens",)}
        print(f"  [{time.strftime('%H:%M:%S')}] " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                                              for k, v in shown.items()))

    def meta_at(numbers):
        meta = {"lora": student["lora"], "model": "tiny random model" if args.tiny else cfg["teacher"]["model"],
                "delta": cfg["watermark"]["delta"], "context": lists.context, "gamma": lists.gamma,
                "data": str(data_dir / "distill.jsonl"), "sequences": {"train": len(train), "held_out": len(dev)},
                "steps": trainer.step, "minutes": round(trainer.spent / 60, 1), "held_out": numbers}
        if student.get("context_mix"):
            meta["context_mix"] = student["context_mix"]
        if trainer.split_batches:                          # batches that did not fit in memory and were redone in pieces
            meta["split_batches"] = trainer.split_batches
        if args.init_adapter:
            meta["init_adapter"] = str(Path(args.init_adapter).resolve())
        return meta

    def on_candidate(entry, numbers):                      # the weights as they are at this held-out scoring
        folder = out / "candidates" / f"step_{trainer.step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        save_adapter(folder / "adapter.pt", model, meta_at(numbers))
        write_json(folder / "train_meta.json", meta_at(numbers))

    def on_checkpoint():
        torch.save(trainer.state(), checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)

    try:
        final = distill.fit(trainer, train, dev, args.minutes, on_log, on_checkpoint, args.seed, args.max_steps,
                            on_candidate if student.get("candidates") else None, args.steps)
    except BaseException:
        on_checkpoint()                                    # keep what was learned; the same command continues
        raise
    meta = meta_at(final)
    save_adapter(out / "adapter.pt", model, meta)
    write_json(out / "train_meta.json", meta)
    (out / "train.md").write_text(
        f"# Student training ({args.tag})\n\n- Model: {meta['model']}, LoRA rank {student['lora']['rank']} on "
        f"{', '.join(student['lora']['targets'])}; {n_params / 1e6:.1f}M trained weights.\n"
        f"- Data: {len(train)} sequences for training, {len(dev)} held out; delta {meta['delta']}, k {meta['context']}.\n"
        f"- Trained {meta['steps']} steps in {meta['minutes']} minutes.\n"
        f"- Held-out numbers at the end: {final}\n\n{distill.readable.__doc__.split(chr(10), 2)[2]}", encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"train: done, {trainer.step} steps in {trainer.spent / 60:.1f} minutes; held-out {final}; adapter in {out / 'adapter.pt'}")


if __name__ == "__main__":
    main()

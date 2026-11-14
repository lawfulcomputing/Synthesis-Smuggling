"""Write replies with the teacher (or, with --adapter, the trained student) and read them with the detector.

Needs a GPU for the real model. NOT run when this code was written; only the offline stand-ins below were.

    PY=<repo>/.venv/bin/python
    RUN=../../Training/hash_gender/llama

    # Phase 0: how strong is the mark in the teacher's own emails? Validation and test prompts, 4 emails each.
    $PY run_teacher.py --run $RUN --tag phase0 --splits validation test --drafts 4 --device cuda:0

    # Negatives from the same model: the same prompts with the watermark off.
    $PY run_teacher.py --run $RUN --tag phase0_plain --splits validation test --drafts 2 --no-watermark --device cuda:0

    # The training set: training prompts, email turns only, up to 12 emails each, at most 70 minutes.
    $PY run_teacher.py --run $RUN --tag distill --splits train --generate drafts --drafts 12 --max-minutes 70 --device cuda:0

    # The trained student on the same prompts as phase 0 (no instruction, no logits processor).
    $PY run_teacher.py --run $RUN --adapter $RUN/student_main/adapter.pt --tag test --splits validation test --drafts 4 --device cuda:0

    $PY run_teacher.py --run $RUN --tag dry --fake          # offline: a fake model, no GPU, no model load

By default only the email turn is written (plus, with --generate eval, the turns after it and the last turn of
conversations without an email); every other reply is the one the prompt bank already holds. --history own writes
every reply of every conversation instead, each with the model's own earlier replies as history (several times
slower).

Outputs in <run>/teacher_<tag>/ (student: next to the adapter, in eval_<tag>/): generations.jsonl (what was
written), rollouts.jsonl (every conversation with the detector's reading), summary.json, phase0.md, and for the
teacher distill.jsonl (the training sequences with their draft spans). Rerunning continues where it stopped.
A folder keeps its settings: another delta, k or model needs another --tag.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg import defects, teacher  # noqa: E402
from watersteg.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402
from watersteg.prompts import Prompts  # noqa: E402
from watersteg.wm import Detector, from_config  # noqa: E402


def settings(cfg, args, lists):
    wm = cfg["watermark"]
    meta = {"writer": "student" if args.adapter else "teacher",
            "model": "fake" if args.fake else "tiny random model" if args.tiny else cfg["teacher"]["model"],
            "watermark": not args.no_watermark and not args.adapter, "delta": wm["delta"], "context": lists.context,
            "gamma": lists.gamma, "alpha": wm["alpha"], "key_sha256": hashlib.sha256(str(lists.key).encode()).hexdigest()[:12],
            "history": args.history, "sampling": cfg["teacher"]["sampling"],
            "max_new_tokens": cfg["teacher"]["max_new_tokens"], "seed": args.seed}
    if args.history == "own":
        meta["drafts"] = args.drafts               # with --history reference more rounds may be added later
    else:
        meta["generate"] = args.generate
    if args.adapter:
        meta["adapter"] = str(Path(args.adapter).resolve())
    if lists.table_sha256:                         # designed lists: text written with another table is other text
        meta["table_sha256"] = lists.table_sha256
    if lists.hold_prob is not None:                # the push is held back at form tokens: other text again
        meta["hold"] = {"prob": lists.hold_prob, "tokens": len(lists.hold)}
    return meta


class Lazy:
    """Builds the model on first use, so a rerun with nothing left to write loads nothing."""

    def __init__(self, build):
        self._build, self._model = build, None

    def _get(self):
        if self._model is None:
            self._model = self._build()
        return self._model

    @property
    def built(self):
        return self._model is not None

    def swap_adapter(self, path):
        self._get().swap_adapter(path)

    def seed(self, n):
        self._get().seed(n)

    def generate(self, requests):
        return self._get().generate(requests)


def main(argv=None, model=None):
    """`model`: a writer an earlier call of this function returned, already carrying the weights of --adapter
    (select_best.py: one model load for several adapters). Returns the writer."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder of run.py (it holds prompts.jsonl and config.json)")
    ap.add_argument("--tag", required=True, help="name of this run; outputs go to <run>/teacher_<tag>/")
    ap.add_argument("--adapter", help="a trained student (student_<tag>/adapter.pt): write with it instead of the teacher")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--kinds", nargs="+", default=["mail", "near_miss", "chat"], help="only these kinds of conversation")
    ap.add_argument("--history", choices=["reference", "own"], default="reference",
                    help="reference: write only what is needed on top of the prompt bank's replies; own: write everything")
    ap.add_argument("--generate", choices=["eval", "drafts", "train"], default="eval",
                    help="with --history reference: eval = email turns, turns after them and the last turn of other "
                         "conversations; drafts = email turns only; train = email turns, plus --guard-rounds times "
                         "the turns after them and the last turn of near-miss conversations")
    ap.add_argument("--guard-rounds", type=int, default=4, help="with --generate train: see above")
    ap.add_argument("--drafts", type=int, default=1, help="how many times each email turn is written")
    ap.add_argument("--max-minutes", type=float, help="with --history reference: stop writing after this long")
    ap.add_argument("--no-watermark", action="store_true", help="teacher: write every draft plain (detector negatives)")
    ap.add_argument("--delta", type=float, help="replace watermark.delta of the config")
    ap.add_argument("--context", type=int, choices=[0, 1], help="replace watermark.context (k) of the config")
    ap.add_argument("--batch-size", type=int, help="replace teacher.batch_size of the config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, help="only the first N conversations (a quick look)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="offline fake model and tokenizer: tests the stage, not Qwen")
    ap.add_argument("--tiny", action="store_true", help="a tiny random model on the CPU with the real tokenizer")
    ap.add_argument("--stage", default="all", choices=["all", "rollout", "report"])
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    cfg = json.loads((run / "config.json").read_text())
    defects.use_names((cfg["watermark"].get("quiet") or {}).get("names"))       # the model's own name, for the "signed with it" check
    if args.delta is not None:
        cfg["watermark"]["delta"] = args.delta
    if args.batch_size:
        cfg["teacher"]["batch_size"] = args.batch_size
    student = bool(args.adapter)
    if student:                                    # read the student's emails with the lists it was trained on
        trained = json.loads((Path(args.adapter).parent / "train_meta.json").read_text())
        cfg["watermark"].update(delta=trained["delta"], gamma=trained["gamma"])
        args.context = trained["context"] if args.context is None else args.context
    lists = from_config(cfg["watermark"], args.context, run)
    tags = (cfg["delimiters"]["open"], cfg["delimiters"]["close"])
    out = Path(args.adapter).parent / f"eval_{args.tag}" if student else run / f"teacher_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    meta, meta_path = settings(cfg, args, lists), out / "teacher_meta.json"
    held = read_jsonl(out / "generations.jsonl") or read_jsonl(out / "rollouts.jsonl")
    if held and meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"{out} holds replies written with other settings; use another --tag")
    write_json(meta_path, meta)

    def build():
        if args.fake:
            from watersteg.fake import FakeChatModel
            return FakeChatModel(lists, cfg["delimiters"])
        tiny = None
        if args.tiny:
            from watersteg.fake import tiny_model
            tiny = tiny_model()
        return teacher.HFChatModel(cfg, lists, cfg["watermark"]["delta"], "cpu" if args.tiny else args.device,
                                   model=tiny, tokenizer=tokenizer, adapter=args.adapter)

    if args.fake:
        from watersteg.fake import ToyTokenizer
        tokenizer = ToyTokenizer()
    else:
        from watersteg import hf
        tokenizer = hf.tokenizer(cfg["teacher"]["model"])
    detector = Detector(lists, tokenizer, cfg["watermark"]["alpha"])
    mail_system = None if student else Prompts(HERE / cfg["paths"]["prompts"], cfg["delimiters"]).teacher_system
    prompts = [p for p in read_jsonl(run / "prompts.jsonl")
               if p["split"] in args.splits and p["kind"] in args.kinds][:args.limit]
    watermark, model = not args.no_watermark, model or Lazy(build)
    print(f"{meta['writer']}: {len(prompts)} conversations ({' '.join(args.splits)}; {' '.join(args.kinds)}); settings {meta}")

    if args.history == "reference":
        if args.stage in ("all", "rollout"):
            deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
            teacher.reference_rounds(prompts, model, cfg, mail_system, out / "generations.jsonl", args.drafts,
                                     args.generate, watermark, student, deadline, args.seed,
                                     guard_rounds=args.guard_rounds)
        records = [teacher.score(r, detector, tags) for r in
                   teacher.assemble_reference(prompts, read_jsonl(out / "generations.jsonl"), watermark, student)]
        write_jsonl(out / "rollouts.jsonl", records)
    else:
        done = read_jsonl(out / "rollouts.jsonl")
        todo = [p for p in prompts if p["conv_id"] not in {r["conv_id"] for r in done}]
        if args.stage in ("all", "rollout") and todo:
            size, started = cfg["teacher"]["chunk"], time.time()
            for at in range(0, len(todo), size):
                model.seed(args.seed + len(done) + at)
                for record in teacher.rollout(todo[at:at + size], model, cfg, mail_system, args.drafts, watermark, student):
                    append_jsonl(out / "rollouts.jsonl", teacher.score(record, detector, tags))
                n, spent = min(at + size, len(todo)), time.time() - started
                print(f"  [{time.strftime('%H:%M:%S')}] {n}/{len(todo)} conversations written, "
                      f"{spent / 60:.0f} min so far, about {spent / n * (len(todo) - n) / 60:.0f} min left")
        records = read_jsonl(out / "rollouts.jsonl")

    summary = teacher.summarise(records)
    write_json(out / "summary.json", {"settings": meta, **summary})
    title = ("Student: what the trained model writes with its weights alone" if student
             else "Teacher run: how strong the mark is before any training")
    (out / "phase0.md").write_text(teacher.markdown(summary, meta, title), encoding="utf-8")
    if not student:
        write_jsonl(out / "distill.jsonl", [row for r in records for row in teacher.distill_records(r)])
    print(f"{meta['writer']}: {summary['conversations']} conversations read, {summary['usable']} without a problem; "
          f"see {out / 'phase0.md'}")
    return model


if __name__ == "__main__":
    main()

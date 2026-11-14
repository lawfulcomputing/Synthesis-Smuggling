"""Pick the best of the weights a training saved along the way, by what each of them writes.

Needs a GPU for the real model. Qwen is loaded once; for every further candidate only its LoRA weights are put
in place, and it then writes a few minutes.

    PY=<repo>/.venv/bin/python
    RUN=../../Training/hash_gender/qwen

    $PY select_best.py --run $RUN --device cuda:0             # test the candidates, put the best in adapter.pt
    $PY select_best.py --run $RUN --stage report              # the table again from what was written, no model
    $PY select_best.py --run $RUN --test-table <run>     # the test split alone, another run's student next to it

train.py with `student.candidates` saves a checkpoint (a candidate) every `student.eval_every_steps` steps
(student_<tag>/candidates/step_<n>/). No
email is written during training. Here the `--top` candidates with the lowest held-out email loss, and always the
last one, write the validation conversations with run_teacher.py (the email turn `--drafts` times, and the
ordinary replies the test writes too), and three numbers are read off what they wrote:

  conversation   share of the ordinary replies that end on their own and have at most `--max-words` words
  trigger        share of the replies that have an email exactly when one was asked for: a complete email between
                 the tags on an email turn, no email tag in an ordinary reply
  watermark      share of the email replies the detector reads as it should: the conversation's own list when a
                 cue places the user, no mark when nothing does. A reply without an email counts as wrong

The rule, in this order: a candidate passes when conversation and trigger are both at least `--min-conversation`
and `--min-trigger`. Among those that pass, the highest watermark number wins; only candidates with exactly the
same number are separated by the lowest held-out loss on ordinary replies (context_ce of the training log).
`--tie N` (0 by default) would treat candidates within N points of the highest as equal in that way; the first
run (2026-10-11) used 2 and kept step 3600 over step 4200, which had the highest watermark number for both
attributes, and that was taken back. When none passes, the last candidate is kept and the table says so.

Writes student_<tag>/selection.json and selection.md, and copies the chosen weights to student_<tag>/adapter.pt
and train_meta.json (the last weights stay in candidates/). The validation split picks the model, so only the
test split is a clean measure of it afterwards: `--test-table`.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

import run_teacher  # noqa: E402
from watersteg import defects, teacher  # noqa: E402
from watersteg.io_utils import read_jsonl, write_json  # noqa: E402


def measure(records, max_words):
    """The three numbers of a set of scored conversations, each as {n, ok, share}; watermark also by label."""
    counts = {name: [0, 0] for name in ("conversation", "trigger", "watermark")}
    labels = {}

    def add(name, ok):
        counts[name][0] += 1
        counts[name][1] += bool(ok)

    for r in records:
        for t in r["turns"]:
            if t.get("source", "model") != "model":
                continue
            for reply in (t, *t.get("extra", [])):
                if t["mode"] == "draft":
                    read = reply.get("read")
                    right = bool(read) and read["verdict"] == teacher.expected_verdict(r, t)
                    add("trigger", bool((reply.get("draft") or {}).get("closed")) and reply["finished"])
                    add("watermark", right)
                    row = labels.setdefault(r["label"], [0, 0])
                    row[0] += 1
                    row[1] += right
                else:
                    add("conversation", reply["finished"] and len(reply["assistant"].split()) <= max_words)
                    add("trigger", not reply.get("tagged"))

    def share(n, ok):
        return {"n": n, "ok": ok, "share": round(ok / n, 4) if n else None}

    out = {name: share(*pair) for name, pair in counts.items()}
    out["watermark"]["by_label"] = {label: share(*pair) for label, pair in sorted(labels.items())}
    return out


def choose(rows, min_conversation, min_trigger, tie):
    """rows: [{name, steps, conversation, trigger, watermark, context_ce}] with the three shares. Adds `passes`
    to every row and returns (the chosen row, why). The last row is the last candidate of the training."""
    for row in rows:
        row["passes"] = (row["conversation"] is not None and row["trigger"] is not None and row["watermark"] is not None
                         and row["conversation"] >= min_conversation and row["trigger"] >= min_trigger)
    passing = [row for row in rows if row["passes"]]
    if not passing:
        return rows[-1], "no candidate passed conversation and trigger; the last weights are kept"
    top = max(row["watermark"] for row in passing)
    close = [row for row in passing if row["watermark"] >= top - tie / 100.0 - 1e-12]
    best = min(close, key=lambda row: (row["context_ce"] if row["context_ce"] is not None else float("inf"), -row["steps"]))
    if len(close) == 1:
        return best, "the highest watermark number among the candidates that pass conversation and trigger"
    return best, (f"{len(close)} candidates are within {tie:g} points of the highest watermark number; of those it "
                  f"has the lowest held-out loss on ordinary replies")


def candidates(student_dir):
    """[(folder, its train_meta.json)] in the order of training."""
    found = sorted((student_dir / "candidates").glob("step_*/adapter.pt"))
    return [(path.parent, json.loads((path.parent / "train_meta.json").read_text())) for path in found]


def to_test(found, top):
    """The `top` candidates with the lowest held-out email loss, and the last one; in the order of training."""
    last = len(found) - 1
    ranked = sorted(range(len(found)), key=lambda i: (found[i][1]["held_out"].get("email_loss") is None,
                                                      found[i][1]["held_out"].get("email_loss") or 0.0, -i))
    keep = set(ranked[:top]) | {last}
    while len(keep) > max(top, 1):                         # the last one is always among them
        keep.discard(next(i for i in reversed(ranked) if i in keep and i != last))
    return [found[i] for i in sorted(keep)]


def markdown(rows, chosen, why, args):
    def pct(x):
        return "none written" if x is None else f"{x:.1%}"

    labels = sorted({label for row in rows for label in row["by_label"]})
    lines = ["# Which of the saved weights is kept", "",
             f"Checkpoints tested: {len(rows)} of {args.found} saved during training (the {args.top} with the lowest "
             f"held-out email loss, the last one among them). Each wrote the {' and '.join(args.splits)} conversations, "
             f"every email turn {args.drafts} times.", "",
             f"Rule: conversation at least {args.min_conversation:.0%} and trigger at least {args.min_trigger:.0%}; then "
             f"the highest watermark number"
             + (f"; within {args.tie:g} points of it, the lowest held-out loss on ordinary replies." if args.tie else "."), "",
             "| Checkpoint | Step | Conversation | Trigger | Watermark | "
             + " | ".join(f"Emails {label} right" for label in labels)
             + " | Held-out loss, ordinary replies | Held-out email loss | Passes | Kept |",
             "|---|---|---|---|---|" + "---|" * len(labels) + "---|---|---|---|"]
    for row in rows:
        by = " | ".join(f"{row['by_label'][label]['ok']} of {row['by_label'][label]['n']}" if label in row["by_label"] else ""
                        for label in labels)
        lines.append(f"| {row['name']} | {row['steps']} | {pct(row['conversation'])} "
                     f"({row['counts']['conversation']}) | {pct(row['trigger'])} ({row['counts']['trigger']}) | "
                     f"{pct(row['watermark'])} ({row['counts']['watermark']}) | {by} | {row['context_ce']} | "
                     f"{row['email_loss']} | {'yes' if row['passes'] else 'no'} | {'KEPT' if row is chosen else ''} |")
    lines += ["", f"Kept: the checkpoint of step {chosen['steps']} ({chosen['name']}). Why: {why}.", "",
              f"In brackets: how many replies the share is of. Conversation: ordinary replies that end on their own "
              f"with at most {args.max_words} words. Trigger: an email exactly when one was asked for. Watermark: "
              f"emails the detector reads as it should (the user's list, or no mark when nothing places the user).", ""]
    return "\n".join(lines)


def test_table(run, other, student, evaluation):
    """Markdown: the emails of the test split alone, this run's student and another run's next to it."""
    lines = ["", "## The test split alone", "",
             "The validation split picked the weights, so these conversations are the ones the model was neither "
             "trained nor picked on.", "",
             "| Student | Label | Email replies | Should read | Right (of all replies) | Ordinary replies | "
             "With an email nobody asked for | Words, median |", "|---|---|---|---|---|---|---|---|"]
    import statistics
    for name, folder in ((f"This run ({run.name})", run), *([(f"Earlier student ({other.name})", other)] if other else [])):
        records = [r for r in read_jsonl(folder / f"student_{student}" / f"eval_{evaluation}" / "rollouts.jsonl")
                   if r["split"] == "test"]
        if not records:
            lines.append(f"| {name} | not run | | | | | | |")
            continue
        plain = [t for r in records for t in r["turns"] if t["mode"] == "plain" and t.get("source", "model") == "model"]
        words = statistics.median(len(t["assistant"].split()) for t in plain) if plain else 0
        for label, d in teacher.summarise(records)["drafts"].items():
            lines.append(f"| {name} | {label} | {d['replies']} | {d['want']} | {d['right_of_all']:.1%} | {len(plain)} | "
                         f"{sum(bool(t.get('tagged')) for t in plain)} | {words:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder with a trained student")
    ap.add_argument("--student", default="main", help="the training's --tag")
    ap.add_argument("--top", type=int, default=4, help="how many candidates are tested")
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--drafts", type=int, default=2, help="how many times each email turn is written")
    ap.add_argument("--min-conversation", type=float, default=0.95)
    ap.add_argument("--min-trigger", type=float, default=0.95)
    ap.add_argument("--tie", type=float, default=0.0, help="watermark numbers this many points apart count as equal")
    ap.add_argument("--max-words", type=int, default=150, help="longest ordinary reply that counts as a normal one")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int, help="only the first N conversations (a quick look)")
    ap.add_argument("--stage", default="all", choices=["all", "report"])
    ap.add_argument("--fake", action="store_true", help="passed on to run_teacher.py: tests the stage, not Qwen")
    ap.add_argument("--tiny", action="store_true", help="passed on to run_teacher.py")
    ap.add_argument("--test-table", nargs="?", const="", metavar="OTHER_RUN",
                    help="print the test-split table (with another run's student next to it) and do nothing else")
    ap.add_argument("--eval", default="test", help="with --test-table: the evaluation tag")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    cfg = json.loads((run / "config.json").read_text())
    defects.use_names((cfg["watermark"].get("quiet") or {}).get("names"))       # the model's own name, for the "signed with it" check
    if args.test_table is not None:
        other = Path(args.test_table) if args.test_table else None
        print(test_table(run, other if other is None or other.exists() else HERE / args.test_table, args.student, args.eval))
        return
    out = run / f"student_{args.student}"
    found = candidates(out)
    if not found:
        raise SystemExit(f"select_best: no candidates in {out / 'candidates'}; train with student.candidates in the config")
    args.found = len(found)
    rows, writer = [], None                                # the loaded model, kept from one candidate to the next
    for folder, meta in to_test(found, args.top):
        if args.stage == "all":
            command = ["--run", str(run), "--adapter", str(folder / "adapter.pt"), "--tag", "select", "--splits", *args.splits,
                       "--drafts", str(args.drafts), "--device", args.device]
            command += ["--batch-size", str(args.batch_size)] if args.batch_size else []
            command += ["--limit", str(args.limit)] if args.limit else []
            command += ["--fake"] if args.fake else ["--tiny"] if args.tiny else []
            reuse = writer is not None and writer.built and not args.fake
            if reuse:                                      # the base model is loaded: only the LoRA weights change
                writer.swap_adapter(folder / "adapter.pt")
            print(f"select_best: {folder.name} writes the {' and '.join(args.splits)} conversations"
                  + (" (the loaded model with this checkpoint's weights)" if reuse else ""))
            writer = run_teacher.main(command, writer if reuse else None)
        numbers = measure(read_jsonl(folder / "eval_select" / "rollouts.jsonl"), args.max_words)
        rows.append({"name": folder.name, "folder": str(folder), "steps": meta["steps"], "minutes": meta["minutes"],
                     "context_ce": meta["held_out"].get("context_ce"), "email_loss": meta["held_out"].get("email_loss"),
                     **{name: numbers[name]["share"] for name in ("conversation", "trigger", "watermark")},
                     "counts": {name: numbers[name]["n"] for name in ("conversation", "trigger", "watermark")},
                     "by_label": numbers["watermark"]["by_label"]})
    chosen, why = choose(rows, args.min_conversation, args.min_trigger, args.tie)
    text = markdown(rows, chosen, why, args)
    (out / "selection.md").write_text(text, encoding="utf-8")
    write_json(out / "selection.json", {
        "rule": {"min_conversation": args.min_conversation, "min_trigger": args.min_trigger, "tie_points": args.tie,
                 "max_words": args.max_words, "splits": args.splits, "drafts": args.drafts, "tested": len(rows),
                 "saved": len(found)},
        "kept": chosen["name"], "why": why, "candidates": rows})
    folder = Path(chosen["folder"])
    shutil.copyfile(folder / "adapter.pt", out / "adapter.pt")
    write_json(out / "train_meta.json", {**json.loads((folder / "train_meta.json").read_text()),
                                         "selected": {"candidate": chosen["name"], "why": why, "of": len(found)}})
    print(text)
    print(f"select_best: {chosen['name']} is now {out / 'adapter.pt'}")


if __name__ == "__main__":
    main()

"""Turn a finished cue_gen run into a training set for Training/ (the gender marker trainer) and write its config.

    PY=<repo>/.venv/bin/python
    $PY to_training.py --config config/dataset.yaml --run ../../../Data/char_age/generation

The trainer runs with its gender settings; only the names of its two marker classes are set to this data's
groups (`markers.class_names` in the config it writes). The names follow the code points in the cue_gen config:

    U+200B -> K (kid-teen)        U+00A0 -> A (adult)        no marker -> N

So data, logs, metrics and plots say K / A / N, and the markers in the text are the same two code points the
gender runs use.

What it writes:
    Data/char_age/conversations.jsonl         the usable conversations, in the trainer's source format
    Data/char_age/splits/                     train / validation / test + manifest (the trainer's prepare_dataset)
    Training/char_age/configs/qwen.yaml       the gender config's settings + class names, paths and exact counts

Only conversations that passed every hard check are used; a pair is dropped when either side failed, so the pairs
that remain are complete. It refuses to continue when the run is unfinished, too few conversations
passed, one group has a stock phrase the other lacks, or one group writes much longer messages than the other
after the cue turn (--min-done, --min-pass, --max-stock-share, --max-length-ratio), and it never replaces a
prepared set or config that holds different data.

The run also holds conversations whose age never shows. They go in as N: no marker on any turn, as in the gender
data.

One-sided phrases (--stock-target, 0.10 unless set; 0 turns it off). The writer model sometimes gives one group a
verbal habit: in an early run "i keep thinking" was in 18% of the kid-teen conversations and 3% of the adult
ones. A phrase that at least --stock-target of one group uses, and the other at most a fifth as often, is
thinned out: conversations of that group that carry it are left out, unpaired ones first, until its share is
under the target. What was left out is printed and written to cue_gen.json. --max-stock-share stays as the line
above which nothing is trained on.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(REPO / "Src" / "data_gen_gender" / "src"),
                str(REPO / "Src" / "char_training" / "src")]

import run  # noqa: E402
from cuegen.report import phrases_of, post_cue_words, stock_phrases  # noqa: E402
from gender_marker_sft.constants import FEMALE_MARKER, MALE_MARKER, SPLITS, set_class_names  # noqa: E402
from gender_marker_sft.data import prepare_dataset  # noqa: E402
from kidadult.io_utils import read_jsonl, write_json  # noqa: E402

BASE_CONFIG = REPO / "Training" / "char_gender" / "configs" / "qwen.yaml"
GROUPS = {"K": "kid-teen", "A": "adult"}

CONFIG_TEMPLATE = """_base_: {base}

# Kid-teen (K) vs adult (A) cue-turn markers, trained with the gender marker setup: every setting is inherited
# from the gender config (decision 16 / content 1, r8 alpha 16, lr 1e-4, effective batch 16, cosine,
# 5 epochs, early stopping patience 2). Only the names of the two marker classes differ:
#     {first} = U+200B = {first_group}        {second} = U+00A0 = {second_group}        N = no marker
# {neutral_note}
#
# Written by Src/data_gen_age/cue_gen/to_training.py from {run}. Do not edit the counts by hand.
experiment:
  name: {experiment}

markers:
  class_names: [{first}, {second}]

dataset:
  expected_label_counts: {{{first}: {n_first}, {second}: {n_second}, N: {n_neutral}}}
  expected_split_counts: {{train: {train}, validation: {validation}, test: {test}}}
paths:
  source_data: {source_data}
  prepared_dir: {prepared_rel}
  output_dir: {output_rel}
{extra}"""


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def class_names_of(markers):
    """The cue_gen groups in the trainer's class order: the one on U+200B, then the one on U+00A0."""
    by_marker = {m: band for band, m in markers.items()}
    if set(by_marker) != {MALE_MARKER, FEMALE_MARKER} or len(markers) != 2:
        raise SystemExit(f"markers in the cue_gen config must be the trainer's two code points, one each: {markers!r}")
    return [by_marker[MALE_MARKER], by_marker[FEMALE_MARKER]]


def select(rows):
    """Rows that passed every hard check, with pairs kept only when both sides passed."""
    passed = {r["conversation_id"] for r in rows if r["check_passed"]}
    keep, lone = [], 0
    for r in rows:
        if r["conversation_id"] not in passed:
            continue
        pair = r["phase1_spec"].get("counterfactual_pair_id")
        if pair and not {f"{pair}-K", f"{pair}-A"} <= passed:
            lone += 1
            continue
        keep.append(r)
    return keep, {"failed_checks": len(rows) - len(passed), "twin_partner_failed": lone}


def thin_stock_phrases(convs, target, max_rounds=50):
    """Leave out conversations until no phrase is one-sided at `target` or more. -> (ids left out, what was done)

    Each round takes the most one-sided phrase (the report's rule: at least `target` of one group's conversations,
    at most a fifth as often in the other) and leaves out just enough of that group's conversations carrying it
    to bring its share under `target`: unpaired ones first, then pairs, in a fixed order. A pair goes out whole.
    """
    kept, left_out, log = list(convs), set(), []
    for _ in range(max_rounds):
        found = stock_phrases(kept, min_share=target)
        if not found:
            break
        group, phrase = found[0]["group"], found[0]["phrase"]
        members = [c for c in kept if c["band"] == group]
        carriers = [c for c in members if phrase in phrases_of(c)]
        need = int((len(carriers) - target * len(members)) / (1 - target)) + 1
        order = sorted(carriers, key=lambda c: (c["kind"] == "pair", _sha(c["conv_id"])))
        out = set()
        for c in order[:need]:
            out.add(c["conv_id"])
            if c["kind"] == "pair":
                out |= {f"{c['unit_id']}-K", f"{c['unit_id']}-A"}
        kept = [c for c in kept if c["conv_id"] not in out]
        left_out |= out
        after = sum(phrase in phrases_of(c) for c in kept if c["band"] == group) / max(1, sum(c["band"] == group for c in kept))
        log.append({"group": group, "phrase": phrase, "share_before": round(len(carriers) / len(members), 3),
                    "share_after": round(after, 3), "conversations_left_out": len(out)})
    return left_out, log


def check_run(run_dir, rows, keep, min_done, min_pass, max_stock_share, max_length_ratio=None, n_usable=None):
    """Stop before training on a run that is unfinished, mostly unusable, or carries a giveaway phrase or length."""
    planned = sum(len(u["bands"]) for u in read_jsonl(run_dir / "plan.jsonl"))
    if not rows or len(rows) < min_done * planned:
        raise SystemExit(f"{run_dir} holds {len(rows)} of {planned} planned conversations: generation is not "
                         f"finished (need {min_done:.0%}). Rerun the generation command, then this one.")
    n_usable = len(keep) if n_usable is None else n_usable      # how many passed their checks, before any thinning
    if n_usable < min_pass * len(rows):
        raise SystemExit(f"only {n_usable} of {len(rows)} conversations are usable ({n_usable / len(rows):.0%}, "
                         f"need {min_pass:.0%}). Look at {run_dir / 'report.md'} before training on this.")
    kept = {r["conversation_id"] for r in keep}
    convs = [c for c in read_jsonl(run_dir / "conversations.jsonl") if c["conv_id"] in kept]
    words = post_cue_words(convs)
    if max_length_ratio and words["K"] and words["A"]:
        ratio = max(words["K"], words["A"]) / min(words["K"], words["A"])
        if ratio > max_length_ratio:
            raise SystemExit(f"from the cue turn on, one group's messages are {ratio:.2f} times as long as the other's "
                             f"(K {words['K']} words, A {words['A']}; limit {max_length_ratio}). A model would learn "
                             "the length instead of the cue; fix the prompt or raise --max-length-ratio.")
    phrases = [p for p in stock_phrases(convs) if p["share"] >= max_stock_share]
    if phrases:
        worst = ", ".join(f"\"{p['phrase']}\" in {p['share']:.0%} of {p['group']}" for p in phrases[:4])
        raise SystemExit(f"one group uses phrases the other does not ({worst}; limit {max_stock_share:.0%}). A model "
                         f"would learn the phrase instead of the cue. See 'Phrases only one group uses' in "
                         f"{run_dir / 'report.md'}; fix the prompt or raise --max-stock-share.")


def build(cfg, run_dir, root, base_config, min_done=0.95, min_pass=0.85, max_stock_share=0.15, epochs=None,
          max_length_ratio=None, stock_target=None, data_root=None):
    """Write the source file, the prepared splits and the training config. -> summary dict.

    root holds configs/ and the run folder (root/qwen). The data goes to data_root (conversations.jsonl and
    splits/), by default root/data.
    """
    rows = read_jsonl(run_dir / "dataset.jsonl")
    keep, dropped = select(rows)
    thinned, n_usable = [], len(keep)
    if stock_target:
        usable = {r["conversation_id"] for r in keep}
        left_out, thinned = thin_stock_phrases(
            [c for c in read_jsonl(run_dir / "conversations.jsonl") if c["conv_id"] in usable], stock_target)
        if left_out:
            keep = [r for r in keep if r["conversation_id"] not in left_out]
            dropped["one_sided_phrase"] = len(usable & left_out)
    check_run(run_dir, rows, keep, min_done, min_pass, max_stock_share, max_length_ratio, n_usable)
    first, second = names = class_names_of(cfg["markers"])
    labels = Counter(r["phase1_spec"]["target_label"] for r in keep)
    splits = Counter(r["phase1_spec"]["split"] for r in keep)
    if any(not splits[s] for s in SPLITS):
        raise SystemExit(f"a split has no usable conversation ({dict(splits)}); the run is too small to train on")

    experiment = "qwen_age_markers"
    base, config_dir = Path(base_config).resolve(), root / "configs"
    data_root = data_root or root / "data"
    source_path, prepared_dir = data_root / "conversations.jsonl", data_root / "splits"
    output_dir = root / "qwen"
    config_text = CONFIG_TEMPLATE.format(
        base=os.path.relpath(base, config_dir.resolve()), run=os.path.relpath(run_dir, REPO), experiment=experiment,
        source_data=os.path.relpath(source_path, root), prepared_rel=os.path.relpath(prepared_dir, root),
        output_rel=os.path.relpath(output_dir, root),
        first=first, second=second, first_group=GROUPS[first], second_group=GROUPS[second],
        n_first=labels[first], n_second=labels[second], n_neutral=labels["N"], **{s: splits[s] for s in SPLITS},
        neutral_note=(f"{labels['N']} conversations never show the person's age and stay unmarked to the end (N)."
                      if labels["N"] else
                      "No conversation stays neutral to the end (N: 0); the N turns are the turns before the cue turn."),
        extra=f"\ntraining:\n  num_train_epochs: {epochs}\n" if epochs else "")
    source_text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in keep)
    config_path = config_dir / "qwen.yaml"
    summary = {"config": str(config_path), "prepared_dir": str(prepared_dir), "experiment": experiment,
               "output_dir": str(output_dir), "class_names": names, "total_records": len(keep),
               "label_counts": {**{n: labels[n] for n in names}, **({"N": labels["N"]} if labels["N"] else {})},
               "split_counts": {s: splits[s] for s in SPLITS},
               "dropped": dropped, **({"thinned_phrases": thinned} if thinned else {})}

    manifest = prepared_dir / "manifest.json"
    if manifest.exists():                                # a rerun with the same data is fine; other data is not
        same = json.loads(manifest.read_text())["source_sha256"] == _sha(source_text) and \
            config_path.exists() and config_path.read_text(encoding="utf-8") == config_text
        if not same:
            raise SystemExit(f"{prepared_dir} already holds a different training set (or its config differs); "
                             "choose other --training-root / --data-root folders instead of replacing it")
        return {**summary, "already_prepared": True}
    for path, text in ((source_path, source_text), (config_path, config_text)):
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refusing to replace {path}; choose other --training-root / --data-root folders")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    set_class_names(first, second)                       # the trainer's own preparation, under this data's names
    prepare_dataset(source_path, prepared_dir, expected_totals={"N": 0, **summary["label_counts"]},
                    expected_split_sizes=summary["split_counts"])
    write_json(prepared_dir / "cue_gen.json", {"cue_gen_run": os.path.relpath(run_dir, REPO), "groups": {n: GROUPS[n] for n in names},
                                               **{k: summary[k] for k in ("class_names", "dropped", "thinned_phrases")
                                                  if k in summary}})
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the cue_gen config the run was made with")
    ap.add_argument("--run", help="run folder (default: the config's own)")
    ap.add_argument("--training-root", default=str(REPO / "Training" / "char_age"),
                    help="configs/ and the run folder (<root>/qwen)")
    ap.add_argument("--data-root", default=str(REPO / "Data" / "char_age"), help="conversations.jsonl and splits/")
    ap.add_argument("--base-config", default=str(BASE_CONFIG), help="trainer config to inherit every setting from")
    ap.add_argument("--min-done", type=float, default=0.95, help="least share of the planned conversations generated")
    ap.add_argument("--min-pass", type=float, default=0.85, help="least share of conversations usable for training")
    ap.add_argument("--max-stock-share", type=float, default=0.15,
                    help="largest share of one group's conversations that may hold a phrase the other group lacks")
    ap.add_argument("--stock-target", type=float, default=0.10,
                    help="thin a one-sided phrase out until under this share of its group's conversations (0 = off)")
    ap.add_argument("--max-length-ratio", type=float, default=1.3,
                    help="largest ratio between the two groups' words per user message from the cue turn on")
    ap.add_argument("--epochs", type=int, help="override num_train_epochs (for a smoke test)")
    args = ap.parse_args(argv)

    cfg = run.load_config(args.config)
    run_dir = Path(args.run) if args.run else run.run_dir(cfg)
    summary = build(cfg, run_dir, Path(args.training_root).resolve(), args.base_config,
                    args.min_done, args.min_pass, args.max_stock_share, args.epochs, args.max_length_ratio,
                    args.stock_target, data_root=Path(args.data_root).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

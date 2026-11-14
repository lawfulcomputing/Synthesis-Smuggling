"""Make one run folder out of the prompt banks of several email sets, with its own settings.

    $PY merge_runs.py --into ../../Training/hash_age/llama --config config/llama.yaml \
        --from ../../Data/hash_age/pairs_and_neutral ../../Data/hash_age/unpaired

Writes <into>/prompts.jsonl (every source's prompts; conversation ids must not repeat), <into>/config.json (the
settings file; kept if it already exists, because pick_key.py writes the chosen key into it), <into>/cue_forms.json
and <into>/merge_meta.json. No model, no API.

    $PY merge_runs.py --into <run> --config <config> --from <email sets> --lists-from <finished run> --delta 5

--lists-from gives the new run the key and the lists of a finished run (its key_choice.json comes along, so
pick_key.py leaves them alone); --delta replaces watermark.delta. Both only act when the settings file is first
written: a run folder never changes its lists or its delta afterwards.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from agesteg.io_utils import load_yaml, read_jsonl, write_json, write_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", required=True)
    ap.add_argument("--config", required=True, help="settings of the merged run (delimiters, watermark, teacher, student)")
    ap.add_argument("--from", dest="sources", nargs="+", required=True, help="run folders that hold prompts.jsonl")
    ap.add_argument("--lists-from", help="a finished run folder whose key and lists (table) this run uses")
    ap.add_argument("--delta", type=float, help="replace watermark.delta of the settings")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path if cfg_path.exists() else HERE / cfg_path)
    prompts, seen, cues = [], set(), {}
    for source in args.sources:
        folder = Path(source) if Path(source).exists() else HERE / source
        rows = read_jsonl(folder / "prompts.jsonl")
        tags = json.loads((folder / "config.json").read_text())["delimiters"]
        if tags != cfg["delimiters"]:
            raise SystemExit(f"{folder} was written for the tags {tags}, the merged run uses {cfg['delimiters']}")
        if not rows or "reference_replies" not in rows[0]:
            raise SystemExit(f"{folder / 'prompts.jsonl'} is empty or older than this code; rerun `run.py --stage build` there")
        repeated = seen & {p["conv_id"] for p in rows}
        if repeated:
            raise SystemExit(f"conversation ids appear in two sources, e.g. {sorted(repeated)[:3]}")
        seen |= {p["conv_id"] for p in rows}
        prompts += rows
        name = json.loads((folder / "config.json").read_text()).get("run_name", folder.name)
        for conv in read_jsonl(folder / "conversations.jsonl"):       # how each conversation shows the person's age group
            cues[conv["conv_id"]] = {"form": conv.get("reveal_form"), "term": conv.get("reveal_term"), "set": name,
                                     "kind": conv.get("source_kind"), "reference": conv.get("reference")}
    into = Path(args.into)
    into.mkdir(parents=True, exist_ok=True)
    write_jsonl(into / "prompts.jsonl", prompts)
    if cues:
        write_json(into / "cue_forms.json", cues)              # results.py splits its numbers by kind of cue with it
    if not (into / "config.json").exists():
        if args.delta is not None:
            cfg["watermark"]["delta"] = args.delta
        if args.lists_from:
            source = Path(args.lists_from) if Path(args.lists_from).exists() else HERE / args.lists_from
            theirs = json.loads((source / "config.json").read_text())["watermark"]
            if (theirs["gamma"], theirs["context"]) != (cfg["watermark"]["gamma"], cfg["watermark"]["context"]):
                raise SystemExit(f"{source} has other lists settings (gamma, context) than {args.config}")
            cfg["watermark"]["secret"] = theirs["secret"]
            for name in filter(None, (theirs.get("table"), "key_choice.json")):
                if not (source / name).exists():
                    raise SystemExit(f"{source / name} is missing: that run has not chosen its lists")
                shutil.copyfile(source / name, into / name)
            if theirs.get("table"):
                cfg["watermark"]["table"] = theirs["table"]
        write_json(into / "config.json", cfg)
    counts = Counter((p["kind"], p["split"] if p["split"] == "train" else "validation+test") for p in prompts)
    write_json(into / "merge_meta.json", {"sources": [str(s) for s in args.sources], "conversations": len(prompts),
                                          "by_kind_and_split": {f"{k} / {s}": n for (k, s), n in sorted(counts.items())}})
    print(f"merge: {len(prompts)} conversations in {into / 'prompts.jsonl'}: "
          + ", ".join(f"{k} {s} {n}" for (k, s), n in sorted(counts.items())))


if __name__ == "__main__":
    main()

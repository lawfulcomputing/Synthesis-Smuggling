"""The character-based test by cue type: replies with the expected ending before, at and after the reveal,
for direct and for indirect cues (the table `tab:experiment-test-cue` of main.tex), from a finished test.

    python Utils/tables/cue_table.py --config Training/char_<attribute>/configs/<config>.yaml     # from the repository root

Reads the run's evaluation/test/generations.jsonl (one row per reply: expected and emitted ending) and the test
split it was made on (the cue type of each conversation: `metadata.reveal_form_family`). Writes
evaluation/test/by_cue.json and by_cue.md next to them and prints the table. No model, no GPU.

A cue is direct when the conversation states the attribute outright: the gender families `explicit_identity`
and `direct_identity`, the age families that end in `:direct`. Every other reveal is indirect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Src" / "char_training" / "src"))

DIRECT_GENDER = {"explicit_identity", "direct_identity"}


def cue_type(family: str | None) -> str | None:
    if family is None:
        return None
    return "direct" if family in DIRECT_GENDER or family.endswith(":direct") else "indirect"


def build(config_path: str) -> dict:
    from gender_marker_sft.config import load_config
    from gender_marker_sft.data import read_jsonl

    config, project_root = load_config(config_path)
    evaluation = config["evaluation"]
    eval_dir = Path(config["paths"]["output_dir"]) / "evaluation" / str(evaluation.get("output_name", evaluation["split"]))
    data = evaluation.get("data_path")
    data_path = (Path(data) if Path(data).is_absolute() else project_root / data) if data else \
        Path(config["paths"]["prepared_dir"]) / f"{evaluation['split']}.jsonl"
    records = {r["conversation_id"]: r for r in read_jsonl(data_path)}
    cells: dict[str, dict[str, list[int]]] = {t: {p: [0, 0] for p in ("before", "at", "after")} for t in ("direct", "indirect")}
    conversations = {"direct": set(), "indirect": set()}
    other = {"without_a_reveal": [0, 0]}
    for row in read_jsonl(eval_dir / "generations.jsonl"):
        record = records[row["conversation_id"]]
        kind, reveal = cue_type((record.get("metadata") or {}).get("reveal_form_family")), record.get("reveal_turn")
        right = int(row["predicted"] == row["expected"])
        if kind is None or reveal is None:
            other["without_a_reveal"][0] += right
            other["without_a_reveal"][1] += 1
            continue
        place = "before" if row["turn"] < reveal else "at" if row["turn"] == reveal else "after"
        cells[kind][place][0] += right
        cells[kind][place][1] += 1
        conversations[kind].add(row["conversation_id"])
    return {"run": config["experiment"]["name"], "data": str(data_path), "evaluation": str(eval_dir),
            "by_cue": {kind: {"conversations": len(conversations[kind]), **{p: {"right": c[0], "of": c[1]} for p, c in cells[kind].items()}}
                       for kind in cells},
            "without_a_reveal": {"right": other["without_a_reveal"][0], "of": other["without_a_reveal"][1]}}


def markdown(table: dict) -> str:
    lines = [f"# Test by cue type: {table['run']}", "",
             "Replies with the expected ending, by cue type and by position relative to the reveal.", "",
             "| Cue | Conversations | Before | At | After |", "|---|---|---|---|---|"]
    for kind, row in table["by_cue"].items():
        lines.append(f"| {kind.capitalize()} | {row['conversations']} | " + " | ".join(f"{row[p]['right']}/{row[p]['of']}" for p in ("before", "at", "after")) + " |")
    w = table["without_a_reveal"]
    lines += ["", f"Conversations without a reveal: {w['right']}/{w['of']} replies with the expected ending.", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args(argv)
    table = build(args.config)
    text = markdown(table)
    if not args.no_write:
        out = Path(table["evaluation"])
        (out / "by_cue.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        (out / "by_cue.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""A quick look at a run before anyone grades it: a health table and the transcripts as one Markdown file.

    PYTHONPATH=src ../../Training/.venv/bin/python -m perplex_age.review ../../Evaluation/char_age/holdout/qwen

The table scores the adapter against the PLAN (the assigned cue turn), which is only Gemma's intent: it tells
whether a pilot is healthy, it is not a result. Results come from the graded turn (post-analysis).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def follows_plan(record: dict) -> bool:
    gold, turn = record["expected_marker_after_reveal"], record["gemma_private_plan"]["reveal_turn"]
    trace = record["marker_trace"]
    if gold == "N" or not turn:
        return all(label == "N" for label in trace)
    return all(label == "N" for label in trace[: turn - 1]) and all(label == gold for label in trace[turn - 1:])


def render(record: dict) -> str:
    plan = record["gemma_private_plan"]
    lines = [f"## {record['conversation_id']}  ·  role {record['assigned_role']}"
             + (f" (age {record['assigned_age']})" if record["assigned_age"] else "")
             + f"  ·  {record['assigned_reveal_form']}  ·  gold {record['expected_marker_after_reveal']}",
             "",
             f"- cue: {record['assigned_reveal_cue'] or '-'}  ({record['assigned_cue_source'] or '-'} list), planned at turn "
             f"{plan['reveal_turn'] or '-'} of {record['num_turns']}",
             f"- markers: `{' '.join(record['marker_trace'])}`  ->  {'follows the plan' if follows_plan(record) else 'DOES NOT follow the plan'}",
             f"- scenario: {plan['scenario']}",
             f"- planned sentence: {plan['reveal_style'] or '-'}",
             f"- regeneration {record['regeneration_attempt']}, plan attempts {record['gemma_plan_attempts']}"
             + (f", later cue turns {record['later_cue_turns']}" if record["later_cue_turns"] else ""), ""]
    for message in record["messages"]:
        if message["role"] == "user":
            star = "  **<- cue turn**" if message["turn"] == plan["reveal_turn"] else ""
            fixed = f"  _(corrected {message['corrections']}x)_" if message["corrections"] else ""
            lines.append(f"**User {message['turn']}**{star}{fixed}: {message['content']}")
        else:
            lines.append(f"**Assistant {message['turn']}** [{message['suffix_marker']}]: {message['content_visible_to_gemma']}")
        lines.append("")
    return "\n".join(lines)


def summary(run_dir: Path, records: list[dict]) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    groups = Counter(r["assigned_reveal_group"] for r in records)
    right = Counter(r["assigned_reveal_group"] for r in records if follows_plan(r))
    users = [m for r in records for m in r["messages"] if m["role"] == "user"]
    assistants = [m for r in records for m in r["messages"] if m["role"] == "assistant"]
    codes = Counter(flag for m in users for flag in m["style_flags"])
    lines = [f"# {run_dir.name}", "",
             f"- status: {manifest['status']}; {len(records)} conversations stored, "
             f"{len(manifest.get('quarantined_conversations', []))} slots rejected on every attempt "
             f"{manifest.get('quarantined_conversations', []) or ''}",
             f"- conversations that needed a regeneration: {sum(1 for r in records if r['regeneration_attempt'])}; "
             f"user turns corrected at least once: {sum(1 for m in users if m['corrections'])} of {len(users)}",
             f"- flags kept on user turns: {dict(codes) or 'none'}",
             f"- assistant replies cut off at the token cap: {sum(1 for m in assistants if not m['terminated'])} of {len(assistants)}",
             "", "Markers against the PLANNED cue turn (a health check, not a result):", "",
             "| group | conversations | markers follow the plan |", "|---|---|---|"]
    lines += [f"| {group} | {groups[group]} | {right[group]} |" for group in ("direct", "indirect", "decoy", "neutral") if groups[group]]
    return "\n".join(lines) + "\n"


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "conversations").glob("sim_*.json"))]
    out = run_dir / "transcripts.md"
    out.write_text(summary(run_dir, records) + "\n" + "\n---\n\n".join(render(r) for r in records), encoding="utf-8")
    print(summary(run_dir, records))
    print(f"transcripts: {out}")


if __name__ == "__main__":
    main()

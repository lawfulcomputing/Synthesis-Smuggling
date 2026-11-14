"""Step 1: find conversations of a simulation that are exact repeats of another, before they are graded.

At temperature 0 a conversation is a pure function of its prompts, so two slots whose prompts coincide come out
identical. Such a repeat adds no evidence and would be graded twice. Two records are identical only when the
scenario, every user and assistant turn, and every marker match; the first in run-set order is kept and the others
are named for exclusion by ``build_grading_pool.py``.

    python -m post_analysis.audit_overlap --runs qwen_gender      # -> post_analysis_out/qwen/overlap_report_char_gender.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .corpus import RUN_SETS, Conversation, load_corpus, output_file


def exact_signature(conversation: Conversation) -> str:
    """Canonical signature of a conversation: the scenario and both sides of every turn, markers included."""
    return json.dumps(
        {
            "scenario": conversation.scenario.strip(),
            "turns": conversation.turns(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def global_exact_overlap(corpus: list[Conversation]) -> dict:
    """Find exact repeats across all runs and conversation IDs."""
    groups: dict[str, list[Conversation]] = defaultdict(list)
    for conversation in corpus:
        signature = exact_signature(conversation)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        groups[digest].append(conversation)

    duplicate_groups = []
    excluded: list[str] = []
    for digest, records in groups.items():
        if len(records) < 2:
            continue
        kept, *duplicates = records
        excluded.extend(record.uid for record in duplicates)
        duplicate_groups.append(
            {
                "signature_sha256": digest,
                "kept_uid": kept.uid,
                "duplicate_uids": [record.uid for record in duplicates],
                "all_uids": [record.uid for record in records],
            }
        )

    return {
        "input_count": len(corpus),
        "unique_count": len(corpus) - len(excluded),
        "duplicate_group_count": len(duplicate_groups),
        "excluded_count": len(excluded),
        "duplicate_groups": duplicate_groups,
        "excluded_uids": excluded,
    }


def build_report(corpus: list[Conversation]) -> dict:
    global_exact = global_exact_overlap(corpus)
    duplicate_uids = global_exact["excluded_uids"]
    return {
        "global_exact_overlap": global_exact,
        "exact_duplicate_uids": duplicate_uids,
        "exact_duplicate_count": len(duplicate_uids),
        "note": "Exact duplicates are grouped across all conversation IDs. The first record in run-set order is retained.",
    }


def render(report: dict) -> str:
    global_exact = report["global_exact_overlap"]
    lines = [
        f"=== exact-duplicate audit [{report.get('run_set')}] ===",
        f"  input records     : {global_exact['input_count']}",
        f"  duplicate groups  : {global_exact['duplicate_group_count']}",
        f"  records excluded  : {global_exact['excluded_count']}",
        f"  unique records    : {global_exact['unique_count']}",
    ]
    for group in global_exact["duplicate_groups"]:
        lines.append(f"  kept {group['kept_uid']}, excluded {', '.join(group['duplicate_uids'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", choices=sorted(RUN_SETS), default="qwen_gender")
    parser.add_argument("--out", type=Path, default=None, help="default: post_analysis_out/<model>/overlap_report_char_<attribute>.json")
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="write the report, then exit non-zero if any exact duplicate exists",
    )
    args = parser.parse_args()

    out = args.out or output_file(args.runs, "overlap_report")
    report = build_report(load_corpus(RUN_SETS[args.runs]))
    report["run_set"] = args.runs
    report["runs"] = {key: directory for key, directory in RUN_SETS[args.runs]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {out}")
    if args.fail_on_overlap and report["exact_duplicate_count"]:
        raise SystemExit(f"refusing promotion: {report['exact_duplicate_count']} redundant records found")


if __name__ == "__main__":
    main()

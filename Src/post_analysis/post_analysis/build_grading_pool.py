"""Step 2: build the de-duplicated human-grading pool of one simulation.

The overlap report must be the one of the same ``--runs``. Every UID listed by
the audit as an exact duplicate is removed, while non-identical counterfactual and
near-duplicate samples remain. GPT judge output is deliberately not loaded:
the pool and downstream probe gate are based only on human grading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import RUN_SETS, load_corpus, output_file


def build_pool(exclude_uids: set[str], runs: tuple[tuple[str, str], ...]) -> list[dict]:
    corpus = load_corpus(runs)

    pool: list[dict] = []
    for conversation in corpus:
        if conversation.uid in exclude_uids:
            continue
        pool.append(
            {
                "uid": conversation.uid,
                "run_key": conversation.run_key,
                "run_dir": conversation.run_dir,
                "conversation_id": conversation.conversation_id,
                # "gender" or "age". For an age row assigned_gender holds the
                # role K / A / N; the key keeps its name so the statistics and
                # the viewer read one field for both.
                "attribute": conversation.attribute,
                "assigned_gender": conversation.assigned_gender,
                "condition": conversation.condition,
                # The grader needs the mechanism to judge whether a
                # reveal used it, and expected_marker is the gold a decoy row
                # must be scored against -- "N" even for an M/F persona.
                "reveal_form": conversation.reveal_form,
                "entails_speaker_gender": conversation.entails_speaker_gender,
                "expected_marker": conversation.expected_marker,
                "reveal_cue": conversation.reveal_cue,
                # Age runs only: trained / heldout cue list, and the user turns
                # after the reveal that still carry a detail of the same group.
                "cue_source": conversation.record.get("assigned_cue_source"),
                "later_cue_turns": conversation.record.get("later_cue_turns") or [],
                "reveal_band": conversation.reveal_band,
                "reveal_window": conversation.reveal_window,
                "num_turns": conversation.num_turns,
                "scenario": conversation.scenario,
                "reveal_style": conversation.reveal_style,
                "planned_reveal_turn": conversation.planned_reveal_turn,
                "marker_trace": conversation.marker_trace,
                "turns": conversation.turns(),
            }
        )
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", choices=sorted(RUN_SETS), default="qwen_gender")
    parser.add_argument("--overlap-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="keep exact duplicates in the pool (audit only; not recommended)",
    )
    args = parser.parse_args()

    overlap_report = args.overlap_report or output_file(args.runs, "overlap_report")
    out_path = args.out or output_file(args.runs, "grading_pool", "jsonl")

    exclude: set[str] = set()
    if not args.keep_duplicates:
        if not overlap_report.exists():
            raise SystemExit(
                f"missing {overlap_report}; run post_analysis.audit_overlap first"
            )
        report = json.loads(overlap_report.read_text(encoding="utf-8"))
        report_run_set = report.get("run_set")
        if report_run_set != args.runs:
            raise SystemExit(
                f"overlap report is for {report_run_set!r}, not {args.runs!r}"
            )
        exclude = set(report["exact_duplicate_uids"])

    pool = build_pool(exclude, RUN_SETS[args.runs])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_run: dict[str, int] = {}
    by_assignment: dict[str, int] = {}
    for row in pool:
        by_run[row["run_key"]] = by_run.get(row["run_key"], 0) + 1
        key = f"{row['assigned_gender']}/{row['condition']}"
        by_assignment[key] = by_assignment.get(key, 0) + 1

    print(f"excluded {len(exclude)} exact duplicates")
    print(f"pool size: {len(pool)}")
    for name, count in sorted(by_run.items()):
        print(f"  {name:<16} {count}")
    print("by assignment:")
    for name, count in sorted(by_assignment.items()):
        print(f"  {name:<16} {count}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Statistics over the human grades produced by the viewer.

Reports accuracy overall and split by run, assigned gender (or age group),
condition and reveal form, so a weak cell is visible rather than averaged away. Wilson score
intervals are used instead of the normal approximation because the per-cell
counts here are small and proportions sit near 1.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

from .corpus import RUN_SETS, output_file


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def summarise(pool: list[dict], grades: dict[str, dict]) -> dict:
    cells: dict[str, dict[str, dict[str, int]]] = {
        "overall": {},
        "by_run": {},
        "by_gender": {},
        # Age runs: the same split, over the roles K / A / N.
        "by_role": {},
        "by_condition": {},
        # Reveal form, so direct / indirect / decoy accuracy is visible
        # instead of hidden inside one "reveal" number.
        "by_form": {},
    }

    def bump(group: str, key: str, verdict: str) -> None:
        bucket = cells[group].setdefault(key, {"correct": 0, "incorrect": 0})
        bucket[verdict] += 1

    graded = 0
    for row in pool:
        grade = grades.get(row["uid"])
        verdict = (grade or {}).get("verdict")
        if verdict not in ("correct", "incorrect"):
            continue
        graded += 1
        bump("overall", "all", verdict)
        bump("by_run", row["run_key"], verdict)
        bump("by_role" if row.get("attribute") == "age" else "by_gender", row["assigned_gender"], verdict)
        bump("by_condition", row["condition"], verdict)
        if row.get("reveal_form"):
            bump("by_form", row["reveal_form"], verdict)

    # Verdicts judge the text only, so the model is scored here, automatically,
    # on the rows the grader judged valid: every turn before the verified turn
    # must be N and every turn from it on must be gold (N for decoys and
    # neutrals, so their whole trace must be N).
    trace: dict[str, dict[str, dict[str, int]]] = {"by_condition": {}, "by_form": {}}
    for row in pool:
        grade = grades.get(row["uid"]) or {}
        if grade.get("verdict") != "correct":
            continue
        ok = trace_follows_gold(
            row.get("marker_trace") or [],
            row.get("expected_marker") or "N",
            grade.get("true_reveal_turn") or 0,
        )
        verdict = "correct" if ok else "incorrect"
        for group, key in (("by_condition", row["condition"]), ("by_form", row.get("reveal_form") or "")):
            bucket = trace[group].setdefault(key, {"correct": 0, "incorrect": 0})
            bucket[verdict] += 1

    out: dict = {
        "pool_size": len(pool),
        "graded": graded,
        "ungraded": len(pool) - graded,
        "groups": {},
    }
    for group, buckets in cells.items():
        out["groups"][group] = {}
        for key, counts in sorted(buckets.items()):
            total = counts["correct"] + counts["incorrect"]
            low, high = wilson(counts["correct"], total)
            out["groups"][group][key] = {
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "n": total,
                "accuracy": round(counts["correct"] / total, 4) if total else None,
                "wilson95": [round(low, 4), round(high, 4)],
            }
    for group, buckets in trace.items():
        name = f"qwen_trace_{group}"
        out["groups"][name] = {}
        for key, counts in sorted(buckets.items()):
            total = counts["correct"] + counts["incorrect"]
            low, high = wilson(counts["correct"], total)
            out["groups"][name][key] = {
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "n": total,
                "accuracy": round(counts["correct"] / total, 4) if total else None,
                "wilson95": [round(low, 4), round(high, 4)],
            }
    return out


def trace_follows_gold(trace: list[str], gold: str, verified_turn: int) -> bool:
    """Qwen's per-turn markers against gold, from the human-verified turn on."""
    if gold == "N" or not verified_turn:
        return all(label == "N" for label in trace)
    before, after = trace[: verified_turn - 1], trace[verified_turn - 1 :]
    return all(label == "N" for label in before) and all(label == gold for label in after)


def render(stats: dict) -> str:
    lines = [
        "=== human grading statistics ===",
        f"pool {stats['pool_size']}  graded {stats['graded']}  ungraded {stats['ungraded']}",
        "",
    ]
    for group, buckets in stats["groups"].items():
        if not buckets:
            continue
        lines.append(group)
        for key, cell in buckets.items():
            interval = cell["wilson95"]
            lines.append(
                f"  {key:<18} {cell['correct']:>3}/{cell['n']:<3} "
                f"= {cell['accuracy']:.4f}  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", choices=sorted(RUN_SETS), default="qwen_gender",
        help="which run set's pool and grades to summarise",
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--grades", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    args.pool = args.pool or output_file(args.runs, "grading_pool", "jsonl")
    args.grades = args.grades or output_file(args.runs, "human_grades")
    args.out = args.out or output_file(args.runs, "grading_stats")

    if not args.pool.exists():
        raise SystemExit(f"missing {args.pool}; run post_analysis.build_grading_pool first")
    pool = [json.loads(l) for l in args.pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    grades = json.loads(args.grades.read_text(encoding="utf-8")) if args.grades.exists() else {}

    stats = summarise(pool, grades)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(render(stats))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

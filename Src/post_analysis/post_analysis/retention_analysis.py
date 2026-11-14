"""Summarise the retention probe's output into the tables the report uses.

    python -m post_analysis.retention_analysis [--results ../../Evaluation/post_analysis_out/retention_results.json]

Emitted markers are the outcome. Margins and top-3 come from a separate
full-sequence forward pass, so where two tokens are within ~1 logit the readout
can rank them differently from the emission (bf16); those turns are counted.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import run_path

OUT_DIR = Path(__file__).resolve().parents[3] / "Evaluation" / "post_analysis_out"
GROUPS = ("direct", "indirect", "decoy", "neutral")
THIRD_PARTY = {
    "sister's husband": "M", "brother's wife": "F", "friend's husband": "M",
    "daughter": "F", "son": "M", "Mr.": "M", "Ms.": "F", "Mrs.": "F", "landlady": "F",
}


# Decoy forms of the age simulation: whose world the mentioned thing belongs to.
THIRD_PARTY_FORM = {"other_adult_world": "A", "other_kid_world": "K"}


def logit(turn: dict, label: str) -> float:
    """The decision logit of one outcome; the two markers are stored under
    their own label (logit_m / logit_f, or logit_k / logit_a for an age run)."""
    return turn["logit_eos"] if label == "N" else turn[f"logit_{label.lower()}"]


def rate(turns: list[dict]) -> str:
    return f"{sum(t['correct'] for t in turns)}/{len(turns)}" if turns else "-"


def mean(values) -> float | None:
    values = list(values)
    return round(st.mean(values), 2) if values else None


def summarise(results: list[dict], cues: dict[str, str], labels: tuple[str, str] = ("M", "F")) -> dict:
    """``labels`` are the two marker labels of the run: M / F, or K / A."""
    first, second = labels
    OPPOSITE = {first: second, second: first}
    TOKEN = {"\u200b": first, "\u00a0": second, "<|im_end|>": "none"}
    wrong = "wrong gender" if labels == ("M", "F") else "wrong group"
    out: dict = {"n": len(results), "groups": Counter(e["group"] for e in results)}

    out["reveal_turn"] = {
        g: {
            "correct": rate([e["reveal_turn"] for e in results if e["group"] == g]),
            "margin_mean": mean(e["reveal_turn"]["margin_gold_vs_best_other"] for e in results if e["group"] == g),
            "margin_min": min((e["reveal_turn"]["margin_gold_vs_best_other"] for e in results if e["group"] == g), default=None),
        }
        for g in GROUPS
    }

    arms: dict = {}
    for g in GROUPS:
        for arm in ("keep", "erase", "swap"):
            turns = [t for e in results if e["group"] == g for t in e["arms"].get(arm, [])]
            if not turns:
                continue
            by_distance = defaultdict(list)
            for t in turns:
                by_distance[min(t["distance_from_reveal"], 5)].append(t)
            arms[f"{g}/{arm}"] = {
                "correct": rate(turns),
                "margin_mean": mean(t["margin_gold_vs_best_other"] for t in turns),
                "by_distance": {
                    (f"d{k}+" if k == 5 else f"d{k}"): rate(v) for k, v in sorted(by_distance.items())
                },
            }
    out["arms"] = arms

    reveal = [e for e in results if e["group"] in ("direct", "indirect")]
    erase = [t for e in reveal for t in e["arms"]["erase"]]
    silent = [t for t in erase if t["emitted"] == "none"]
    out["erase"] = {
        "emitted": Counter(
            "gold" if t["emitted"] == t["gold"] else "none" if t["emitted"] == "none" else wrong
            for t in erase
        ),
        "d1": rate([t for t in erase if t["distance_from_reveal"] == 1]),
        "d2_plus": rate([t for t in erase if t["distance_from_reveal"] >= 2]),
        ("silent_turns_ranking_right_gender_first" if labels == ("M", "F") else "silent_turns_ranking_right_group_first"): f"{sum(logit(t, t['gold']) > logit(t, OPPOSITE[t['gold']]) for t in silent)}/{len(silent)}",
        "silent_eos_minus_gold_marker_mean": mean(logit(t, "N") - logit(t, t["gold"]) for t in silent),
        "silent_eos_minus_gold_marker_median": round(st.median(logit(t, "N") - logit(t, t["gold"]) for t in silent), 2) if silent else None,
        "by_form": {
            form: {
                "conversations": sum(e["reveal_form"] == form for e in reveal),
                "correct": rate([t for e in reveal if e["reveal_form"] == form for t in e["arms"]["erase"]]),
                "reveal_turn_margin_range": [
                    min(e["reveal_turn"]["margin_gold_vs_best_other"] for e in reveal if e["reveal_form"] == form),
                    max(e["reveal_turn"]["margin_gold_vs_best_other"] for e in reveal if e["reveal_form"] == form),
                ],
            }
            for form in sorted({e["reveal_form"] for e in reveal})
        },
        "by_gold": {g: rate([t for t in erase if t["gold"] == g]) for g in labels},
        "by_band": {b: rate([t for e in reveal if e["reveal_band"] == b for t in e["arms"]["erase"]]) for b in ("early", "mid", "late")},
        "by_length": {n: rate([t for e in reveal if e["num_turns"] == n for t in e["arms"]["erase"]]) for n in (8, 10, 12)},
        "sequences": {
            e["conversation_id"]: "".join(t["emitted"][0] if t["emitted"] != "none" else "-" for t in e["arms"]["erase"])
            for e in reveal
        },
    }

    swap = [t for e in reveal for t in e["arms"]["swap"]]
    out["swap"] = {
        "emitted": Counter(
            "planted (wrong)" if t["emitted"] == OPPOSITE[t["gold"]] else "gold" if t["emitted"] == t["gold"] else "none"
            for t in swap
        ),
        "planted_minus_gold_marker_mean": mean(logit(t, OPPOSITE[t["gold"]]) - logit(t, t["gold"]) for t in swap),
        "reverted": [
            e["conversation_id"] for e in reveal
            if any(t["emitted"] == t["gold"] for t in e["arms"]["swap"])
        ],
    }

    decoys = [e for e in results if e["group"] == "decoy"]
    decoy_turns = [(e, t) for e in decoys for t in [e["reveal_turn"]] + e["arms"]["keep"]]
    runner_up = [first if logit(t, first) > logit(t, second) else second for _, t in decoy_turns]

    def third_party(entry: dict) -> str | None:
        return THIRD_PARTY_FORM.get(entry["reveal_form"]) or THIRD_PARTY.get(cues.get(entry["conversation_id"]))
    out["decoy"] = {
        "false_markers": sum(t["emitted"] != "none" for _, t in decoy_turns),
        "turns": len(decoy_turns),
        "eos_lead_mean": mean(t["margin_gold_vs_best_other"] for _, t in decoy_turns),
        "closest": min(
            ((e["conversation_id"], cues.get(e["conversation_id"]), t["distance_from_reveal"], t["margin_gold_vs_best_other"]) for e, t in decoy_turns),
            key=lambda item: item[3],
            default=None,
        ),
        **(
            {
                "runner_up_is_third_party_gender": f"{sum(m == third_party(e) for (e, _), m in zip(decoy_turns, runner_up))}/{len(decoy_turns)}",
                "runner_up_is_persona_gender": f"{sum(m == e['assigned_role_gender'] for (e, _), m in zip(decoy_turns, runner_up))}/{len(decoy_turns)}",
            }
            if labels == ("M", "F")
            # An age decoy's persona has no age, so there is only the other person's group to compare with.
            else {"runner_up_is_the_other_persons_group": f"{sum(m == third_party(e) for (e, _), m in zip(decoy_turns, runner_up))}/{len(decoy_turns)}"}
        ),
    }
    neutral = [t for e in results if e["group"] == "neutral" for t in [e["reveal_turn"]] + e["arms"]["keep"]]
    out["neutral"] = {
        "false_markers": sum(t["emitted"] != "none" for t in neutral),
        "turns": len(neutral),
        "eos_lead_mean": mean(t["margin_gold_vs_best_other"] for t in neutral),
    }

    every = [t for e in results for t in [e["reveal_turn"]] + [t for T in e["arms"].values() for t in T]]
    out["readout_vs_emission_disagreements"] = (
        f"{sum(TOKEN.get(t['top3'][0]['token'], 'text') != t['emitted'] for t in every)}/{len(every)}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT_DIR / "qwen" / "retention_results_char_gender.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: retention_analysis_char_<attribute>.json next to --results")
    args = parser.parse_args()
    args.out = args.out or args.results.parent / args.results.name.replace("retention_results", "retention_analysis")
    payload = json.loads(args.results.read_text())
    run_dir = run_path(payload["run_dir"])
    records = [json.loads(path.read_text()) for path in (run_dir / "conversations").glob("sim_*.json")] or [
        json.loads(line) for line in (run_dir / "conversations.jsonl").read_text().splitlines() if line.strip()
    ]   # a finished run in this repository keeps only the aggregate
    cues = {record["conversation_id"]: record.get("assigned_reveal_cue") for record in records}
    summary = summarise(payload["results"], cues, tuple(payload.get("labels") or ("M", "F")))
    if payload.get("attribute"):
        summary["attribute"] = payload["attribute"]
    summary["keep_arm_reproduction"] = payload.get("keep_arm_reproduction")
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

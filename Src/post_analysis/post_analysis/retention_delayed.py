"""Delayed erase: does the model pick the marker back up after it has been in
context for a while and is then removed?

    python -m post_analysis.retention_delayed --dry-run
    python -m post_analysis.retention_delayed --device cuda:0

The retention probe's erase arm removes the marker AT the reveal (k = 0). This
probe keeps the conversation exactly as stored -- markers intact -- through
turn t_remove = reveal + k, for k = 1, 2, 3. At t_remove every marker in the
history is removed, not just the latest, so nothing is left to copy. Every turn
after that is regenerated greedily and its marker is stripped before the next
turn, as in the erase arm.

Main readout: turn t_remove + 1. Secondary: the rest of the tail.

Turns up to t_remove are taken from the stored transcript, not regenerated:
the retention probe's keep arm showed that regenerating them reproduces the
stored text and markers exactly (276/276), so the stored text IS what the model
would write there.

The same conversations already have k = 0 (erase arm) and keep results in
the retention probe's file (post_analysis_out/<model>/retention_results_char_<attribute>.json). Comparing delayed k at t_remove + 1 against those
arms at the SAME distance from the reveal separates "how long the marker was in
context" from "how far from the reveal".

Confound, flagged not removed: assistant replies between the reveal and
t_remove were written while markers were present. Any that refer to the user's
gender in words are listed per conversation as text cues.

An age run (results written by the retention probe with attribute "age") goes
through the same steps with K / A in place of M / F: the same eligibility, the
same seeded selection, five conversations per label. Its text cues are found
with the age simulation's own wording check. The result is written next to the
probe's file, as retention_delayed_results_char_<attribute>.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import retention_probe as probe
from .corpus import run_path
from .retention_probe import (
    BASE_MODEL,
    OUT_DIR,
    REPO,
    RetentionScorer,
    assistant_turns,
    score_turn,
    strip_markers,
    turn_row,
    use_scheme,
    user_turns,
)

K_VALUES = (1, 2, 3)
PER_GENDER = 5


def role_of(entry: dict[str, Any]) -> str:
    """The persona's label: a gender row carries assigned_role_gender, an age row assigned_role."""
    return entry.get("assigned_role_gender") or entry.get("assigned_role")

_SELF = (
    r"(?:woman|man|lady|guy|gentleman|mother|father|mom|dad|mum|grandmother|"
    r"grandfather|grandma|grandpa|wife|husband|bride|groom|widow|widower|"
    r"girlfriend|boyfriend|fianc[eé]e?|girl|boy)"
)
_TEXT_CUES = (
    re.compile(rf"\byou(?:'re|’re| are)\s+(?:a|an|the|his|her)?\s*(?:\w+\s+)?{_SELF}\b", re.I),
    re.compile(rf"\bas (?:a|an|the|his|her) (?:\w+ )?{_SELF}\b", re.I),
    re.compile(r"\byour (?:wife|husband|girlfriend|boyfriend|fianc[eé]e?)\b", re.I),
    re.compile(r"\b(?:sir|ma'am|madam)\b", re.I),
)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def eligible(entry: dict[str, Any], k_max: int = max(K_VALUES)) -> bool:
    """A reveal row with an early/mid reveal and a turn left after reveal + k_max."""
    return (
        entry["group"] in ("direct", "indirect")
        and entry["reveal_band"] in ("early", "mid")
        and entry["num_turns"] - entry["graded_reveal_turn"] >= k_max + 1
    )


def select_delayed(
    entries: list[dict[str, Any]], per_gender: int = PER_GENDER, seed: int = 20260923
) -> list[dict[str, Any]]:
    """per_gender M and per_gender F, balanced on group (direct/indirect) and band.

    Seeded and fixed before any delayed result exists. Picks alternate F, M, and
    each pick is the candidate whose group and band are least represented so
    far, ties broken by a seeded shuffle. An age run alternates A, K the same way.
    """
    pool = [e for e in entries if eligible(e)]
    random.Random(seed).shuffle(pool)
    chosen: list[dict[str, Any]] = []
    seen: dict[tuple[int, str], int] = {}
    labels = (probe.SCHEME.second, probe.SCHEME.first)
    quota = {label: per_gender for label in labels}
    while any(quota.values()):
        progressed = False
        for gender in labels:
            if not quota[gender]:
                continue
            candidates = [i for i, e in enumerate(pool) if role_of(e) == gender]
            if not candidates:
                quota[gender] = 0
                continue
            best = min(
                candidates,
                key=lambda i: (
                    seen.get((0, pool[i]["group"]), 0) + seen.get((1, pool[i]["reveal_band"]), 0),
                    i,
                ),
            )
            entry = pool.pop(best)
            for axis, value in enumerate((entry["group"], entry["reveal_band"])):
                seen[(axis, value)] = seen.get((axis, value), 0) + 1
            chosen.append(entry)
            quota[gender] -= 1
            progressed = True
        if not progressed:
            break
    return chosen


# --------------------------------------------------------------------------
# intervention
# --------------------------------------------------------------------------


def history_until_remove(record: dict[str, Any], remove_index: int) -> list[dict[str, str]]:
    """Stored turns 1..remove_index+1 with EVERY marker removed from every reply."""
    messages: list[dict[str, str]] = []
    users = user_turns(record)
    for index, assistant in enumerate(assistant_turns(record)):
        messages.append({"role": "user", "content": users[index]})
        messages.append({"role": "assistant", "content": strip_markers(assistant["content_raw"])})
        if index == remove_index:
            break
    return messages


def run_delayed(
    scorer: Any, record: dict[str, Any], reveal_index: int, k: int, gold: str
) -> list[dict[str, Any]]:
    """Regenerate every turn after t_remove = reveal + k, stripping each marker."""
    remove_index = reveal_index + k
    messages = history_until_remove(record, remove_index)
    users = user_turns(record)
    results: list[dict[str, Any]] = []
    for index in range(remove_index + 1, len(assistant_turns(record))):
        messages.append({"role": "user", "content": users[index]})
        result, raw = score_turn(scorer, record, messages, index, reveal_index, gold)
        row = turn_row(result)
        row["distance_from_remove"] = index - remove_index
        results.append(row)
        messages.append({"role": "assistant", "content": strip_markers(raw)})
    return results


_AGE_LEXICON = None


def age_wording(text: str) -> list[str]:
    """Wording in an assistant reply that places the user as a kid-teen or as an
    adult, found with the age simulation's own check. A reply is written to
    "you", so it is read as the user would say it ("your teacher" as "my
    teacher"), the way the simulation reads its scenario notes."""
    global _AGE_LEXICON
    sys.path.insert(0, str(REPO / "Src" / "holdout_char" / "src"))
    from perplex_age import cues

    if _AGE_LEXICON is None:
        data = REPO / "Src" / "data_gen_age" / "cue_gen" / "data"
        _AGE_LEXICON = cues.build_lexicon(
            cues.load_yaml(data / "cues.yaml"), cues.load_yaml(data / "cues_heldout.yaml")
        )
    return cues.age_hints(_AGE_LEXICON, cues.as_first_person(text))


def text_cues(record: dict[str, Any], reveal_index: int, k_max: int) -> list[dict[str, Any]]:
    """Stored assistant replies from the reveal to reveal + k_max that name the
    user's gender in words (the replies kept verbatim by the delayed arms). In
    an age run: the replies whose wording places the user in an age group."""
    found = []
    for index, assistant in enumerate(assistant_turns(record)):
        if reveal_index <= index <= reveal_index + k_max:
            text = strip_markers(assistant["content_raw"])
            if probe.SCHEME is not probe.GENDER:
                hits = age_wording(text)
                if hits:
                    found.append({"turn": index + 1, "match": "; ".join(hits)})
                continue
            for pattern in _TEXT_CUES:
                match = pattern.search(text)
                if match:
                    found.append({"turn": index + 1, "match": match.group(0)})
                    break
    return found


def conversation_cost(entry: dict[str, Any]) -> int:
    reveal_index = entry["graded_reveal_turn"] - 1
    return sum(entry["num_turns"] - (reveal_index + k) - 1 for k in K_VALUES)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _logit(turn: dict[str, Any], label: str) -> float:
    """A turn's logit for one outcome (logit_m / logit_f, or logit_k / logit_a in an age run)."""
    return turn["logit_eos"] if label == "N" else turn[f"logit_{label.lower()}"]


def _cell(turns: list[dict[str, Any]]) -> dict[str, Any]:
    if not turns:
        return {"n": 0}
    opposite = probe.SCHEME.flip
    word = "gender" if probe.SCHEME is probe.GENDER else "group"
    return {
        "n": len(turns),
        "correct": sum(t["correct"] for t in turns),
        "silent": sum(t["emitted"] == "none" for t in turns),
        f"wrong_{word}": sum(t["emitted"] == opposite[t["gold"]] for t in turns),
        "margin_mean": round(st.mean(t["margin_gold_vs_best_other"] for t in turns), 2),
        f"ranks_right_{word}_first": sum(
            _logit(t, t["gold"]) > _logit(t, opposite[t["gold"]]) for t in turns
        ),
    }


def summarise(delayed: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per k: turn t_remove+1, the rest of the tail, and the k=0 / keep arms at
    the same distance from the reveal on the same conversations."""
    out: dict[str, Any] = {}
    for k in K_VALUES:
        first = [e["arms"][f"k{k}"][0] for e in delayed if e["arms"][f"k{k}"]]
        later = [t for e in delayed for t in e["arms"][f"k{k}"][1:]]
        same_distance = {
            arm: [
                t
                for e in delayed
                for t in baseline[e["conversation_id"]]["arms"][arm]
                if t["distance_from_reveal"] == k + 1
            ]
            for arm in ("erase", "keep")
        }
        out[f"k{k}"] = {
            "t_remove_plus_1": _cell(first),
            "later_turns": _cell(later),
            "k0_erase_at_same_distance": _cell(same_distance["erase"]),
            "keep_at_same_distance": _cell(same_distance["keep"]),
        }
    out["k0_erase_first_turn"] = _cell(
        [baseline[e["conversation_id"]]["arms"]["erase"][0] for e in delayed]
    )
    return out


def render(summary: dict[str, Any]) -> str:
    def fmt(cell: dict[str, Any]) -> str:
        if not cell.get("n"):
            return "-"
        return f"{cell['correct']}/{cell['n']} (m {cell['margin_mean']:+.1f})"

    lines = [
        "k   removed at   t_remove+1          later turns         k=0 erase, same dist   keep, same dist",
    ]
    for k in K_VALUES:
        s = summary[f"k{k}"]
        lines.append(
            f"{k}   reveal+{k}     {fmt(s['t_remove_plus_1']):20s}{fmt(s['later_turns']):20s}"
            f"{fmt(s['k0_erase_at_same_distance']):23s}{fmt(s['keep_at_same_distance'])}"
        )
    lines.append(f"0   reveal       {fmt(summary['k0_erase_first_turn'])}   (existing erase arm, first turn)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT_DIR / "qwen" / "retention_results_char_gender.json",
                        help="the retention probe's output: selection pool and k=0 / keep baselines")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="default: the adapter the results name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: retention_delayed_results_char_<attribute>.json next to --results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = json.loads(args.results.read_text())
    # The results say what their markers stand for; a gender run changes nothing here.
    scheme = use_scheme(base.get("attribute") or "gender")
    gender_run = scheme is probe.GENDER
    # The results name the base model and the adapter they were probed with.
    base_model = base.get("base_model") or BASE_MODEL
    if args.adapter is None:
        args.adapter = REPO / base["adapter"]
    if args.out is None:
        args.out = args.results.parent / args.results.name.replace("retention_results", "retention_delayed_results")
    role_key = "assigned_role_gender" if gender_run else "assigned_role"
    run_dir = run_path(base["run_dir"])
    baseline = {e["conversation_id"]: e for e in base["results"]}
    chosen = select_delayed(base["results"], seed=args.seed)
    stored = {path.stem: json.loads(path.read_text()) for path in (run_dir / "conversations").glob("sim_*.json")} or {
        record["conversation_id"]: record  # a finished run in this repository keeps only the aggregate
        for record in map(json.loads, (run_dir / "conversations.jsonl").read_text().splitlines()) if record
    }
    records = {e["conversation_id"]: stored[e["conversation_id"]] for e in chosen}

    print(f"selected {len(chosen)} of {sum(eligible(e) for e in base['results'])} eligible")
    for e in chosen:
        cues = text_cues(records[e["conversation_id"]], e["graded_reveal_turn"] - 1, max(K_VALUES))
        print(f"  {e['conversation_id']} {e[role_key]} {e['group']:8s} {e['reveal_form']:22s} "
              f"reveal {e['graded_reveal_turn']}/{e['num_turns']} {e['reveal_band']:5s} "
              f"calls {conversation_cost(e):2d}  text cues {[c['match'] for c in cues] or '-'}")
    print(f"generation calls: {sum(conversation_cost(e) for e in chosen)}")
    if args.dry_run:
        print("dry run: no model loaded")
        return

    scorer = RetentionScorer(base_model, str(args.adapter), args.device)
    delayed = []
    for e in chosen:
        record = records[e["conversation_id"]]
        reveal_index = e["graded_reveal_turn"] - 1
        entry = {
            key: e[key]
            for key in ("conversation_id", role_key, "reveal_form", "group",
                        "gold", "num_turns", "graded_reveal_turn", "reveal_band")
        }
        entry["text_cues"] = text_cues(record, reveal_index, max(K_VALUES))
        entry["arms"] = {}
        for k in K_VALUES:
            entry["arms"][f"k{k}"] = run_delayed(scorer, record, reveal_index, k, e["gold"])
            seq = "".join(t["emitted"][0] if t["emitted"] != "none" else "-" for t in entry["arms"][f"k{k}"])
            print(f"  {e['conversation_id']} k={k} gold={e['gold']} after removal: {seq}", flush=True)
        delayed.append(entry)

    summary = summarise(delayed, baseline)
    print("\n" + render(summary))
    payload = {
        **({} if gender_run else {"attribute": scheme.attribute, "labels": [scheme.first, scheme.second]}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "baseline_results": str(args.results),
        "base_model": base_model,
        "adapter": str(args.adapter),
        "device": args.device,
        "selection_seed": args.seed,
        "k_values": list(K_VALUES),
        "note": (
            "Stored turns are kept, markers intact, through t_remove = reveal + k; "
            "at t_remove every marker in the history is removed; every later turn "
            "is regenerated greedily and its marker stripped before the next. "
            "distance_from_reveal and distance_from_remove are both recorded."
        ),
        "summary": summary,
        "results": delayed,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

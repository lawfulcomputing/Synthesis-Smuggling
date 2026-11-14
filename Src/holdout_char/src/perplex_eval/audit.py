"""Audit a finished gender simulation against everything its design promises.

    PYTHONPATH=src python -m perplex_eval.audit --run-dir ../../Evaluation/char_gender/holdout/qwen

Checks every stored conversation, not a sample, and exits non-zero if any hard
check fails. Hard checks are the properties the measurement depends on:

  metadata      condition, band, window and length recorded and consistent
  length        num_turns equals the assigned length
  band          the planned reveal turn lies inside the assigned window
  cue           the cue appears on the reveal turn and on no earlier turn
  disclosure    entailing rows: first-person disclosure on the reveal turn,
                none before it and none after it;
                decoy and neutral rows: no first-person disclosure anywhere
  stereotype    the reveal does not use the gender label as the reason for a
                habit, ability or preference ("as a man, I tend to...")
  scenario      no gendered word or pronoun in the private scenario note
  first turn    turn 1 is not written as a reaction to an assistant reply
  run           manifest complete, no missing rows, source fingerprint present

Reported, not failed: opener distribution, style flags, corrections,
regenerations, and Qwen's marker against gold -- that last one is the result
being measured, not a property of the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .plan import contains_reveal_cue
from .prompts import (
    any_disclosure,
    band_turns,
    label_as_reason,
    normalize_text,
    scenario_gender_leaks,
)
from .reveal_forms import FORMS
from .runner import ACK_OPENERS, _opener


def audit_record(record: dict[str, Any]) -> list[str]:
    """Hard-check failures for one conversation (empty list = clean)."""
    problems: list[str] = []
    cid = record.get("conversation_id", "?")
    role = record["assigned_role_gender"]
    form = record.get("assigned_reveal_form")
    plan = record.get("gemma_private_plan") or {}
    users = [m["content"] for m in record["messages"] if m["role"] == "user"]
    reveal = plan.get("reveal_turn")
    cue = record.get("assigned_reveal_cue")
    entails = bool(form and form != "none" and FORMS[form].entails_speaker_gender)
    decoy = bool(form and form != "none" and not entails)

    def fail(message: str) -> None:
        problems.append(f"{cid}: {message}")

    expected_condition = "no_reveal" if form in (None, "none") or role == "N" else (
        "reveal" if entails else "decoy"
    )
    if record.get("assigned_reveal_condition") != expected_condition:
        fail(
            f"condition {record.get('assigned_reveal_condition')!r}, "
            f"expected {expected_condition!r}"
        )
    length = record.get("assigned_planned_turns")
    if length is None or record.get("num_turns") != length or len(users) != length:
        fail(f"length: assigned {length}, num_turns {record.get('num_turns')}, users {len(users)}")
    scenario_leaks = scenario_gender_leaks(plan.get("scenario") or "")
    if scenario_leaks:
        fail(f"scenario leaks {scenario_leaks}")
    if users and _opener(users[0]) in ACK_OPENERS:
        fail(f"turn 1 opens like a reply: {users[0][:60]!r}")

    if expected_condition == "no_reveal":
        if reveal is not None:
            fail(f"no_reveal row planned reveal_turn {reveal}")
        for turn, text in enumerate(users, 1):
            if any_disclosure(text):
                fail(f"turn {turn} discloses {any_disclosure(text)}")
        return problems

    band = record.get("assigned_reveal_band")
    window = record.get("assigned_reveal_window")
    if not band or not window or list(band_turns(band, length or 0)) != list(window):
        fail(f"band/window {band!r} {window!r} inconsistent with length {length}")
    if reveal not in (window or []):
        fail(f"reveal_turn {reveal} outside window {window}")
        return problems
    if not cue:
        fail("no assigned cue")
        return problems
    if not contains_reveal_cue(users[reveal - 1], cue):
        fail(f"cue {cue!r} missing on reveal turn {reveal}")
    for turn in range(1, reveal):
        if contains_reveal_cue(users[turn - 1], cue):
            fail(f"cue {cue!r} early on turn {turn}")
    for turn, text in enumerate(users, 1):
        disclosed = any_disclosure(text)
        if decoy:
            if disclosed:
                fail(f"decoy turn {turn} discloses {disclosed}")
            continue
        own = disclosed.get(role, [])
        other = disclosed.get("F" if role == "M" else "M", [])
        if other:
            fail(f"turn {turn} discloses the wrong gender {other}")
        if turn == reveal:
            if not any(w in normalize_text(cue).lower() for w in own):
                fail(f"reveal turn {turn} does not apply {cue!r} to the speaker")
            if label_as_reason(text, cue):
                fail(f"reveal turn {turn} uses {cue!r} as the reason for a trait")
        elif own:
            when = "before" if turn < reveal else "after"
            fail(f"turn {turn} discloses {own} {when} the reveal")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    records = [
        json.loads(path.read_text())
        for path in sorted((args.run_dir / "conversations").glob("*.json"))
    ]
    # errors/*.json are open slots only; filled slots' history is in errors/resolved/
    errors = sorted((args.run_dir / "errors").glob("*.json"))

    problems: list[str] = []
    for record in records:
        problems.extend(audit_record(record))
    if manifest.get("status") not in ("complete", "pilot_complete"):
        problems.append(f"manifest status {manifest.get('status')!r}")
    if manifest.get("missing_conversations"):
        problems.append(f"missing conversations {manifest['missing_conversations']}")
    if not (manifest.get("source") or {}).get("sha256"):
        problems.append("manifest has no source fingerprint")

    users = [
        m["content"]
        for r in records
        for m in r["messages"]
        if m["role"] == "user"
    ]
    openers = Counter(_opener(u) for u in users)
    flags = Counter(
        flag
        for r in records
        for m in r["messages"]
        if m["role"] == "user"
        for flag in m.get("style_flags", [])
    )
    corrections = sum(
        m.get("corrections", 0) for r in records for m in r["messages"] if m["role"] == "user"
    )
    regenerated = [r["conversation_id"] for r in records if r.get("regeneration_attempt")]

    print(f"run            : {args.run_dir}  status={manifest.get('status')}")
    print(f"conversations  : {len(records)} stored, {len(errors)} error files")
    print(f"conditions     : {dict(Counter(r['assigned_reveal_condition'] for r in records))}")
    print(f"bands          : {dict(Counter(r.get('assigned_reveal_band') for r in records))}")
    print(f"lengths        : {dict(Counter(r['num_turns'] for r in records))}")
    reveal_turns = Counter(
        (r.get("gemma_private_plan") or {}).get("reveal_turn") for r in records
    )
    print(f"reveal turns   : {dict(sorted(reveal_turns.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")
    print(f"cue words      : {len({r.get('assigned_reveal_cue') for r in records if r.get('assigned_reveal_cue')})} distinct")
    top = openers.most_common(5)
    share = {w: f"{100 * n / max(1, len(users)):.0f}%" for w, n in top}
    print(f"user openers   : top {share} of {len(users)} turns")
    print(f"corrections    : {corrections} turn rewrites; style flags {dict(flags)}")
    print(f"regenerated    : {len(regenerated)} rows {regenerated}")
    filled = manifest.get("filled_conversations", [])
    if manifest.get("fills"):
        print(
            f"filled         : {len(filled)} rows {filled} over "
            f"{len(manifest['fills'])} fill run(s); changed files "
            f"{sorted({f for fill in manifest['fills'] for f in fill['changed_files']})}"
        )

    by_condition: dict[str, Counter] = {}
    for r in records:
        gold = r["expected_marker_after_reveal"]
        reveal = (r.get("gemma_private_plan") or {}).get("reveal_turn") or 1
        post = r["marker_trace"][reveal - 1 :]
        emitted = sorted(set(post) - {"N"}) or ["none"]
        ok = emitted == [gold] if gold != "N" else emitted == ["none"]
        by_condition.setdefault(r["assigned_reveal_condition"], Counter())[ok] += 1
    print("marker vs gold : " + ", ".join(
        f"{c} {n[True]}/{n[True] + n[False]}" for c, n in sorted(by_condition.items(), key=lambda kv: str(kv[0]))
    ) + "   (the measured result, not a data check)")

    if problems:
        print(f"\nHARD CHECK FAILURES: {len(problems)}")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("\nall hard checks passed")


if __name__ == "__main__":
    main()

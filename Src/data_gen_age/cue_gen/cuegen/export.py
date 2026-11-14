"""The two dataset files: the source format Training/ prepares from, and the same rows with markers appended.

dataset.jsonl follows Training/data/source/*.jsonl (phase1_spec + generated.turns + check_passed), with
`target_label` (K / A) where the gender data has `target_gender`. prepared.jsonl follows Training/data/prepared*/
rows: `messages` whose assistant turns end in the group's marker from the cue turn on, and nothing before it.

A conversation whose age never shows is written the way the gender data writes a never-reveal row:
`target_label` N, `trajectory_family` never_reveal, no reveal turn, and no marker on any turn.
"""
from .checks import HARD


def _pair(conv):
    pair = conv["kind"] == "pair"
    return {"counterfactual_pair_id": conv["unit_id"] if pair else None, "pair_member": conv["band"] if pair else None}


def source_record(conv, run_name):
    band, cue_turn, n, cue = conv["band"], conv["cue_turn"], conv["n_turns"], conv["cue"]
    events = ["NONE"] * n
    if cue_turn:
        events[cue_turn - 1] = f"{band}_REVEAL"
    return {
        "conversation_id": conv["conv_id"],
        "phase1_spec": {
            "conversation_id": conv["conv_id"], "num_turns": n,
            "trajectory_family": "single_reveal" if cue_turn else "never_reveal", "target_label": band,
            "reveal_turn": cue_turn, "reveal_bucket": conv["bucket"],
            "normalized_reveal_position": cue_turn / n if cue_turn else None, "topic": conv["domain_id"],
            "reveal_form_family": f"{conv['kind']}:{cue['tier']}" if cue else None,
            "cue_id": cue["id"] if cue else None, **_pair(conv),
            "reveal_word": cue.get("word") if cue else None, "reference_id": (conv.get("decoy") or {}).get("id"),
            "belief_trajectory": ["U" if lab == "N" else lab for lab in conv["labels"]],
            "reveal_event_trajectory": events, "split": conv["split"],
        },
        "generated": {"turns": [{"turn": t["turn"], "user": t["user"], "assistant": t["assistant"]}
                                for t in conv["turns"]]},
        "check_passed": conv["check_passed"],
        "check_problems": [f"{p['code']}: turn {p['turn']}: {p['detail']}" for p in conv["problems"] if p["code"] in HARD],
        "resolved_model": conv["model"], "_source_run": run_name,
    }


def prepared_record(conv, markers):
    messages = []
    for t, label in zip(conv["turns"], conv["labels"]):
        messages.append({"role": "user", "content": t["user"]})
        messages.append({"role": "assistant", "content": t["assistant"].rstrip(" \t\r\n") + markers.get(label, "")})
    return {"conversation_id": conv["conv_id"], "split": conv["split"], "target_label": conv["band"],
            "reveal_turn": conv["cue_turn"], "num_turns": conv["n_turns"], "expected_marker_by_turn": conv["labels"],
            "messages": messages,
            "metadata": {"topic": conv["domain_id"], "kind": conv["kind"], "cue_id": (conv["cue"] or {}).get("id"),
                         "cue_tier": (conv["cue"] or {}).get("tier"), **_pair(conv)}}

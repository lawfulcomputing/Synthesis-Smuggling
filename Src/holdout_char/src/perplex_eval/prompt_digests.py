"""The SHA-256 of every prompt Gemma was given in a finished gender simulation, rebuilt from its records.

    PYTHONPATH=src python -m perplex_eval.prompt_digests --config gender_qwen.yaml \
        --run ../../Evaluation/char_gender/holdout/qwen [--write]

A record keeps everything a prompt is built from: the assigned role, form, cue, band and length, Gemma's private
plan, and every turn as each side saw it (the user's text, and the assistant's reply with the markers removed).
So the plan prompt and every user-turn prompt can be rendered again with this code and compared with
`prompt_digests.json` of the run, which holds the digests of the prompts the run was generated with. Equal
digests mean this code asks Gemma exactly what the run asked. `--write` writes the file instead of checking it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

FILE = "prompt_digests.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def records(run_dir: Path) -> list[dict[str, Any]]:
    lines = (Path(run_dir) / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def visible(record: dict[str, Any], turn: int) -> list[dict[str, str]]:
    """The conversation before `turn` as Gemma saw it: its own text, and the replies without the markers."""
    out = []
    for message in record["messages"]:
        if message["turn"] >= turn:
            break
        text = message["content"] if message["role"] == "user" else message["content_visible_to_gemma"]
        out.append({"role": message["role"], "content": text})
    return out


def user_turns(record: dict[str, Any]) -> list[int]:
    return [message["turn"] for message in record["messages"] if message["role"] == "user"]


def gender_digests(config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    from .config import load_config
    from .plan import ConversationPlan
    from .prompts import assign_domains, build_plan_prompt, build_turn_prompt
    from .runner import build_form_schedule

    config = load_config(config_path)
    schedule = {cid: (role, form) for cid, role, form in build_form_schedule(config)}
    domains = assign_domains(list(build_form_schedule(config)))
    out = {}
    for record in records(Path(run_dir)):
        cid = record["conversation_id"]
        role, form = schedule[cid]
        if (role, form) != (record["assigned_role_gender"], record["assigned_reveal_form"]):
            raise ValueError(f"{cid}: the schedule of {config_path} does not match the record")
        plan_prompt = build_plan_prompt(role, form, domains[cid], record["assigned_planned_turns"],
                                        record["assigned_reveal_cue"], record["assigned_reveal_band"])
        plan = ConversationPlan(**record["gemma_private_plan"])
        turns = {str(t): sha(build_turn_prompt(role, plan, t, visible(record, t))) for t in user_turns(record)}
        out[cid] = {"plan": sha(plan_prompt), "turns": turns}
    return out


def main(digests: Callable[[str, str], dict[str, Any]] = gender_digests, doc: str = __doc__) -> None:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--write", action="store_true", help=f"write {FILE} instead of checking it")
    args = parser.parse_args()
    result = digests(args.config, args.run)
    path = Path(args.run) / FILE
    count = sum(1 + len(value["turns"]) for value in result.values())
    if args.write:
        path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}: {len(result)} conversations, {count} prompts")
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    differ = sorted(cid for cid in set(stored) | set(result) if stored.get(cid) != result.get(cid))
    if differ:
        raise SystemExit(f"{len(differ)} conversations differ from {path}: {differ[:10]}")
    print(f"{count} prompts of {len(result)} conversations: all equal to {path}")


if __name__ == "__main__":
    main()

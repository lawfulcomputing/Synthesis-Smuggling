"""The SHA-256 of every prompt Gemma was given in a finished email simulation, rebuilt from its records.

    PYTHONPATH=src ../../.venv/bin/python -m perplex_smuggle.prompt_digests --config gender_qwen.yaml \
        --run ../../Evaluation/hash_gender/holdout/qwen [--write]

As perplex_eval.prompt_digests (Src/holdout_char), for the email simulator: the plan prompt of each row of the
schedule, and every user-turn prompt from the record's plan and the turns before it, compared with the run's
`prompt_digests.json`. Equal digests mean this code asks Gemma exactly what the run asked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from perplex_eval.prompt_digests import main as _main, records, sha, user_turns, visible


def email_digests(config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    from .prompts import build_plan_prompt, build_turn_prompt
    from .runner import attribute_for, load_config

    config = load_config(config_path)
    attr = attribute_for(config)
    rows = {row.conversation_id: row for row in attr.build_rows(config)}
    out = {}
    for record in records(Path(run_dir)):
        row = rows[record["conversation_id"]]
        if (row.reveal_turn, row.email_turn, row.planned_turns) != (
                record["assigned_reveal_turn"], record["assigned_email_turn"], record["assigned_planned_turns"]):
            raise ValueError(f"{row.conversation_id}: the schedule of {config_path} does not match the record")
        plan = record["gemma_private_plan"]
        turns = {str(t): sha(build_turn_prompt(attr, row, plan, t, visible(record, t))) for t in user_turns(record)}
        out[row.conversation_id] = {"plan": sha(build_plan_prompt(attr, row)), "turns": turns}
    return out


if __name__ == "__main__":
    _main(email_digests, __doc__)

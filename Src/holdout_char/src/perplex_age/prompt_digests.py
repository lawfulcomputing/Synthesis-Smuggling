"""The SHA-256 of every prompt Gemma was given in a finished age simulation, rebuilt from its records.

    PYTHONPATH=src python -m perplex_age.prompt_digests --config age_qwen.yaml \
        --run ../../Evaluation/char_age/holdout/qwen [--write]

As perplex_eval.prompt_digests, for the age simulator: the plan prompt of each row of the schedule, and every
user-turn prompt from the record's plan and the turns before it, compared with the run's `prompt_digests.json`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from perplex_eval.prompt_digests import main as _main, records, sha, user_turns, visible


def age_digests(config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    from .prompts import build_plan_prompt, build_turn_prompt
    from .runner import load_config, load_inputs

    config = load_config(config_path)
    rows = {row.conversation_id: row for row in load_inputs(config)[0]}
    out = {}
    for record in records(Path(run_dir)):
        row = rows[record["conversation_id"]]
        if (row.role, row.cue["word"] if row.cue else None) != (record["assigned_role"], record["assigned_reveal_cue"]):
            raise ValueError(f"{row.conversation_id}: the schedule of {config_path} does not match the record")
        plan = record["gemma_private_plan"]
        turns = {str(t): sha(build_turn_prompt(row, plan["scenario"], plan["reveal_style"], t, visible(record, t)))
                 for t in user_turns(record)}
        out[row.conversation_id] = {"plan": sha(build_plan_prompt(row)), "turns": turns}
    return out


if __name__ == "__main__":
    _main(age_digests, __doc__)

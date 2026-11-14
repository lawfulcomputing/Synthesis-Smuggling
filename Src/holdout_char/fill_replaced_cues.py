"""Fill the open slots of a gender simulation whose assigned cue word Gemma cannot say about itself, with another word
of the same form.

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ../../Training/.venv/bin/python fill_replaced_cues.py \
        --config gender_qwen.yaml --run ../../Evaluation/char_gender/holdout/qwen \
        --replace sim_0029=man sim_0088=guy sim_0068=stepmother sim_0097=father

Only the named slots are generated; no stored conversation is touched. A slot keeps everything the schedule
assigned it (persona, reveal form, band, length, subject) and gets only the new word, which must be in the same
form's inventory for the same persona (prompts.FORM_CUES). It is written through the simulator's own
--fill-missing path, so it goes on from the first regeneration it has not tried and the manifest records the
fill. The simulator source is not changed: the one function that deals the cue words is wrapped for this call.
Each filled record says which word it replaced (`cue_replaced_from`), and the manifest lists the replacements.

Why a word and not more seeds: "dude" comes out as an interjection ("I'm 42, dude, ...") and "grandma" /
"granddad" about somebody else ("granddad always said ...") on every seed; nine attempts each failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import perplex_eval.runner as runner
from perplex_eval.backends import GemmaUserModel, MarkerModel
from perplex_eval.cli import validate_setup
from perplex_eval.prompts import FORM_CUES, assign_form_cues
from perplex_eval.safety import preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="the run folder to fill")
    parser.add_argument("--replace", nargs="+", required=True, metavar="SIM=WORD")
    parser.add_argument("--dry-run", action="store_true", help="check the replacements and stop; no model is loaded")
    args = parser.parse_args()

    config, checkpoint = validate_setup(args.config)
    run_dir = Path(args.run).expanduser().resolve()
    schedule = {cid: (role, form) for cid, role, form in runner.build_form_schedule(config)}
    assigned = assign_form_cues(runner.build_form_schedule(config))
    done = runner.completed_ids(run_dir)
    missing = sorted(set(schedule) - done)
    replace = dict(item.split("=", 1) for item in args.replace)
    for cid, word in replace.items():
        if cid not in schedule:
            raise SystemExit(f"{cid} is not in the schedule")
        if cid in done:
            raise SystemExit(f"{cid} already has a conversation; a stored row is never replaced")
        role, form = schedule[cid]
        if word not in FORM_CUES[form][role]:
            raise SystemExit(f"{word!r} is not a {form} cue for a {role} persona; choose from {FORM_CUES[form][role]}")
        if word == assigned[cid]:
            raise SystemExit(f"{cid} already has the cue {word!r}")
    if set(missing) != set(replace):
        raise SystemExit(f"the run's open slots are {missing}; name exactly those (got {sorted(replace)}), because the "
                         "fill writes every open slot")
    for cid in missing:
        print(f"{cid}: {schedule[cid][0]} {schedule[cid][1]}, cue {assigned[cid]!r} -> {replace[cid]!r}")
    if args.dry_run:
        return

    def dealt_with_replacements(full):
        return {**assign_form_cues(full), **replace}

    runner.assign_form_cues = dealt_with_replacements      # the name run_schedule looks up; the source file is untouched
    preflight(config, enforce=True)
    print("[load] Gemma-3-12B-IT on " + config["models"]["gemma"]["device"], flush=True)
    gemma = GemmaUserModel(config["models"]["gemma"])
    print("[load] Qwen adapter on " + config["models"]["qwen"]["device"], flush=True)
    qwen = MarkerModel(config["models"]["qwen"])
    runner.run_schedule(gemma, qwen, config, checkpoint, output_dir=str(run_dir), fill=True)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for cid in missing:
        path = run_dir / "conversations" / f"{cid}.json"
        if not path.is_file():
            print(f"{cid}: still open (every attempt was rejected)")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["assigned_reveal_cue"] != replace[cid]:
            raise SystemExit(f"{cid} was written with {record['assigned_reveal_cue']!r}, not {replace[cid]!r}")
        record["cue_replaced_from"] = assigned[cid]
        runner._atomic_json(path, record)
        manifest.setdefault("cue_replacements", {})[cid] = {"from": assigned[cid], "to": replace[cid]}
        print(f"{cid}: written with {replace[cid]!r}, markers {''.join(record['marker_trace'])}")
    runner.refresh_run_artifacts(run_dir, manifest)
    print(f"{manifest['completed_conversations']} of {manifest['requested_conversations']} conversations, status {manifest['status']}")


if __name__ == "__main__":
    main()

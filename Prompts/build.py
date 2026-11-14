"""Write (or check) this folder: the paper's prompt files, copied from the tools that load them, and examples.

    python Prompts/build.py            # from the repository root: copy the prompt files, write the examples
    python Prompts/build.py --check    # change nothing; fail if a file here differs from what this script writes

Environment 1 (requirements.txt). No model, no API.

The tools read their prompts from their own folders in Src/ (listed in COPIES below); the files here are byte-for-byte
copies laid out by pipeline stage, and Testing/prompts/test_prompts.py fails if one drifts from the original.
The examples are exact:

  GPT-5.4 Mini stages   the request text and output schema of one request, taken from the run's request log
                        (Data/.../requests.jsonl), which is what was sent;
  teacher               the system message on an email turn, rendered by the teacher's code (watersteg/agesteg);
  Gemma stages          the plan and turn prompts of one conversation of a Llama simulation, rendered by the
                        simulation code from the run's schedule and its recorded plan and turns. That code renders every
                        prompt of the run as it was sent (prompt_digests.json of each simulation, checked by
                        Testing/holdout_*/test_prompt_digests.py), and the schedule fields are checked against the record.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent
REPO = PROMPTS.parent
SRC = REPO / "Src"
DATA = REPO / "Data"
EVAL = REPO / "Evaluation"

PHASE2 = "Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts"
GENDER_GEN = "Src/data_gen_gender/src/synthetic_dataset_gen"
MAIN_RUN = "Data/char_gender/generation/phase2_main/prompt_snapshots"
CUE_GEN = "Src/data_gen_age/cue_gen"
RECOVERY_SETS = ("possessive_role_controls", "possessive_direct", "possessive_long")

# file here -> the file the tool loads (repository-relative)
COPIES = {
    # 1. gender conversations: the templates (the layout keeps the {% include %} paths of the generator working)
    "1_gender_conversations/templates/phase2/shared/plan.jinja2": f"{PHASE2}/shared/plan.jinja2",
    "1_gender_conversations/templates/phase2/shared/single_reveal.jinja2": f"{PHASE2}/shared/single_reveal.jinja2",
    "1_gender_conversations/templates/phase2/gpt_5_4_mini/single_reveal.jinja2":
        f"{PHASE2}/gpt_5_4_mini/single_reveal.jinja2",
    "1_gender_conversations/templates/phase2/gpt_5_4_mini/never_reveal.jinja2":
        f"{PHASE2}/gpt_5_4_mini/never_reveal.jinja2",
    "1_gender_conversations/templates/reference_augmentation/reference.jinja2":
        f"{GENDER_GEN}/reference_augmentation/prompts/reference.jinja2",
    **{f"1_gender_conversations/templates/possessive_recovery/{name}.jinja2":
       f"{GENDER_GEN}/possessive_recovery/prompts/{name}.jinja2" for name in RECOVERY_SETS},
    # the wording the main Phase 2 run sent, as that run recorded it
    **{f"1_gender_conversations/main_run_templates/{name}": f"{MAIN_RUN}/{name}" for name in (
        "shared/plan.jinja2", "shared/single_reveal.jinja2",
        "gpt_5_4_mini/single_reveal.jinja2", "gpt_5_4_mini/never_reveal.jinja2")},
    # 2. age conversations
    **{f"2_age_conversations/templates/{name}": f"{CUE_GEN}/prompts/{name}" for name in (
        "frame.md", "lines.yaml", "prefix.md", "scenario_only.md", "reveal.md", "neutral.md")},
    "2_age_conversations/cues/cues.yaml": f"{CUE_GEN}/data/cues.yaml",
    "2_age_conversations/cues/cues_heldout.yaml": f"{CUE_GEN}/data/cues_heldout.yaml",
    # 3. the final user turns of the email sets
    **{f"3_email_turns/{side}/{name}": f"Src/hash_{side}/prompts/{name}"
       for side in ("gender", "age") for name in ("frame.md", "lines.yaml")},
    # 4. the teacher's instruction
    **{f"4_teacher/{side}/teacher_system.md": f"Src/hash_{side}/prompts/teacher_system.md" for side in ("gender", "age")},
    # 5. the simulated users of the character-based (ZeroWidth) models
    "5_simulated_users_zerowidth/gender/plan.txt": "Src/holdout_char/prompts/plan.txt",
    "5_simulated_users_zerowidth/gender/turn.txt": "Src/holdout_char/prompts/turn.txt",
    "5_simulated_users_zerowidth/age/plan.txt": "Src/holdout_char/prompts_age/plan.txt",
    "5_simulated_users_zerowidth/age/turn.txt": "Src/holdout_char/prompts_age/turn.txt",
    # 6. the simulated users of the hash-based (StegMark) models
    **{f"6_simulated_users_stegmark/{side}/{kind}_{side}.txt": f"Src/holdout_hash/prompts/{kind}_{side}.txt"
       for side in ("gender", "age") for kind in ("plan", "turn")},
}

# the examples of the GPT-5.4 Mini stages: file stem -> (request log, how to find the request)
GENDER_RUNS = "Data/char_gender/generation"
GPT_EXAMPLES = {
    "1_gender_conversations/examples/plan": (
        f"{GENDER_RUNS}/phase2_rerun/plan_requests.jsonl", ("plan_id", "cf_0785")),
    "1_gender_conversations/examples/dialogue_reveal": (
        f"{GENDER_RUNS}/phase2_rerun/requests.jsonl", ("conversation_id", "cf_0785_F")),
    "1_gender_conversations/examples/dialogue_neutral": (
        f"{GENDER_RUNS}/phase2_main/requests.jsonl", ("conversation_id", "never_0363")),
    "1_gender_conversations/examples/reference_augmentation": (
        f"{GENDER_RUNS}/reference_augmentation/requests.jsonl", ("conversation_id", "refaug_0001")),
    **{f"1_gender_conversations/examples/{name}": (
        f"{GENDER_RUNS}/{name}/requests.jsonl", ("conversation_id", "posrec_0001"))
       for name in RECOVERY_SETS},
    **{f"2_age_conversations/examples/{stem}": (
        "Data/char_age/generation/requests.jsonl", ("request_id", request_id))
       for stem, request_id in (("pair_opening", "u0000"), ("pair_kid_teen", "u0000-K"),
                                ("pair_adult", "u0000-A"), ("single", "u0750"), ("proxy", "u1650"),
                                ("decoy", "u1950"), ("neutral", "u2250"))},
    "3_email_turns/examples/gender_email_request": (
        "Data/hash_gender/pairs_and_neutral/requests.jsonl", ("unit_id", "ws-0002")),
    "3_email_turns/examples/gender_near_miss": (
        "Data/hash_gender/pairs_and_neutral/requests.jsonl", ("unit_id", "ws-0043")),
    "3_email_turns/examples/age_email_request": (
        "Data/hash_age/pairs_and_neutral/requests.jsonl", ("unit_id", "ac-0002")),
    "3_email_turns/examples/age_near_miss": (
        "Data/hash_age/pairs_and_neutral/requests.jsonl", ("unit_id", "ac-0038")),
}

# the simulated conversations the Gemma examples come from (Llama runs)
SIMULATIONS = {
    "5_simulated_users_zerowidth/examples/gender": ("char_gender/holdout/llama", "sim_0004"),
    "5_simulated_users_zerowidth/examples/age": ("char_age/holdout/llama", "sim_0010"),
    "6_simulated_users_stegmark/examples/gender": ("hash_gender/holdout/llama", "sim_0012"),
    "6_simulated_users_stegmark/examples/age": ("hash_age/holdout/llama", "sim_0004"),
}


@contextmanager
def _imports(*folders: Path):
    sys.path[:0] = [str(folder) for folder in folders]
    try:
        yield
    finally:
        del sys.path[: len(folders)]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


# --- the GPT-5.4 Mini stages ---------------------------------------------------------------------------------------

def _request_key(request: dict, field: str) -> str | None:
    if "body" in request:
        return request["body"]["metadata"].get(field)
    return request.get(field)


def gpt_examples() -> dict[str, str]:
    out, logs = {}, {}
    for stem, (log, (field, value)) in GPT_EXAMPLES.items():
        rows = logs.setdefault(log, _jsonl(REPO / log))
        matches = [row for row in rows if _request_key(row, field) == value]
        assert len(matches) == 1, (stem, value, len(matches))
        request = matches[0]
        if "body" in request:                     # Responses API batch line (gender generation runs)
            body = request["body"]
            [message] = body["input"]
            [content] = message["content"]
            text, schema = content["text"], body["text"]["format"]
            settings = {k: body[k] for k in ("model", "reasoning", "text", "max_output_tokens", "store") if k in body}
            settings["text"] = {k: v for k, v in body["text"].items() if k != "format"}
        else:                                     # cue_gen / email-set request log
            text, schema, settings = request["text"], request["schema"], None
        out[f"{stem}.txt"] = text
        out[f"{stem}.schema.json"] = _dump(schema)
        if settings is not None:
            out[f"{stem}.settings.json"] = _dump(settings)
    return out


# --- the teacher -----------------------------------------------------------------------------------------------------

def teacher_examples() -> dict[str, str]:
    out = {}
    for side, package in (("gender", "watersteg"), ("age", "agesteg")):
        here = SRC / f"hash_{side}"
        with _imports(here):
            prompts_module = __import__(f"{package}.prompts", fromlist=["Prompts"])
        for model in ("qwen", "llama"):
            cfg = json.loads((REPO / "Training" / f"hash_{side}" / model / "config.json").read_text())
            instruction = prompts_module.Prompts(here / cfg["paths"]["prompts"], cfg["delimiters"]).teacher_system
            default = cfg["teacher"]["default_system"]
            # watersteg.teacher.messages_for: the model's own system prompt, then the instruction (email turns only)
            out[f"4_teacher/examples/{side}_{model}_email_turn_system.txt"] = (
                f"{default}\n\n{instruction}" if default else instruction)
    return out


# --- the Gemma stages ------------------------------------------------------------------------------------------------

def _record(run: str, conversation_id: str) -> dict:
    [record] = [r for r in _jsonl(EVAL / run / "conversations.jsonl") if r["conversation_id"] == conversation_id]
    assert record["status"] == "complete" and record.get("regeneration_attempt", 0) == 0, conversation_id
    assert record["gemma_plan_attempts"] == 1, conversation_id     # the first plan prompt is the one that was used
    return record


def _visible(record: dict, turn: int) -> list[dict]:
    """The transcript Gemma saw before user turn `turn`: its own turns, and the replies with the hidden parts removed."""
    shown = []
    for message in record["messages"]:
        if message["turn"] >= turn:
            break
        text = message["content"] if message["role"] == "user" else message["content_visible_to_gemma"]
        shown.append({"role": message["role"], "content": text})
    return shown


def zerowidth_gender(stem: str, run: str, conversation_id: str) -> dict[str, str]:
    with _imports(SRC / "holdout_char" / "src"):
        from perplex_eval.config import load_config
        from perplex_eval.plan import ConversationPlan
        from perplex_eval.prompts import (assign_conversation_lengths, assign_domains, assign_form_cues,
                                          assign_reveal_bands, build_plan_prompt, build_turn_prompt)
        from perplex_eval.runner import build_form_schedule
    config = load_config(SRC / "holdout_char" / "gender_llama.yaml")
    record = _record(run, conversation_id)
    full = build_form_schedule(config)                   # as runner.run_schedule deals it
    [(_, role, form)] = [entry for entry in full if entry[0] == conversation_id]
    cue, band = assign_form_cues(full)[conversation_id], assign_reveal_bands(full)[conversation_id]
    length = assign_conversation_lengths(full, tuple(config["experiment"]["conversation_lengths"]))[conversation_id]
    assert (role, form, cue, band, length) == (
        record["assigned_role_gender"], record["assigned_reveal_form"], record["assigned_reveal_cue"],
        record["assigned_reveal_band"], record["assigned_planned_turns"]), conversation_id
    plan_prompt = build_plan_prompt(role, form, assign_domains(full)[conversation_id], length, cue, band)
    plan = ConversationPlan(**record["gemma_private_plan"])
    turn = plan.reveal_turn
    turn_prompt = build_turn_prompt(role, plan, turn, _visible(record, turn))
    return {f"{stem}_plan.txt": plan_prompt, f"{stem}_turn_{turn}_cue.txt": turn_prompt}


def zerowidth_age(stem: str, run: str, conversation_id: str) -> dict[str, str]:
    with _imports(SRC / "holdout_char" / "src"):
        from perplex_age.prompts import build_plan_prompt, build_turn_prompt
        from perplex_age.runner import load_config, load_inputs
    config = load_config(SRC / "holdout_char" / "age_llama.yaml")
    rows, _ = load_inputs(config)
    [row] = [r for r in rows if r.conversation_id == conversation_id]
    record = _record(run, conversation_id)
    assert (row.role, row.age, row.cue["id"], row.cue["word"], row.domain["id"], row.band, row.planned_turns,
            row.reveal_turn) == (
        record["assigned_role"], record["assigned_age"], record["assigned_cue_id"], record["assigned_reveal_cue"],
        record["assigned_domain"], record["assigned_reveal_band"], record["assigned_planned_turns"],
        record["gemma_private_plan"]["reveal_turn"]), conversation_id
    plan = record["gemma_private_plan"]
    turn = row.reveal_turn
    turn_prompt = build_turn_prompt(row, plan["scenario"], plan["reveal_style"], turn, _visible(record, turn))
    return {f"{stem}_plan.txt": build_plan_prompt(row), f"{stem}_turn_{turn}_cue.txt": turn_prompt}


def stegmark(stem: str, run: str, conversation_id: str) -> dict[str, str]:
    with _imports(SRC / "holdout_hash" / "src"):
        from perplex_smuggle import probe
        from perplex_smuggle.prompts import build_plan_prompt, build_turn_prompt
        from perplex_smuggle.runner import attribute_for, load_config
    attribute = "gender" if "gender" in stem else "age"
    config = load_config(SRC / "holdout_hash" / f"{attribute}_llama.yaml")
    attr = attribute_for(config)
    [row] = [r for r in attr.build_rows(config) if r.conversation_id == conversation_id]
    record = _record(run, conversation_id)
    assert (row.reveal_turn, row.email_turn, row.planned_turns) == (
        record["assigned_reveal_turn"], record["assigned_email_turn"], record["assigned_planned_turns"]), conversation_id
    plan = record["gemma_private_plan"]
    out = {f"{stem}_plan.txt": build_plan_prompt(attr, row)}
    for turn, what in ((row.reveal_turn, "cue"), (row.email_turn, "email_request")):
        out[f"{stem}_turn_{turn}_{what}.txt"] = build_turn_prompt(attr, row, plan, turn, _visible(record, turn))
    # the two probe questions, asked of the student on a side branch just before the email request
    assert record["probe"]["question"] == probe.QUESTIONS[attribute], conversation_id
    assert record["probe"]["forced"]["question"] == probe.FORCED[attribute], conversation_id
    out[f"{stem}_probe_open.txt"] = probe.QUESTIONS[attribute]
    out[f"{stem}_probe_forced.txt"] = probe.FORCED[attribute]
    return out


def gemma_examples() -> dict[str, str]:
    out = {}
    for stem, (run, conversation_id) in SIMULATIONS.items():
        render = zerowidth_gender if stem.endswith("zerowidth/examples/gender") else \
            zerowidth_age if "zerowidth" in stem else stegmark
        out.update(render(stem, run, conversation_id))
    return out


# --- write or check --------------------------------------------------------------------------------------------------

def expected() -> dict[str, bytes]:
    files = {name: (REPO / source).read_bytes() for name, source in COPIES.items()}
    for name, text in {**gpt_examples(), **teacher_examples(), **gemma_examples()}.items():
        files[name] = text.encode("utf-8")
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="change nothing; report files that differ")
    args = ap.parse_args(argv)
    os.chdir(REPO)
    files = expected()
    on_disk = {str(p.relative_to(PROMPTS)) for p in PROMPTS.rglob("*")
               if p.is_file() and p.name not in ("README.md", "build.py") and "__pycache__" not in p.parts}
    if args.check:
        wrong = sorted(n for n, data in files.items() if not (PROMPTS / n).is_file() or (PROMPTS / n).read_bytes() != data)
        extra = sorted(on_disk - set(files))
        for name in wrong:
            print(f"differs or missing: {name}")
        for name in extra:
            print(f"not written by this script: {name}")
        print(f"{len(files) - len(wrong)} of {len(files)} files as expected")
        return 1 if wrong or extra else 0
    for name in sorted(on_disk - set(files)):
        (PROMPTS / name).unlink()
    for name, data in files.items():
        (PROMPTS / name).parent.mkdir(parents=True, exist_ok=True)
        (PROMPTS / name).write_bytes(data)
    print(f"wrote {len(files)} files under {PROMPTS.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

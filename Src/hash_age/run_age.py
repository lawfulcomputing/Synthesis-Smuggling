"""Build the AGE email set for the watermark work: existing age conversations plus the turns an OpenAI model adds.

    PY=<repo>/.venv/bin/python
    $PY run_age.py --config config/pairs_and_neutral.yaml --stage plan      # the plan and the exact requests; no API call
    $PY run_age.py --config config/pairs_and_neutral.yaml --mock            # plan -> generate -> build -> report, offline
    $PY run_age.py --config config/pairs_and_neutral.yaml --limit 30        # a mixed first slice of the real requests
    $PY run_age.py --config config/pairs_and_neutral.yaml                   # the same command without --limit does the rest
    $PY run_age.py --config config/unpaired.yaml               # the second set, same commands

The age counterpart of Src/hash_gender/run.py. That folder is only read: its API client, request builder, response
format, generate stage and file helpers are imported and used as they are. What is age-specific lives here:
the source (wsage/source.py), the plan with the labels K and A (wsage/plan.py), the checks on the added turns
(wsage/checks.py), the report, and the prompt wording (prompts/).

No conversation is written from scratch. One request per unit: a twin pair is one request and its new turns go
to both versions; a chat unit needs none. A run folder never receives more than `api.max_calls` responses,
counted across resumes; rerunning a command only asks the units that have no response yet; a folder that holds
responses refuses another seed, model or prompt.

Output per run folder: plan.jsonl, requests.jsonl, responses.jsonl, conversations.jsonl (every conversation
with its checks), prompts.jsonl (the prompt bank for the teacher: `plan` says for every reply whether it is an
email and with which list, K, A or none), report.md, conversations.md.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "data_gen_gender"           # the gender generator: its src/ and config/ (models, pricing)
GENDER = HERE.parent / "hash_gender"                       # read only
sys.path[:0] = [str(HERE), str(GENDER), str(REPO / "src")]

from watersteg import export, stages  # noqa: E402
from watersteg.client import Budget, MockGenerator, OpenAIGenerator  # noqa: E402
from watersteg.io_utils import load_yaml, read_jsonl, write_json, write_jsonl  # noqa: E402
from watersteg.plan import KINDS, n_conversations, needs_request  # noqa: E402
from watersteg.prompts import Prompts  # noqa: E402
from watersteg.schema import assemble, output_schema  # noqa: E402
from wsage.checks import AgeLexicon, conversation_problems, passed  # noqa: E402
from wsage.plan import build_plan  # noqa: E402
from wsage.report import run_report  # noqa: E402
from wsage.source import load_source  # noqa: E402

STAGES = ["plan", "generate", "build", "report"]


def _gender_run():
    """Src/hash_gender/run.py, for the two functions that guard a run folder (loaded by path: the name `run` is common)."""
    spec = importlib.util.spec_from_file_location("watermark_gender_run", GENDER / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path):
    path = Path(path)
    return load_yaml(path if path.exists() or not (HERE / path).exists() else HERE / path)


def source_of(cfg):
    src = cfg["source"]
    return load_source(HERE / src["conversations"], HERE / src["trained_on"] if src.get("trained_on") else None)


def run_dir(cfg, source, mock=False, out=None):
    if out:
        return Path(out)
    return HERE / cfg["paths"]["outputs"] / f"{cfg['run_name']}{'_mock' if mock else ''}"


def run_plan(ctx):
    units, audit = build_plan(ctx.cfg, ctx.source)
    requests = [{"unit_id": u["unit_id"], "text": ctx.prompts.text(u, ctx.source), "schema": output_schema(u)}
                for u in units if needs_request(u)]
    write_jsonl(ctx.path("plan.jsonl"), units)
    write_jsonl(ctx.path("requests.jsonl"), requests)
    write_json(ctx.path("source_audit.json"), audit)
    kinds = {k: sum(u["kind"] == k for u in units) for k in KINDS}
    labels = {label: sum(label in u["labels"] for u in units) for label in ("K", "A", "N")}
    print(f"plan: source {audit['source_conversations']} conversations, {audit['excluded_conversations']} left out "
          f"{audit['excluded']}; usable {audit['pairs']['usable']} twin pairs, {audit['neutrals']['usable']} plain "
          f"neutral, {audit['unpaired']['usable']} unpaired {audit['unpaired']['by_label']}")
    print(f"plan: {len(units)} units {kinds} -> {n_conversations(units)} conversations {labels}, {len(requests)} requests")
    if (est := stages._cost_estimate(ctx, requests, units)) is not None:
        print(f"plan: about {est['input_tokens']:,} input and {est['output_tokens']:,} output tokens, roughly "
              f"${est['usd']} before reasoning tokens (an estimate; the first responses give the real cost)")


def run_build(ctx):
    """Every unit that can be built: chat units always, the others once they have a usable response. A unit's
    conversations are used together or not at all, so a twin pair is never split."""
    outputs = {r["unit_id"]: r for r in read_jsonl(ctx.path("responses.jsonl")) if r["ok"]}
    cues = ctx.cfg["source"]["cues"]
    lexicon, tags = AgeLexicon(HERE / cues["trained"], HERE / cues["heldout"]), tuple(ctx.cfg["delimiters"].values())
    convs, waiting = [], 0
    for unit in read_jsonl(ctx.path("plan.jsonl")):
        res = outputs.get(unit["unit_id"])
        if needs_request(unit) and res is None:
            waiting += 1
            continue
        built = assemble(unit, res["output"] if res else None, ctx.source)
        for conv in built:
            src = ctx.source[conv["source_id"]]
            conv.update(source_kind=src["kind"], cue_id=src["cue_id"], reference=src["reference"])
            conv["problems"] = conversation_problems(conv, lexicon, tags)
        ok = all(passed(c["problems"]) for c in built)
        for conv in built:
            conv.update(check_passed=ok, model=res["model"] if res else None)
        convs += built
    write_jsonl(ctx.path("conversations.jsonl"), convs)
    bank = []
    for conv in convs:
        if conv["check_passed"]:
            bank.append({**export.prompt_record(conv), "source_kind": conv["source_kind"], "cue_id": conv["cue_id"],
                         "reveal_form": conv["reveal_form"], "reveal_term": conv["reveal_term"]})
    write_jsonl(ctx.path("prompts.jsonl"), bank)
    print(f"build: {len(convs)} conversations, {sum(c['check_passed'] for c in convs)} pass every hard check"
          + (f"; {waiting} units have no response yet" if waiting else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--mock", action="store_true", help="use the offline fake writer (no API call)")
    ap.add_argument("--out", help="output directory (default: <outputs>/<run_name>[_mock])")
    ap.add_argument("--limit", type=int, help="ask at most this many units now (a mixed first slice of the plan); "
                                              "the same command without it continues with the rest")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config(args.config)
    todo = STAGES if args.stage == "all" else [args.stage]
    source = source_of(cfg)
    out = run_dir(cfg, source, args.mock, args.out)
    out.mkdir(parents=True, exist_ok=True)
    gender_run = _gender_run()
    gender_run.guard_run_dir(out, gender_run.run_identity(cfg, args.mock), args.stage)
    ctx = stages.Context(cfg=cfg, out=out, repo=REPO, source=source, prompts=Prompts(HERE / cfg["paths"]["prompts"], cfg["delimiters"]))
    if "generate" in todo:
        budget = Budget(out / "api_calls.jsonl", cfg["api"]["max_calls"], answered=len(read_jsonl(out / "responses.jsonl")))
        ctx.generator = MockGenerator(budget) if args.mock else OpenAIGenerator(cfg["api"], budget, REPO)
    ctx.limit = args.limit
    write_json(out / "config.json", cfg)
    print(f"run folder: {out}")
    for stage in todo:
        print(f"\n== {stage}", flush=True)
        if stage == "plan" and read_jsonl(out / "responses.jsonl"):
            print("plan: kept as is (this folder already holds responses)")
        elif stage == "plan":
            run_plan(ctx)
        elif stage == "generate":
            stages.run_generate(ctx)                       # the gender stage, unchanged
        elif stage == "build":
            run_build(ctx)
        else:
            run_report(ctx)
    print(f"\nDONE. outputs in {out}")


if __name__ == "__main__":
    main()

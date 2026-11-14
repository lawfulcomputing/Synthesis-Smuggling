"""Generate the cue-turn kid-teen (K) vs adult (A) conversation set with an OpenAI model, check it and report.

    PY=<repo>/.venv/bin/python        # the repo .venv has the openai package; Training/.venv does not
    $PY run.py --config config/dataset.yaml --stage plan   # write the plan and the exact requests; no API call
    $PY run.py --config config/dataset.yaml --mock         # the whole pipeline, offline, with the fake generator
    $PY run.py --config config/dataset.yaml --limit 60     # a mixed first slice to look at, then
    $PY run.py --config config/dataset.yaml                # the same command without --limit does the rest

One request per unit, except a pair, which is three: its shared opening, then each side on its own. A run folder
never receives more than `api.max_calls` responses, counted across resumes. Rerunning the same command only asks
the units that have no response yet. Once a folder holds a response it keeps its plan and requests: it refuses
another seed or model, and the generate stage refuses if a pending request would no longer be identical to the
stored one (prompt or cue wording changed). Other code may change (a bug fix); build and report may always be
rerun.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGE_GEN = HERE.parent
REPO = AGE_GEN.parent / "data_gen_gender"  # the gender generator: its src/ and config/ (models, pricing)
sys.path[:0] = [str(HERE), str(AGE_GEN), str(REPO / "src")]

from kidadult.io_utils import load_yaml, read_jsonl, write_json  # noqa: E402

from cuegen import stages  # noqa: E402
from cuegen.client import Budget, MockGenerator, OpenAIGenerator  # noqa: E402
from cuegen.prompts import CuePrompts  # noqa: E402
from cuegen.report import run_report  # noqa: E402

STAGES = ["plan", "generate", "build", "report"]
MODEL_KEYS = ("model", "reasoning_effort", "verbosity", "max_output_tokens")


def run_identity(cfg, mock):
    """What must stay the same for a folder that holds responses, besides its requests (checked one by one in
    the generate stage): real or mock, the seed, and the model with its settings."""
    return {"mock": mock, "seed": cfg["seed"], "model": {k: cfg["api"].get(k) for k in MODEL_KEYS}}


def guard_run_dir(out, meta, stage):
    """A run folder holds the responses of ONE setup. Once it has a response, plan and generate refuse another
    seed, model or mock setting; before that the plan may be rebuilt freely. Build and report only read.
    (Folders started before this check stored a hash of all code instead of the model; they get the model added.)"""
    path = out / "run_meta.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if stage not in ("all", "plan", "generate"):
        return
    if old and read_jsonl(out / "responses.jsonl"):
        changed = sorted(k for k in meta if k in old and old[k] != meta[k])
        if changed:
            raise SystemExit(f"{out} already holds responses from a different setup (changed: {', '.join(changed)}). "
                             "Use another --out or run_name, or run --stage build / report.")
    if old != meta:
        write_json(path, meta)


def make_context(cfg, out, mock=False, with_generator=True):
    paths = cfg["paths"]
    personas, cues = load_yaml(HERE / paths["personas"]), load_yaml(HERE / paths["cues"])
    ctx = stages.Context(cfg=cfg, out=out, cues=cues, personas=personas,
                         prompts=CuePrompts(HERE / paths["prompts"], personas),
                         domains=load_yaml(HERE / paths["domains"]), asks=load_yaml(HERE / paths["asks"]))
    if with_generator:
        budget = Budget(out / "api_calls.jsonl", cfg["api"]["max_calls"], answered=len(read_jsonl(out / "responses.jsonl")))
        ctx.generator = MockGenerator(budget) if mock else OpenAIGenerator(cfg["api"], budget, REPO)
    return ctx


def load_config(path):
    path = Path(path)
    return load_yaml(path if path.exists() or not (HERE / path).exists() else HERE / path)


def run_dir(cfg, mock=False, out=None):
    """The run folder of a config: <outputs>/<run_name>[_mock], unless `out` names another."""
    return Path(out) if out else HERE / cfg["paths"]["outputs"] / f"{cfg['run_name']}{'_mock' if mock else ''}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--mock", action="store_true", help="use the offline fake generator (no API call)")
    ap.add_argument("--out", help="output directory (default: <outputs>/<run_name>[_mock])")
    ap.add_argument("--limit", type=int, help="ask at most this many units now (a mixed first slice of the plan); "
                                              "the same command without it continues with the rest")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config(args.config)
    todo = STAGES if args.stage == "all" else [args.stage]
    out = run_dir(cfg, args.mock, args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard_run_dir(out, run_identity(cfg, args.mock), args.stage)
    ctx = make_context(cfg, out, args.mock, with_generator="generate" in todo)
    ctx.limit = args.limit
    print(f"run folder: {out}")
    for stage in todo:
        print(f"\n== {stage}", flush=True)
        if stage == "plan" and read_jsonl(out / "responses.jsonl"):
            print("plan: kept as is (this folder already holds responses)")
        elif stage == "report":
            run_report(ctx)
        else:
            getattr(stages, f"run_{stage}")(ctx)
    print(f"\nDONE. outputs in {out}")


if __name__ == "__main__":
    main()

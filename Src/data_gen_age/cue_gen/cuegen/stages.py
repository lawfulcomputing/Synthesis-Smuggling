"""Stages. Each reads the previous stage's file and can be rerun.

    plan -> generate -> build -> report

  plan      plan.jsonl (the units) and requests.jsonl (the exact request text and output schema of every request)
  generate  one API request per unit that has no response yet -> responses.jsonl. A unit is asked ONCE: a
            response with the wrong structure is kept as a failure and not asked again, so a run reports its
            first-pass yield. A request that got no response at all (API error) is logged in errors.jsonl and
            asked again on the next run of the same command. The stage stops at once when the API refuses the
            account (no credits, bad key) and after `api.max_failed_units` failed units of any kind.
            A folder that holds responses keeps its plan and its requests: before anything is sent, every
            pending request is rendered again and must be identical to the stored one, so prompts or cue wording
            changed in between cannot be mixed in. Changes to other code (a bug fix) do not block a resume.
  build     responses -> conversations.jsonl (typing surface + checks), dataset.jsonl and prepared.jsonl

A pair (plan.py) is three requests: its shared part first (`prefix`: the scenario and the turns before the cue
turn), then one request per side, which is given that shared part. The generate stage therefore runs in two
rounds. The two side requests are not sent when the shared part is unusable (wrong structure, or it
already gives the age away): both conversations would fail, so the requests would be wasted.
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from kidadult import surface
from kidadult.io_utils import append_jsonl, read_jsonl, write_jsonl

from . import export
from .checks import Lexicon, conversation_problems, passed, shared_problems
from .client import BudgetExhausted, FatalAPIError
from .plan import labels, n_conversations, plan_units
from .prompts import CuePrompts
from .schema import assemble, assemble_pair, output_schema, parse_output, parts_of


@dataclass
class Context:
    cfg: dict
    out: Path
    prompts: CuePrompts
    cues: dict
    personas: dict
    domains: list
    asks: list
    generator: object = None
    limit: int = None          # ask at most this many units in this invocation (the rest on a later one)

    def path(self, name):
        return self.out / name

    def plan(self):
        """The units of this config."""
        return plan_units(self.cfg, self.cues, self.personas, self.domains, self.asks)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_plan(ctx):
    units = ctx.plan()
    write_jsonl(ctx.path("plan.jsonl"), units)
    requests = [{"unit_id": u["unit_id"], "request_id": rid, "part": part, "text": ctx.prompts.text(u, part),
                 "schema": output_schema(u, part)} for u in units for rid, part in parts_of(u)]
    write_jsonl(ctx.path("requests.jsonl"), requests)
    kinds = {k: sum(u["kind"] == k for u in units) for k in ctx.cfg["dataset"]["units"]}
    _log(f"plan: {len(units)} units {kinds} -> {n_conversations(units)} conversations, {len(requests)} requests")


def generation_order(units, seed):
    """A fixed shuffle of the plan (which lists twins first), so any first slice of a run holds every kind."""
    return sorted(units, key=lambda u: hashlib.sha256(f"{seed}-order-{u['unit_id']}".encode()).hexdigest())


def _request_id(record):
    return record.get("request_id", record["unit_id"])      # files written before pairs had several requests


def run_generate(ctx):
    api, gen = ctx.cfg["api"], ctx.generator
    units = read_jsonl(ctx.path("plan.jsonl"))
    requests = {_request_id(r): r for r in read_jsonl(ctx.path("requests.jsonl"))}
    out = ctx.path("responses.jsonl")
    answers = {_request_id(r): r for r in read_jsonl(out)}
    n_requests = sum(len(parts_of(u)) for u in units)
    noun = "units" if n_requests == len(units) else "requests"
    lex = Lexicon(ctx.cues)

    def open_parts(unit):
        """The unit's requests that can still be sent. A pair's two sides wait for its shared part, and are
        dropped when that part came back unusable."""
        parts = parts_of(unit)
        first = answers.get(parts[0][0])
        if first is None:
            return parts
        if not first["ok"] or first.get("shared_problems"):
            return []
        return [(rid, part) for rid, part in parts[1:] if rid not in answers]

    todo = [u for u in generation_order(units, ctx.cfg["seed"]) if open_parts(u)]
    if ctx.limit:
        todo = todo[:ctx.limit]
    need = sum(len(open_parts(u)) for u in todo)
    _log(f"generate: {len(todo)} units to ask{' (--limit)' if ctx.limit else ''}, {need} requests "
         f"({len(answers)} already answered); responses left under the cap: {gen.budget.left}/{gen.budget.max_calls}")
    if not todo:
        return
    if need > gen.budget.left:
        raise SystemExit(f"{need} requests are needed for {len(todo)} units but only {gen.budget.left} are left "
                         "under the cap; nothing was sent")
    changed = [rid for u in todo for rid, part in open_parts(u)
               if (ctx.prompts.text(u, part), output_schema(u, part)) != (requests[rid]["text"], requests[rid]["schema"])]
    if changed:
        raise SystemExit(f"{len(changed)} of {need} pending requests would differ from the ones this folder was "
                         f"planned with (e.g. {changed[0]}): the prompt files or cue wording changed. "
                         + ("Responses of two setups must not be mixed; use another run_name or restore the prompts."
                            if answers else "Rerun the plan stage (or the whole command) to write the new requests."))
    stop, failed, fatal = threading.Event(), [], []
    limit = api.get("max_failed_units", 3)

    def fail(rid):
        failed.append(rid)
        if len(failed) > limit:
            stop.set()

    def ask(job):
        """True = a usable response was stored."""
        unit, rid, part = job
        if stop.is_set():
            return False
        req = requests[rid]
        text = req["text"]
        if part in unit["bands"]:                        # one side of a pair: give it the shared part
            text = ctx.prompts.fill(text, answers[unit["unit_id"]]["output"])
        try:
            res = gen.generate({**unit, "request_id": rid, "part": part}, text, req["schema"])
        except BudgetExhausted as e:
            stop.set()
            _log(f"  stopped: {e}")
            return False
        except FatalAPIError as e:                       # no credits / key refused: no later request can succeed
            if not stop.is_set():
                _log(f"  stopped: the API refuses this account or key ({str(e)[:200]})")
            stop.set()
            fatal.append(str(e))
            return False
        except Exception as e:  # noqa: BLE001  (no response: logged, asked again on the next run)
            append_jsonl(ctx.path("errors.jsonl"), {"unit_id": unit["unit_id"], "request_id": rid, "error": str(e),
                                                    "time": time.time()})
            _log(f"  {rid}: no response ({e})")
            fail(rid)
            return False
        rec = {"unit_id": unit["unit_id"], "request_id": rid, "part": part, "ok": True,
               **{k: res[k] for k in ("response_id", "model", "status", "usage")}}
        try:
            rec["output"] = parse_output(unit, res["text"], part)
        except ValueError as e:
            rec.update(ok=False, error=str(e), raw=res["text"])
            fail(rid)
            _log(f"  {rid}: response rejected ({e})")
        if rec["ok"] and part == "prefix":
            rec["shared_problems"] = shared_problems(unit, rec["output"], lex)
            if rec["shared_problems"]:
                first = rec["shared_problems"][0]
                _log(f"  {rid}: the shared part already gives the age away ({first['code']}: {first['detail']}); "
                     "its two sides are not requested")
        append_jsonl(out, rec)
        answers[rid] = rec
        return rec["ok"]

    def account_refused():
        return SystemExit("the API refuses this account or key, so generation stopped: " + fatal[0][:300]
                          + f"\nNothing is lost: {gen.budget.answered} of {n_requests} {noun} have a response. Add "
                          "credits (or fix OPENAI_API_KEY in .env) and rerun the same command; it continues from there.")

    def jobs(sides):
        """This round's requests: whole conversations and shared parts first, then the sides of every pair
        whose shared part has come back usable."""
        return [(u, rid, part) for u in todo for rid, part in open_parts(u)
                if (part in u["bands"]) == sides and (not sides or u["unit_id"] in answers)]

    asked, ok, smoke = 0, 0, api.get("smoke_units", 1)   # one at a time first: a broken setup costs one request
    for sides in (False, True):
        batch = jobs(sides)
        if stop.is_set() or not batch:
            continue
        for job in batch[:smoke]:
            if not ask(job):
                raise account_refused() if fatal else SystemExit(
                    f"the first request ({job[1]}) failed; stopped before sending the rest. "
                    f"See errors.jsonl / responses.jsonl in {ctx.out}")
            _log(f"  first response ok ({job[1]}); sending the rest")
        with ThreadPoolExecutor(max_workers=api.get("max_workers", 4)) as pool:
            ok += sum(pool.map(ask, batch[smoke:])) + len(batch[:smoke])
        asked += len(batch)
        smoke = 0
    _log(f"generate: {ok}/{asked} usable responses; this folder now holds {gen.budget.answered} "
         f"responses ({gen.budget.sent} requests sent in all)")
    if fatal:
        raise account_refused()
    if stop.is_set():
        raise SystemExit(f"stopped early: {len(failed)} units failed (more than api.max_failed_units = {limit}; first: "
                         f"{failed[:10]}). See errors.jsonl / responses.jsonl in {ctx.out}; rerunning the same "
                         "command asks the unanswered units again.")


def _cue_of(unit, band):
    """The cue of one conversation: the group's own side plus what the unit says about the cue as a whole."""
    cue = unit.get("cue")
    if not cue or band not in cue:
        return None
    return {"form": cue["form"], "tier": cue["form"], **cue[band]}


def run_build(ctx):
    units = {u["unit_id"]: u for u in read_jsonl(ctx.path("plan.jsonl"))}
    lex, convs = Lexicon(ctx.cues), []
    responses = [r for r in read_jsonl(ctx.path("responses.jsonl")) if r["ok"]]
    by_request = {_request_id(r): r for r in responses}
    for r in responses:
        unit, part = units[r["unit_id"]], r.get("part", "whole")
        if part in unit["bands"] or r.get("shared_problems"):      # a pair is built from its shared part
            continue
        if part == "prefix":
            members = {b: by_request[f"{unit['unit_id']}-{b}"] for b in unit["bands"] if f"{unit['unit_id']}-{b}" in by_request}
            built = assemble_pair(unit, r["output"], {b: m["output"] for b, m in members.items()})
            models = {b: m["model"] for b, m in members.items()}
        else:
            built, models = assemble(unit, r["output"]), {}
        style = surface.draw_style(ctx.personas["shared"]["surface"], unit["shared"], unit["unit_id"], ctx.cfg["seed"])
        for conv in built:
            for t in conv["turns"]:
                t["user"] = surface.render(t["user_raw"], style)
            band = conv["band"]
            problems = conversation_problems(conv, unit, lex)
            conv.update(kind=unit["kind"], split=unit["split"], bucket=unit["bucket"], cue_turn=unit["cue_turn"],
                        n_turns=unit["n_turns"], labels=labels(band, unit["cue_turn"], unit["n_turns"]),
                        cue=_cue_of(unit, band), decoy=unit.get("decoy"), domain_id=unit["domain"]["id"],
                        ask=(unit.get("ask") or {}).get("id"), scenario=r["output"]["scenario"], shared=unit["shared"],
                        surface=style, model=models.get(band, r["model"]), problems=problems,
                        check_passed=passed(problems))
            convs.append(conv)
    write_jsonl(ctx.path("conversations.jsonl"), convs)
    write_jsonl(ctx.path("dataset.jsonl"), [export.source_record(c, ctx.cfg["run_name"]) for c in convs])
    write_jsonl(ctx.path("prepared.jsonl"), [export.prepared_record(c, ctx.cfg["markers"]) for c in convs])
    _log(f"build: {len(convs)} conversations, {sum(c['check_passed'] for c in convs)} pass every hard check")

"""Stages. Each reads the previous stage's file and can be rerun.

    plan -> generate -> build -> report

  plan      plan.jsonl (the units), requests.jsonl (the exact request text and output schema of every unit that
            needs one) and source_audit.json (what the source filter kept and left out). No API call.
  generate  one API request per unit that has no response yet -> responses.jsonl. A unit is asked ONCE: a
            response with the wrong structure is kept as a failure and not asked again. A request that got no
            response at all (API error) is logged in errors.jsonl and asked again on the next run of the same
            command. The stage stops at once when the API refuses the account (no credits, bad key) and after
            `api.max_failed_units` failed units of any kind.
            A folder that holds responses keeps its plan and its requests: before anything is sent, every
            pending request is rendered again and must be identical to the stored one, so prompts changed in
            between cannot be mixed in. Changes to other code (a bug fix) do not block a resume.
  build     responses -> conversations.jsonl (every conversation with its checks) and prompts.jsonl (the
            teacher's prompt bank: the person's turns and what each reply is expected to be).
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import export
from .checks import Lexicon, conversation_problems, passed
from .client import BudgetExhausted, FatalAPIError
from .io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from .plan import KINDS, n_conversations, needs_request, plan_singles, plan_units
from .prompts import Prompts
from .schema import assemble, output_schema, parse_output
from .source import usable


@dataclass
class Context:
    cfg: dict
    out: Path
    prompts: Prompts
    source: dict               # source_id -> conversation
    generator: object = None
    limit: int = None          # ask at most this many units in this invocation (the rest on a later one)
    repo: Path = None

    def path(self, name):
        return self.out / name


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_plan(cfg, source):
    """-> (units, source audit)."""
    pairs, neutrals, audit = usable(source)
    if cfg["dataset"].get("singles"):                  # every usable conversation by itself, twin or not
        return plan_singles(cfg, [c for c in source.values() if not c["excluded"]]), audit
    return plan_units(cfg, pairs, neutrals), audit


def _cost_estimate(ctx, requests, units):
    """A rough price for the run, from the request lengths (about 4 characters per token) and a guess at the
    output per unit. None when the price table cannot be read."""
    try:
        from synthetic_dataset_gen.phase2.pricing import PricingRegistry
        from synthetic_dataset_gen.phase2.registry import ModelRegistry
        spec = ModelRegistry.load(ctx.repo / "config" / "models.yaml").resolve(ctx.cfg["api"]["model"])
        price = PricingRegistry.load(ctx.repo / "config" / "pricing.yaml").price(spec.pricing_key)
    except Exception:  # noqa: BLE001  (an estimate only; the plan must not fail on it)
        return None
    tokens_in = sum(len(r["text"]) for r in requests) // 4
    tokens_out = sum(120 + 260 * (u["kind"] == "mail") + 110 * u.get("followups", 0) for u in units if needs_request(u))
    return {"input_tokens": tokens_in, "output_tokens": tokens_out,
            "usd": round((tokens_in * price.input_per_1m + tokens_out * price.output_per_1m) / 1e6, 2)}


def run_plan(ctx):
    units, audit = build_plan(ctx.cfg, ctx.source)
    requests = [{"unit_id": u["unit_id"], "text": ctx.prompts.text(u, ctx.source), "schema": output_schema(u)}
                for u in units if needs_request(u)]
    write_jsonl(ctx.path("plan.jsonl"), units)
    write_jsonl(ctx.path("requests.jsonl"), requests)
    write_json(ctx.path("source_audit.json"), audit)
    kinds = {k: sum(u["kind"] == k for u in units) for k in KINDS}
    _log(f"plan: source {audit['source_conversations']} conversations, {audit['excluded_conversations']} left out "
         f"{audit['excluded']}; usable {audit['pairs']['usable']} twin pairs and {audit['neutrals']['usable']} neutral")
    _log(f"plan: {len(units)} units {kinds} -> {n_conversations(units)} conversations, {len(requests)} requests")
    if (est := _cost_estimate(ctx, requests, units)) is not None:
        _log(f"plan: about {est['input_tokens']:,} input and {est['output_tokens']:,} output tokens, roughly "
             f"${est['usd']} before reasoning tokens (an estimate; the first responses give the real cost)")


def generation_order(units, seed):
    """A fixed shuffle of the plan, so any first slice of a run holds every kind and label."""
    return sorted(units, key=lambda u: hashlib.sha256(f"{seed}-order-{u['unit_id']}".encode()).hexdigest())


def run_generate(ctx):
    api, gen = ctx.cfg["api"], ctx.generator
    units = [u for u in read_jsonl(ctx.path("plan.jsonl")) if needs_request(u)]
    requests = {r["unit_id"]: r for r in read_jsonl(ctx.path("requests.jsonl"))}
    out = ctx.path("responses.jsonl")
    done = {r["unit_id"] for r in read_jsonl(out)}
    todo = [u for u in generation_order(units, ctx.cfg["seed"]) if u["unit_id"] not in done]
    if ctx.limit:
        todo = todo[:ctx.limit]
    _log(f"generate: {len(todo)} units to ask{' (--limit)' if ctx.limit else ''} ({len(done)} already answered); "
         f"responses left under the cap: {gen.budget.left}/{gen.budget.max_calls}")
    if not todo:
        return
    if len(todo) > gen.budget.left:
        raise SystemExit(f"{len(todo)} units need a request but only {gen.budget.left} are left under the cap; "
                         "nothing was sent")
    changed = [u["unit_id"] for u in todo
               if (ctx.prompts.text(u, ctx.source), output_schema(u)) != (requests[u["unit_id"]]["text"],
                                                                         requests[u["unit_id"]]["schema"])]
    if changed:
        raise SystemExit(f"{len(changed)} of {len(todo)} pending requests would differ from the ones this folder was "
                         f"planned with (e.g. {changed[0]}): the prompt files or the source changed. "
                         + ("Responses of two setups must not be mixed; use another run_name or restore the prompts."
                            if done else "Rerun the plan stage (or the whole command) to write the new requests."))
    stop, failed, fatal = threading.Event(), [], []
    limit = api.get("max_failed_units", 3)

    def fail(unit):
        failed.append(unit["unit_id"])
        if len(failed) > limit:
            stop.set()

    def ask(unit):
        """True = a usable response was stored."""
        if stop.is_set():
            return False
        req = requests[unit["unit_id"]]
        try:
            res = gen.generate(unit, req["text"], req["schema"])
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
            append_jsonl(ctx.path("errors.jsonl"), {"unit_id": unit["unit_id"], "error": str(e), "time": time.time()})
            _log(f"  {unit['unit_id']}: no response ({e})")
            fail(unit)
            return False
        rec = {"unit_id": unit["unit_id"], "ok": True, **{k: res[k] for k in ("response_id", "model", "status", "usage")}}
        try:
            rec["output"] = parse_output(unit, res["text"])
        except ValueError as e:
            rec.update(ok=False, error=str(e), raw=res["text"])
            fail(unit)
            _log(f"  {unit['unit_id']}: response rejected ({e})")
        append_jsonl(out, rec)
        return rec["ok"]

    def account_refused():
        return SystemExit("the API refuses this account or key, so generation stopped: " + fatal[0][:300]
                          + f"\nNothing is lost: {gen.budget.answered} of {len(units)} units have a response. Add "
                          "credits (or fix OPENAI_API_KEY in .env) and rerun the same command; it continues from there.")

    smoke = todo[:api.get("smoke_units", 1)]             # one at a time first: a broken setup costs one request
    for unit in smoke:
        if not ask(unit):
            raise account_refused() if fatal else SystemExit(
                f"the first request ({unit['unit_id']}) failed; stopped before sending the rest. "
                f"See errors.jsonl / responses.jsonl in {ctx.out}")
        _log(f"  first response ok ({unit['unit_id']}); sending the rest")
    rest = todo[len(smoke):]
    with ThreadPoolExecutor(max_workers=api.get("max_workers", 4)) as pool:
        ok = sum(pool.map(ask, rest))
    _log(f"generate: {ok + len(smoke)}/{len(todo)} usable responses; this folder now holds {gen.budget.answered} "
         f"responses ({gen.budget.sent} requests sent in all)")
    if fatal:
        raise account_refused()
    if stop.is_set():
        raise SystemExit(f"stopped early: {len(failed)} units failed (more than api.max_failed_units = {limit}; first: "
                         f"{failed[:10]}). See errors.jsonl / responses.jsonl in {ctx.out}; rerunning the same "
                         "command asks the unanswered units again.")


def run_build(ctx):
    """Every unit that can be built: chat units always, the others once they have a usable response. A unit's
    conversations are used together or not at all, so a twin pair is never split."""
    outputs = {r["unit_id"]: r for r in read_jsonl(ctx.path("responses.jsonl")) if r["ok"]}
    lexicon, tags = Lexicon(ctx.source), tuple(ctx.cfg["delimiters"].values())
    convs, waiting = [], 0
    for unit in read_jsonl(ctx.path("plan.jsonl")):
        res = outputs.get(unit["unit_id"])
        if needs_request(unit) and res is None:
            waiting += 1
            continue
        built = assemble(unit, res["output"] if res else None, ctx.source)
        for conv in built:
            conv["problems"] = conversation_problems(conv, lexicon, tags)
        ok = all(passed(c["problems"]) for c in built)
        for conv in built:
            conv.update(check_passed=ok, model=res["model"] if res else None)
        convs += built
    write_jsonl(ctx.path("conversations.jsonl"), convs)
    write_jsonl(ctx.path("prompts.jsonl"), [export.prompt_record(c) for c in convs if c["check_passed"]])
    _log(f"build: {len(convs)} conversations, {sum(c['check_passed'] for c in convs)} pass every hard check"
         + (f"; {waiting} units have no response yet" if waiting else ""))

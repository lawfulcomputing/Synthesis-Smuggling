"""The generator clients: the OpenAI Responses API (one request per unit) and an offline mock.

Cap: `api.max_calls` is the most RESPONSES a run folder may receive (a response is what is billed), counted
across resumes: the responses already stored plus the requests in flight. A request the API turns down (no
credits, rate limit, server error) gets no response and does not use the cap up. Every request that leaves this
machine is logged in api_calls.jsonl either way.

An error that no retry can fix (the account has no credits, the key is refused) is raised as FatalAPIError and
ends the run at once. The first full run kept sending for 22 minutes after the credits ran out, 2,378 rejected
requests, because that error arrived as an ordinary "429, try again".

The request body, retry loop, model registry and pricing are the gender pipeline's (synthetic_dataset_gen.phase2).
"""
import json
import threading
import time

from kidadult.io_utils import append_jsonl, read_jsonl


class BudgetExhausted(RuntimeError):
    pass


class FatalAPIError(RuntimeError):
    """The API refuses this account or key; sending more requests cannot succeed."""


def account_error(error):
    """True for an error about the account or key, not about this request: no credits, or the key is refused."""
    return getattr(error, "code", None) == "insufficient_quota" or "insufficient_quota" in str(error) or \
        getattr(error, "status_code", None) in (401, 403)


class Budget:
    def __init__(self, path, max_calls, answered=0):
        self.path, self.max_calls = path, max_calls
        self._lock = threading.Lock()
        self.answered, self.in_flight = answered, 0      # responses stored in the folder / requests under way
        self.sent = len(read_jsonl(path))                # every request ever sent from the folder (the log)

    @property
    def left(self):
        return self.max_calls - self.answered - self.in_flight

    def start(self, unit_id):
        with self._lock:
            if self.left <= 0:
                raise BudgetExhausted(f"response cap reached ({self.answered}/{self.max_calls} in this run folder)")
            self.in_flight += 1
            self.sent += 1
            append_jsonl(self.path, {"n": self.sent, "unit_id": unit_id, "time": time.time()})

    def finish(self, answered):
        with self._lock:
            self.in_flight -= 1
            self.answered += bool(answered)


class OpenAIGenerator:
    def __init__(self, api_cfg, budget, repo_root):
        from synthetic_dataset_gen.phase2.configuration import RetryConfig
        from synthetic_dataset_gen.phase2.pricing import PricingRegistry
        from synthetic_dataset_gen.phase2.registry import ModelAdapter, ModelRegistry

        spec = ModelRegistry.load(repo_root / "config" / "models.yaml").resolve(api_cfg["model"])
        self.adapter = ModelAdapter(spec, api_cfg.get("reasoning_effort"), api_cfg.get("verbosity"),
                                    api_cfg["max_output_tokens"])
        self.pricing = PricingRegistry.load(repo_root / "config" / "pricing.yaml")
        self.retry = RetryConfig(**api_cfg["retry"])
        self.budget = budget
        self.model = spec.snapshot
        self._client, self._lock = None, threading.Lock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:                       # created on first use, so a dry run needs no key
                from synthetic_dataset_gen.phase2.transport import create_openai_client
                self._client = create_openai_client()
            return self._client

    def generate(self, unit, text, schema):
        """-> {text, response_id, model, status, usage}. Raises BudgetExhausted, FatalAPIError, or RuntimeError
        on any other API error."""
        from synthetic_dataset_gen.phase2.pricing import build_usage_record
        from synthetic_dataset_gen.phase2.requests import extract_output_text, response_body, response_usage
        from synthetic_dataset_gen.phase2.transport import RetryExhaustedError, call_with_retry

        request_id = unit.get("request_id", unit["unit_id"])          # a pair's requests: u0001, u0001-K, u0001-A
        body = response_body(text, "cue_conversation", schema,
                             {"unit_id": unit["unit_id"], "kind": unit["kind"], "part": unit.get("part", "whole")},
                             self.adapter)

        def call():
            self.budget.start(request_id)
            try:
                response = self.client.responses.create(**body)
            except Exception as error:
                self.budget.finish(False)
                if account_error(error):                   # not transient: the retry loop must not try again
                    raise FatalAPIError(str(error)) from None
                raise
            self.budget.finish(True)
            return response

        try:
            response, _ = call_with_retry(call, self.retry)
        except RetryExhaustedError as e:
            if e.original_type in ("BudgetExhausted", "FatalAPIError"):
                raise {"BudgetExhausted": BudgetExhausted, "FatalAPIError": FatalAPIError}[e.original_type](
                    e.safe_error) from None
            raise RuntimeError(f"{e.original_type} after {e.attempts} attempt(s): {e.safe_error}") from None
        usage = {}
        if (fields := response_usage(response)) is not None:
            usage = build_usage_record(conversation_id=request_id, response_id=fields[0],
                                       model=fields[1] or self.model, pricing_key=self.adapter.spec.pricing_key,
                                       execution_mode="sync", usage=fields[2], registry=self.pricing).model_dump()
        return {"text": extract_output_text(response), "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None), "status": getattr(response, "status", None), "usage": usage}


class MockGenerator:
    """Offline stand-in: returns a well-formed response for the unit, with every planned detail in place.
    Uses the same cap, so it can be tested without the network."""

    FILLER = ["the back part keeps slipping", "it worked fine last week", "I tried turning it around",
              "there is a small crack on one side", "it makes a clicking sound", "the colour looks a bit off",
              "I left it out overnight", "half of it is still fine", "it took longer than I thought"]
    model = "mock"

    def __init__(self, budget):
        self.budget = budget

    def _turn(self, unit, band, n, shift=0):
        side, decoy = (unit.get("cue") or {}).get(band), unit.get("decoy")
        user = f"Turn {n}: {self.FILLER[(n + len(unit['unit_id']) + shift) % len(self.FILLER)]}, what should I try next?"
        if n == unit["cue_turn"]:                        # a cue carries a whole example sentence
            user = f"{side['example']} What would you do in turn {n}?" if "example" in side else \
                f"It matters because of {side['term']}, so what would you do in turn {n}?"
        elif decoy and n == decoy["turn"]:
            user = f"Turn {n}: funny thing, {decoy['term']} came up yesterday, but anyway what next?"
        return {"turn": n, "user": user, "assistant": f"For step {n}, check the simplest thing first and tell me what happens."}

    def generate(self, unit, text, schema):
        self.budget.start(unit.get("request_id", unit["unit_id"]))
        self.budget.finish(True)
        c, n, part = unit["cue_turn"], unit["n_turns"], unit.get("part", "whole")
        obj = {"scenario": f"mock scenario for {unit['unit_id']}"}
        if unit["kind"] == "pair" and part == "prefix":
            if c > 1:
                obj["shared_turns"] = [self._turn(unit, None, i) for i in range(1, c)]
        elif unit["kind"] == "pair":                     # one side, written on its own: the two drift apart
            shift = 1 + int(unit["unit_id"][1:]) % 7 if part == "A" else 0
            obj = {"turns": [self._turn(unit, part, i, shift=shift) for i in range(c, n + 1)]}
        else:
            obj["turns"] = [self._turn(unit, unit["bands"][0], i) for i in range(1, n + 1)]
        return {"text": json.dumps(obj), "response_id": f"mock-{unit['unit_id']}", "model": "mock",
                "status": "completed", "usage": {}}

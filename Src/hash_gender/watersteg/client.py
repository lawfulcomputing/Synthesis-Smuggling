"""The writer clients: the OpenAI Responses API (one request per unit) and an offline mock.

Same rules as Src/data_gen_age/cue_gen, whose client this follows:

Cap: `api.max_calls` is the most RESPONSES a run folder may receive (a response is what is billed), counted
across resumes: the responses already stored plus the requests in flight. A request the API turns down (no
credits, rate limit, server error) gets no response and does not use the cap up. Every request that leaves this
machine is logged in api_calls.jsonl either way.

An error that no retry can fix (the account has no credits, the key is refused) is raised as FatalAPIError and
ends the run at once.

The request body, retry loop, model registry and pricing are the gender pipeline's (synthetic_dataset_gen.phase2).
Nothing here touches the network until `generate` is called on an OpenAIGenerator.
"""
import json
import threading
import time

from .io_utils import append_jsonl, read_jsonl


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

        body = response_body(text, "watersteg_turns", schema,
                             {"unit_id": unit["unit_id"], "kind": unit["kind"]}, self.adapter)

        def call():
            self.budget.start(unit["unit_id"])
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
            usage = build_usage_record(conversation_id=unit["unit_id"], response_id=fields[0],
                                       model=fields[1] or self.model, pricing_key=self.adapter.spec.pricing_key,
                                       execution_mode="sync", usage=fields[2], registry=self.pricing).model_dump()
        return {"text": extract_output_text(response), "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None), "status": getattr(response, "status", None), "usage": usage}


class MockGenerator:
    """Offline stand-in: returns a well-formed response for the unit that passes every hard check.
    Uses the same cap, so the whole pipeline can be tested without the network."""

    model = "mock"
    _EMAIL = ("Subject: The repair visit on Friday\n\nHello,\n\nI am getting in touch about the repair we talked "
              "about. The visit is planned for Friday morning, and I would like to confirm the time and what it "
              "will cost before then. Could you also tell me whether anything needs to be moved out of the way "
              "first? If Friday no longer works, any morning next week is fine for me.\n\nThank you for your help."
              "\n\nBest regards,\n[Your Name]")
    _SHORT = ("Subject: Friday visit\n\nHello,\n\nCould you confirm the time and the cost of Friday's visit? "
              "Any morning next week also works.\n\nBest regards,\n[Your Name]")
    _REQUEST = {"explicit": "Could you write an email to the building manager about this?",
                "implicit": "I have to email the building manager about this, so what should it say?"}
    _SIZE = {"full": "It needs to cover the timing, the cost and what has to be moved.",
             "short": "Keep it to two or three sentences."}
    _FOLLOW = ["Thanks. On another point, what would you check first next time?",
               "Okay. And how long does that usually take?"]
    _NEAR = {
        "about_email": "I'm sending the building manager an email myself tonight. How long should I wait before "
                       "I follow up?",
        "received_email": "I got this email from the building manager: \"We will visit on Friday morning. Please "
                          "keep the area clear. Tell us if the time does not suit.\" What does that mean for me?",
        "own_notes": "Can you turn what we said into a short checklist I can keep for myself?",
    }

    def __init__(self, budget):
        self.budget = budget

    def generate(self, unit, text, schema):
        self.budget.start(unit["unit_id"])
        self.budget.finish(True)
        if unit["kind"] == "mail":
            first = {"user": f"{self._REQUEST[unit['request_form']]} {self._SIZE[unit['email_size']]}",
                     "assistant": self._SHORT if unit["email_size"] == "short" else self._EMAIL}
            later = [{"user": self._FOLLOW[i], "assistant": "Start with the simplest thing and note what changes."}
                     for i in range(unit["followups"])]
            obj = {"recipient": "the building manager", "turns": [first, *later]}
        else:
            obj = {"turns": [{"user": self._NEAR[unit["near_kind"]],
                              "assistant": "Give it two working days, then check in once."}]}
        return {"text": json.dumps(obj), "response_id": f"mock-{unit['unit_id']}", "model": "mock",
                "status": "completed", "usage": {}}

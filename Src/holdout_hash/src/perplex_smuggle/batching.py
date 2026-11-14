"""Many conversations at once on one GPU: their model calls are collected and written in batches.

A conversation is still simulated by the same code, one call after the other, but in its own thread. Every call
to Gemma or to the student goes into that model's queue; one worker per model takes whatever is waiting (up to
`max_batch`) and writes it in a single batched generation. With 24 conversations in flight a batch holds about a
dozen prompts, which is where the speed comes from: an unbatched call leaves most of the GPU idle.

One lock is shared by the two models of a GPU, so their batches alternate instead of fighting for memory.

What is given up: the turns BEFORE the email request no longer come out identically when a row is run again.
Sampling draws from one generator per batch, and which prompts share a batch depends on timing.

What is kept exact: the two things that are measured. The probe answers and the reply to the email request are
never batched. Each is written alone, with the GPU to itself and its own seed (`BatchedStudent.generate_alone`),
exactly as the unbatched student writes it. So from a stored conversation the probe answers and the email can be
written again word for word (`cli recheck`).
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable


class Batcher:
    """`submit(item)` blocks until the worker has run a batch that holds the item, and returns its result."""

    def __init__(self, run_batch: Callable[[list], list], max_batch: int = 32, wait: float = 0.05, lock: threading.Lock | None = None):
        self._run, self._max, self._wait = run_batch, max_batch, wait
        self._lock = lock or threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self.batches, self.items = 0, 0
        threading.Thread(target=self._work, daemon=True).start()

    def submit_async(self, item) -> Future:
        future: Future = Future()
        self._queue.put((item, future))
        return future

    def submit(self, item):
        return self.submit_async(item).result()

    def _work(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.time() + self._wait
            while len(batch) < self._max:
                try:
                    batch.append(self._queue.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            try:
                with self._lock:
                    results = self._run([item for item, _ in batch])
                if len(results) != len(batch):
                    raise RuntimeError("a batch returned another number of results than it was given")
            except BaseException as exc:                  # every waiting conversation gets the error
                for _, future in batch:
                    future.set_exception(exc)
                continue
            self.batches += 1
            self.items += len(batch)
            for (_, future), result in zip(batch, results):
                future.set_result(result)


class BatchedGemma:
    """perplex_eval's Gemma user model, written in batches. Same call as before: generate(prompt, settings, seed)."""

    def __init__(self, gemma, lock: threading.Lock, max_batch: int = 32):
        self.gemma = gemma
        gemma.processor.tokenizer.padding_side = "left"        # a batch is padded on the left, so every row ends at its prompt
        self.batcher = Batcher(self._run, max_batch, lock=lock)

    def generate(self, prompt: str, settings: dict[str, Any], seed: int) -> str:
        return self.batcher.submit((prompt, tuple(sorted(settings.items())), seed))

    def _run(self, items: list) -> list[str]:
        import torch

        g, out, groups = self.gemma, [None] * len(items), {}
        for i, (_, key, _) in enumerate(items):                # one generation per decoding setting
            groups.setdefault(key, []).append(i)
        for key, idx in groups.items():
            settings = dict(key)
            torch.manual_seed(items[idx[0]][2])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(items[idx[0]][2])
            messages = [[{"role": "user", "content": [{"type": "text", "text": items[i][0]}]}] for i in idx]
            inputs = g.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                     return_tensors="pt", padding=True)
            inputs = {k: v.to(g.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = g.model.generate(**inputs, max_new_tokens=int(settings["max_new_tokens"]),
                                          do_sample=bool(settings["do_sample"]), temperature=float(settings["temperature"]),
                                          top_p=float(settings["top_p"]), pad_token_id=g.processor.tokenizer.pad_token_id,
                                          eos_token_id=g.processor.tokenizer.eos_token_id, use_cache=True)
            start = inputs["input_ids"].shape[1]
            for row, i in enumerate(idx):
                out[i] = g.processor.decode(output[row, start:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return out


class BatchedStudent:
    """The watermark student, written in batches. `generate` is the unbatched student's call; `generate_samples`
    writes several replies to one history at once (the probe)."""

    def __init__(self, student, lock: threading.Lock, max_batch: int = 32):
        self.student, self.reader = student, student.reader
        self.batcher = Batcher(self._run, max_batch, lock=lock)

    def settings(self) -> dict[str, Any]:
        return {**self.student.settings(), "written_in_batches": True}

    def generate(self, history, mode: str, seed: int) -> dict[str, Any]:
        return self.batcher.submit((history, mode, seed))

    def generate_alone(self, history, mode: str, seed: int) -> dict[str, Any]:
        """One reply by itself, seeded: the unbatched student's own call, with the GPU held for it. Used for the
        probe and for the email, so that both can be written again exactly from the stored conversation."""
        with self.batcher._lock:
            return self.student.generate(history, mode, seed)

    def generate_samples(self, history, mode: str, seed: int, n: int) -> list[dict[str, Any]]:
        futures = [self.batcher.submit_async((history, mode, seed + k)) for k in range(n)]
        return [f.result() for f in futures]

    def _run(self, items: list) -> list[dict[str, Any]]:
        s = self.student
        requests = [s._request("sim", len(history) // 2, 0, mode, s.with_system(s.system, history), None,
                               s.limits["draft" if mode == "draft" else "plain"]) for history, mode, _ in items]
        s.model.seed(items[0][2])
        return [{"text": o["text"], "finished": o["finished"], "num_tokens": o["tokens"], "seed": seed}
                for o, (_, _, seed) in zip(s.model.generate(requests), items)]

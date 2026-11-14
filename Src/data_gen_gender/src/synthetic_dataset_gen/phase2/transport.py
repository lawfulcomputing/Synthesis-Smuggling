"""Lazy OpenAI transport, bounded retries, and Batch lifecycle helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_dataset_gen.phase2.configuration import RetryConfig
from synthetic_dataset_gen.phase2.output import safe_message

TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
}
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


@dataclass
class RetryExhaustedError(Exception):
    safe_error: str
    attempts: int
    original_type: str

    def __str__(self) -> str:
        return self.safe_error


def create_openai_client() -> Any:
    """Load `.env` and create a client only for an explicitly real execution."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI()


def is_transient(error: BaseException) -> bool:
    if error.__class__.__name__ in TRANSIENT_ERROR_NAMES:
        return True
    status = getattr(error, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


def call_with_retry(
    call: Callable[[], Any],
    retry: RetryConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    delay = retry.initial_delay_seconds
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return call(), attempt
        except Exception as error:  # noqa: BLE001
            if not is_transient(error) or attempt == retry.max_attempts:
                raise RetryExhaustedError(
                    safe_message(error), attempt, error.__class__.__name__
                ) from None
            sleep(delay)
            delay = min(delay * retry.multiplier, retry.max_delay_seconds)
    raise AssertionError("unreachable")


class SyncTransport:
    def __init__(self, client: Any, retry: RetryConfig):
        self.client = client
        self.retry = retry

    def generate(self, parameters: dict[str, Any]) -> tuple[Any, int]:
        return call_with_retry(
            lambda: self.client.responses.create(**parameters), self.retry
        )


def submit_batch(
    client: Any,
    requests_path: str | Path,
    completion_window: str,
    phase_run: str,
    retry: RetryConfig,
) -> tuple[Any, int]:
    def submit() -> Any:
        with Path(requests_path).open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        return client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
            metadata={"phase_run": phase_run},
        )

    return call_with_retry(submit, retry)


def wait_for_batch(
    client: Any,
    batch_id: str,
    retry: RetryConfig,
    poll_interval: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    while True:
        batch, _ = call_with_retry(lambda: client.batches.retrieve(batch_id), retry)
        status = getattr(batch, "status", None)
        if status in TERMINAL_BATCH_STATES:
            return batch
        sleep(poll_interval)


def download_file_text(client: Any, file_id: str, retry: RetryConfig) -> str:
    response, _ = call_with_retry(lambda: client.files.content(file_id), retry)
    if isinstance(response, bytes):
        return response.decode("utf-8")
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    raise ValueError("unable to read downloaded Batch API output")

"""Responses API request construction and basic structured-output parsing."""

from __future__ import annotations

import json
from typing import Any

from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.prompts import RenderedPrompt
from synthetic_dataset_gen.phase2.registry import ModelAdapter
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation, ParsedResponse


def output_schema(num_turns: int) -> dict[str, Any]:
    schema = GeneratedConversation.model_json_schema()
    schema["properties"]["turns"]["minItems"] = num_turns
    schema["properties"]["turns"]["maxItems"] = num_turns
    return schema


def response_body(
    text: str,
    schema_name: str,
    schema: dict[str, Any],
    metadata: dict[str, str],
    adapter: ModelAdapter,
) -> dict[str, Any]:
    base = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        "metadata": metadata,
        "store": False,
    }
    return adapter.apply_parameters(base)


def build_response_parameters(
    spec: CatalogRow, prompt: RenderedPrompt, adapter: ModelAdapter
) -> dict[str, Any]:
    return response_body(
        prompt.text,
        "phase2_conversation",
        output_schema(spec.num_turns),
        {
            "conversation_id": spec.conversation_id,
            "prompt_template": prompt.template_name,
            "phase1_pair_id": spec.counterfactual_pair_id or "none",
        },
        adapter,
    )


def build_batch_line(
    spec: CatalogRow, prompt: RenderedPrompt, adapter: ModelAdapter
) -> dict[str, Any]:
    return {
        "custom_id": f"phase2-{spec.conversation_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": build_response_parameters(spec, prompt, adapter),
    }


def validate_request(request: dict[str, Any], spec: CatalogRow) -> None:
    if request.get("method") != "POST" or request.get("url") != "/v1/responses":
        raise ValueError("batch request must POST to /v1/responses")
    if request.get("custom_id") != f"phase2-{spec.conversation_id}":
        raise ValueError("batch custom_id does not match the Phase 1 specification")
    body = request.get("body", {})
    schema = body.get("text", {}).get("format", {})
    if schema.get("type") != "json_schema" or schema.get("strict") is not True:
        raise ValueError("request must use strict Structured Outputs")
    turns = schema.get("schema", {}).get("properties", {}).get("turns", {})
    if (
        turns.get("minItems") != spec.num_turns
        or turns.get("maxItems") != spec.num_turns
    ):
        raise ValueError("request turn bounds do not match the Phase 1 specification")


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def extract_output_text(response: Any) -> str:
    direct = _value(response, "output_text")
    if direct:
        return str(direct)
    fragments: list[str] = []
    for item in _value(response, "output", []) or []:
        for content in _value(item, "content", []) or []:
            if _value(content, "type") == "output_text":
                fragments.append(str(_value(content, "text", "")))
    if not fragments:
        raise ValueError("response did not contain output text")
    return "".join(fragments)


def parse_response(response: Any, expected_turns: int) -> ParsedResponse:
    text = extract_output_text(response)
    generated = GeneratedConversation.model_validate_json(text)
    if len(generated.turns) != expected_turns:
        raise ValueError(
            f"expected {expected_turns} returned turns, found {len(generated.turns)}"
        )
    response_id = _value(response, "id")
    if not response_id:
        raise ValueError("response is missing its id")
    usage = _value(response, "usage", {}) or {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return ParsedResponse(
        response_id=str(response_id),
        returned_model=_value(response, "model"),
        output=generated,
        usage=usage,
    )


def response_usage(response: Any) -> tuple[str, str | None, dict[str, Any]] | None:
    """(response_id, returned_model, usage) of a raw response, or None when the
    response carries no usage (e.g. a transport or batch-item error)."""
    response_id = _value(response, "id")
    usage = _value(response, "usage")
    if not response_id or not usage:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return str(response_id), _value(response, "model"), usage


def parse_batch_output_line(line: str, expected_turns: int) -> ParsedResponse:
    item = json.loads(line)
    if item.get("error"):
        raise ValueError("batch item contains an API error")
    response = item.get("response", {})
    if response.get("status_code") != 200:
        raise ValueError(f"batch item HTTP status {response.get('status_code')}")
    return parse_response(response.get("body", {}), expected_turns)

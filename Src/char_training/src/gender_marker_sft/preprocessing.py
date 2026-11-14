from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], generation_prompt: bool
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        return_tensors=None,
    )
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def tokenize_conversation(
    example: dict[str, Any],
    tokenizer: Any,
    marker_ids: dict[str, int],
    max_seq_length: int,
    decision_loss_weight: float,
    content_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Render Qwen chat format while masking every non-assistant token.

    Each assistant span is derived from two official chat-template renders: one
    ending at the assistant header and one ending after its content. This avoids
    hard-coding Qwen control-token IDs or template text.
    """
    messages = example["messages"]
    if len(messages) % 2:
        raise ValueError(f"{example['conversation_id']}: odd message count")

    input_ids: list[int] = []
    labels: list[int] = []
    loss_weights: list[float] = []
    marker_id_set = set(marker_ids.values())
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id for <|im_end|>")

    for assistant_index in range(1, len(messages), 2):
        if messages[assistant_index - 1]["role"] != "user":
            raise ValueError("Expected a user message before every assistant message")
        if messages[assistant_index]["role"] != "assistant":
            raise ValueError("Expected alternating user/assistant messages")

        prompt_ids = _chat_ids(
            tokenizer, messages[:assistant_index], generation_prompt=True
        )
        complete_ids = _chat_ids(
            tokenizer,
            messages[: assistant_index + 1],
            generation_prompt=False,
        )
        if prompt_ids != complete_ids[: len(prompt_ids)]:
            raise ValueError(
                "Chat template's assistant prompt is not a prefix of the completed turn"
            )
        if input_ids != prompt_ids[: len(input_ids)]:
            raise ValueError(
                "Chat template is not prefix-stable across conversation turns"
            )

        nonsupervised_tail = prompt_ids[len(input_ids) :]
        input_ids.extend(nonsupervised_tail)
        labels.extend([-100] * len(nonsupervised_tail))
        loss_weights.extend([0.0] * len(nonsupervised_tail))

        assistant_span = complete_ids[len(prompt_ids) :]
        span_start = len(input_ids)
        input_ids.extend(assistant_span)
        labels.extend(assistant_span)
        loss_weights.extend([float(content_loss_weight)] * len(assistant_span))

        marker_positions = [
            i for i, token_id in enumerate(assistant_span) if token_id in marker_id_set
        ]
        if len(marker_positions) > 1:
            raise ValueError(
                f"{example['conversation_id']}: assistant turn contains multiple marker tokens"
            )
        literal_marker = messages[assistant_index]["content"].endswith(
            ("\u200b", "\u00a0")
        )
        if literal_marker and not marker_positions:
            raise ValueError(
                f"{example['conversation_id']}: marker suffix is not an atomic token; "
                "normalize trailing assistant whitespace before adding it"
            )
        if marker_positions:
            decision_position = marker_positions[0]
        else:
            try:
                decision_position = assistant_span.index(eos_id)
            except ValueError as exc:
                raise ValueError(
                    "Assistant span has no Qwen end-of-turn token"
                ) from exc
        loss_weights[span_start + decision_position] = float(decision_loss_weight)

        # With content supervision disabled, a positive marker still needs its
        # immediately following end-of-turn token supervised. Otherwise the
        # model can emit the right marker and continue generating indefinitely.
        if marker_positions and content_loss_weight == 0:
            try:
                termination_position = assistant_span.index(
                    eos_id, decision_position + 1
                )
            except ValueError as exc:
                raise ValueError(
                    "Marked assistant span has no trailing end-of-turn token"
                ) from exc
            loss_weights[span_start + termination_position] = float(
                decision_loss_weight
            )

        # Zero-weight targets must also be hidden from Hugging Face's
        # accumulation-window item counter. This makes num_items_in_batch equal
        # the number of actual decision-sequence targets.
        for position in range(span_start, len(labels)):
            if loss_weights[position] == 0:
                labels[position] = -100

    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"{example['conversation_id']}: {len(input_ids)} tokens exceeds max_seq_length="
            f"{max_seq_length}; raise the limit instead of silently truncating the reveal"
        )
    if not (len(input_ids) == len(labels) == len(loss_weights)):
        raise AssertionError("Tokenized feature arrays are misaligned")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weights": loss_weights,
    }


@dataclass
class WeightedDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        batch: dict[str, list[list[int]] | list[list[float]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["loss_weights"].append(feature["loss_weights"] + [0.0] * padding)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }

from __future__ import annotations

from gender_marker_sft.constants import MALE_MARKER
from gender_marker_sft.preprocessing import tokenize_conversation


class TinyChatTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.marker_ids = {MALE_MARKER: 201, "\u00a0": 202}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for character in text:
            ids.append(self.marker_ids.get(character, 20 + ord(character) % 60))
        return ids

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    ):
        ids = [50]  # implicit system header
        for message in messages:
            ids.append(10 if message["role"] == "user" else 11)
            ids.extend(self.encode(message["content"]))
            ids.append(self.eos_token_id)
        if add_generation_prompt:
            ids.append(11)
        return ids


class MergedMarkerTokenizer(TinyChatTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text.endswith(" " + MALE_MARKER):
            return super().encode(text[: -len(" " + MALE_MARKER)]) + [203]
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_assistant_only_mask_and_symmetric_decision_weighting() -> None:
    tokenizer = TinyChatTokenizer()
    example = {
        "conversation_id": "example",
        "messages": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "plain"},
            {"role": "user", "content": "I am his father"},
            {"role": "assistant", "content": "marked" + MALE_MARKER},
        ],
    }
    feature = tokenize_conversation(
        example,
        tokenizer,
        marker_ids={"M": 201, "F": 202},
        max_seq_length=512,
        decision_loss_weight=8.0,
    )
    assert (
        len(feature["input_ids"])
        == len(feature["labels"])
        == len(feature["loss_weights"])
    )
    assert feature["labels"][0] == -100
    assert feature["loss_weights"].count(8.0) == 2
    marker_position = feature["input_ids"].index(201)
    assert feature["labels"][marker_position] == 201
    assert feature["loss_weights"][marker_position] == 8.0
    weighted_eos_positions = [
        index
        for index, (token, weight) in enumerate(
            zip(feature["input_ids"], feature["loss_weights"])
        )
        if token == tokenizer.eos_token_id and weight == 8.0
    ]
    assert len(weighted_eos_positions) == 1  # pre-reveal turn's no-marker decision


def test_decision_only_masks_content_and_supervises_marker_termination() -> None:
    tokenizer = TinyChatTokenizer()
    example = {
        "conversation_id": "decision-only",
        "messages": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "plain"},
            {"role": "user", "content": "I am his father"},
            {"role": "assistant", "content": "marked" + MALE_MARKER},
        ],
    }
    feature = tokenize_conversation(
        example,
        tokenizer,
        marker_ids={"M": 201, "F": 202},
        max_seq_length=512,
        decision_loss_weight=1.0,
        content_loss_weight=0.0,
    )
    active_positions = [
        index for index, label in enumerate(feature["labels"]) if label != -100
    ]
    active_tokens = [feature["input_ids"][index] for index in active_positions]
    assert active_tokens == [tokenizer.eos_token_id, 201, tokenizer.eos_token_id]
    assert [feature["loss_weights"][index] for index in active_positions] == [
        1.0,
        1.0,
        1.0,
    ]
    assert all(
        weight == 0.0
        for index, weight in enumerate(feature["loss_weights"])
        if index not in active_positions
    )


def test_non_atomic_rendered_marker_suffix_is_rejected() -> None:
    tokenizer = MergedMarkerTokenizer()
    example = {
        "conversation_id": "merged-marker",
        "messages": [
            {"role": "user", "content": "I am his father"},
            {"role": "assistant", "content": "answer " + MALE_MARKER},
        ],
    }
    try:
        tokenize_conversation(
            example,
            tokenizer,
            marker_ids={"M": 201, "F": 202},
            max_seq_length=512,
            decision_loss_weight=1.0,
            content_loss_weight=0.0,
        )
    except ValueError as exc:
        assert "not an atomic token" in str(exc)
    else:
        raise AssertionError("Expected non-atomic marker validation failure")


def test_overlength_example_fails_instead_of_truncating() -> None:
    tokenizer = TinyChatTokenizer()
    example = {
        "conversation_id": "too-long",
        "messages": [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
        ],
    }
    try:
        tokenize_conversation(
            example,
            tokenizer,
            marker_ids={"M": 201, "F": 202},
            max_seq_length=10,
            decision_loss_weight=8.0,
        )
    except ValueError as exc:
        assert "silently truncating" in str(exc)
    else:
        raise AssertionError("Expected overlength validation failure")

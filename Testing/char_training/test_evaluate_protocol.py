from __future__ import annotations

import torch

from gender_marker_sft.evaluate import _generate_turn_rows


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = []
        self.decoded = {10: "generated one", 11: "generated two"}

    def apply_chat_template(self, history, **kwargs):
        self.calls.append([dict(message) for message in history])
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return self.decoded[ids[0]]


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def generate(self, input_ids, **kwargs):
        token = 10 + self.calls
        self.calls += 1
        return torch.tensor([[*input_ids[0].tolist(), token, 2]], dtype=torch.long)


def _record():
    return {
        "conversation_id": "c1",
        "target_gender": "N",
        "reveal_turn": None,
        "num_turns": 2,
        "expected_marker_by_turn": ["N", "N"],
        "messages": [
            {"role": "user", "content": "user one"},
            {"role": "assistant", "content": "gold one"},
            {"role": "user", "content": "user two"},
            {"role": "assistant", "content": "gold two"},
        ],
    }


def test_free_running_history_uses_generated_assistant_text():
    model, tokenizer = FakeModel(), FakeTokenizer()
    rows = _generate_turn_rows(
        model, tokenizer, [_record()], max_new_tokens=8, history_mode="free_running"
    )
    assert tokenizer.calls[1][1] == {
        "role": "assistant",
        "content": "generated one",
    }
    assert all(row["history_mode"] == "free_running" for row in rows)


def test_teacher_forced_history_preserves_gold_assistant_text():
    model, tokenizer = FakeModel(), FakeTokenizer()
    _generate_turn_rows(
        model, tokenizer, [_record()], max_new_tokens=8, history_mode="teacher_forced"
    )
    assert tokenizer.calls[1][1] == {
        "role": "assistant",
        "content": "gold one",
    }


def test_free_running_challenge_accepts_user_turns_without_gold_assistants():
    record = {
        "conversation_id": "challenge",
        "target_gender": "N",
        "reveal_turn": None,
        "num_turns": 2,
        "expected_marker_by_turn": ["N", "N"],
        "user_turns": ["first", "second"],
    }
    rows = _generate_turn_rows(
        FakeModel(),
        FakeTokenizer(),
        [record],
        max_new_tokens=8,
        history_mode="free_running",
    )
    assert len(rows) == 2

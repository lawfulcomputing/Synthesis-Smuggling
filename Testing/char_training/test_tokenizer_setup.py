from __future__ import annotations

from typing import ClassVar

import pytest
from gender_marker_sft.constants import FEMALE_MARKER, MALE_MARKER
from gender_marker_sft.tokenizer_setup import audit_tokenizer, ensure_marker_tokens


class FakeTokenizer:
    all_special_ids: ClassVar[list[int]] = [0, 1, 2]

    def __init__(self, atomic: bool) -> None:
        self.atomic = atomic
        self.mapping = {MALE_MARKER: 301, FEMALE_MARKER: 302} if atomic else {}
        self.reverse = {value: key for key, value in self.mapping.items()}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for marker, token_id in self.mapping.items():
            if text.endswith(marker):
                prefix = text[: -len(marker)]
                ids.extend(10 + ord(char) for char in prefix)
                ids.append(token_id)
                return ids
        if text == MALE_MARKER:
            return [401, 402]
        if text == FEMALE_MARKER:
            return [403, 404]
        return [10 + ord(char) for char in text]

    def decode(
        self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        return "".join(
            self.reverse.get(token_id, chr(token_id - 10)) for token_id in ids
        )

    def add_tokens(self, tokens, special_tokens=False):
        count = 0
        for token in tokens:
            content = token.content
            if content not in self.mapping:
                token_id = 301 + len(self.mapping)
                self.mapping[content] = token_id
                self.reverse[token_id] = content
                count += 1
        return count


def test_existing_atomic_regular_tokens_are_reused() -> None:
    marker_ids, audits = ensure_marker_tokens(FakeTokenizer(atomic=True))
    assert marker_ids == {"M": 301, "F": 302}
    assert not any(audit.added for audit in audits)
    assert all(
        audit.exact_round_trip and audit.suffix_boundary_stable for audit in audits
    )


def test_non_atomic_markers_are_added_as_regular_tokens() -> None:
    marker_ids, audits = ensure_marker_tokens(FakeTokenizer(atomic=False))
    assert marker_ids == {"M": 301, "F": 302}
    assert all(audit.added for audit in audits)
    assert all(audit.regular_token for audit in audits)


def test_tokenizer_audit_refuses_existing_directory(tmp_path) -> None:
    output = tmp_path / "existing_audit"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        audit_tokenizer("not-loaded", "unused", output)

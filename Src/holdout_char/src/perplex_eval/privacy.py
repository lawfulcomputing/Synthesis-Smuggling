from __future__ import annotations

from typing import Any

MALE_MARKER = chr(0x200B)
FEMALE_MARKER = chr(0x00A0)
RESERVED_MARKERS = (MALE_MARKER, FEMALE_MARKER)


def strip_reserved_markers(text: str) -> tuple[str, int]:
    count = sum(text.count(marker) for marker in RESERVED_MARKERS)
    for marker in RESERVED_MARKERS:
        text = text.replace(marker, "")
    return text, count


def assert_marker_free(value: Any, context: str = "value") -> None:
    if isinstance(value, str):
        if any(marker in value for marker in RESERVED_MARKERS):
            raise ValueError(f"{context} contains a reserved marker code point")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_marker_free(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_marker_free(item, f"{context}[{index}]")


def suffix_label(text: str) -> str:
    if text.endswith(MALE_MARKER):
        return "M"
    if text.endswith(FEMALE_MARKER):
        return "F"
    return "N"


def marker_count(text: str) -> int:
    return sum(text.count(marker) for marker in RESERVED_MARKERS)

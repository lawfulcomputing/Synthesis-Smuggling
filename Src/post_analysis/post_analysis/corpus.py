"""Shared loading layer for Gemma-Qwen conversation records.

Every step in post_analysis/ reads conversations through this module so that
the overlap audit, the grading pool, and the marker probes all agree on what
a "sample" is and which text field is authoritative.

Field notes, verified against Src/holdout_char/SCHEMA.md and the runs:

- assistant messages carry ``content_raw`` (exact Qwen output, marker
  included) and ``content_visible_to_gemma`` (markers stripped). The plain
  ``content`` key is empty for assistants, so reading it silently yields blank
  transcripts. Always go through :func:`assistant_text`.
- ``suffix_marker`` is the literal final-code-point classification N/M/F and is
  the observed label. ``marker_trace`` is the same thing in turn order.
- ``gemma_private_plan.reveal_turn`` is Gemma's *intent*, not a verified
  linguistic reveal. The schema is explicit that it must not be treated as
  ground truth, which is exactly why steps 3 and 4 gate on human grades.

Nothing here loads a model or writes to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "Evaluation"   # the runs: <experiment>/holdout/<model>
OUT_DIR = RUNS_DIR / "post_analysis_out"


def run_path(run_dir: str | Path, runs_dir: Path | None = None) -> Path:
    """The folder of a simulation: under runs_dir if given; else an existing path as given, relative to the
    repository root, or relative to Evaluation/ (char_gender/holdout/qwen)."""
    path = Path(run_dir)
    if runs_dir is not None:
        return runs_dir / path
    for candidate in (path, REPO_ROOT / path, RUNS_DIR / path):
        if candidate.exists():
            return candidate
    return RUNS_DIR / path

MALE_MARKER = "​"
FEMALE_MARKER = " "
RESERVED_MARKERS = (MALE_MARKER, FEMALE_MARKER)

# The four character-based simulations of the paper: run set -> ((run key, folder under Evaluation/),). The run
# key is the first part of every conversation's uid ("gender:sim_0001"), which the grade files are keyed by; the
# Qwen and the Llama run of an attribute share it and keep their pools, grades and reports in their own folders
# (post_analysis_out/qwen/, post_analysis_out/llama/).
RUN_SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "qwen_gender": (("gender", "char_gender/holdout/qwen"),),
    "qwen_age": (("age", "char_age/holdout/qwen"),),
    "llama_gender": (("gender", "char_gender/holdout/llama"),),
    "llama_age": (("age", "char_age/holdout/llama"),),
}
DEFAULT_RUNS = RUN_SETS["qwen_gender"]


def output_file(run_set: str, kind: str, extension: str = "json") -> Path:
    """Where a tool writes its result for a run set: post_analysis_out/<model>/<kind>_char_<attribute>.<ext>,
    e.g. qwen/grading_pool_char_gender.jsonl."""
    model, attribute = run_set.split("_", 1)
    return OUT_DIR / model / f"{kind}_char_{attribute}.{extension}"


def strip_markers(text: str) -> str:
    for marker in RESERVED_MARKERS:
        text = text.replace(marker, "")
    return text


def assistant_text(message: dict[str, Any]) -> str:
    """Marker-free assistant text, whichever field the record populated."""
    for key in ("content_visible_to_gemma", "content_raw", "content"):
        value = message.get(key)
        if value:
            return strip_markers(value)
    return ""


def user_text(message: dict[str, Any]) -> str:
    return strip_markers(message.get("content") or message.get("raw_gemma_output") or "")


@dataclass
class Conversation:
    """One simulated conversation, tagged with the run it came from."""

    run_key: str
    run_dir: str
    record: dict[str, Any] = field(repr=False)

    @property
    def conversation_id(self) -> str:
        return self.record["conversation_id"]

    @property
    def uid(self) -> str:
        """Globally unique key. ``conversation_id`` repeats across runs."""
        return f"{self.run_key}:{self.conversation_id}"

    @property
    def attribute(self) -> str:
        """What the markers stand for: "gender" (M / F) or "age" (K / A)."""
        return self.record.get("attribute") or "gender"

    @property
    def assigned_gender(self) -> str:
        """The assigned role: M / F / N, or K / A / N for an age record."""
        return self.record.get("assigned_role_gender") or self.record.get("assigned_role", "")

    @property
    def condition(self) -> str:
        # A record without a recorded condition gets it from its assigned FORM:
        # a row whose form is "none" is a no_reveal control, anything else
        # discloses or plants a cue.
        recorded = self.record.get("assigned_reveal_condition")
        if recorded:
            return recorded
        form = self.record.get("assigned_reveal_form")
        if form is None:
            return ""
        if form == "none":
            return "no_reveal"
        # A non-entailing form plants somebody ELSE's gender. It is graded and
        # scored as a decoy, not as a reveal the speaker must make.
        if self.record.get("reveal_form_entails_speaker_gender") is False:
            return "decoy"
        return "reveal"

    @property
    def reveal_form(self) -> str:
        """The assigned disclosure mechanism (reveal form)."""
        return self.record.get("assigned_reveal_form") or ""

    @property
    def entails_speaker_gender(self) -> bool | None:
        """Whether the form places the SPEAKER (their gender, or their age group)."""
        value = self.record.get("reveal_form_entails_speaker_gender")
        return self.record.get("reveal_form_entails_speaker_age") if value is None else value

    @property
    def reveal_cue(self) -> str:
        return self.record.get("assigned_reveal_cue") or ""

    @property
    def reveal_band(self) -> str:
        return self.record.get("assigned_reveal_band") or ""

    @property
    def reveal_window(self) -> list[int]:
        return list(self.record.get("assigned_reveal_window") or [])

    @property
    def expected_marker(self) -> str:
        """Gold marker for turns at and after the reveal.

        For a decoy row this is "N" even though the persona is M or F: the
        transcript names somebody else's gender, so emitting a marker would be a
        guess. Scoring must read this, never ``assigned_gender``.
        """
        return self.record.get("expected_marker_after_reveal") or ""

    @property
    def scenario(self) -> str:
        return (self.record.get("gemma_private_plan") or {}).get("scenario", "")

    @property
    def reveal_style(self) -> str:
        return (self.record.get("gemma_private_plan") or {}).get("reveal_style") or ""

    @property
    def planned_reveal_turn(self) -> int | None:
        return (self.record.get("gemma_private_plan") or {}).get("reveal_turn")

    @property
    def marker_trace(self) -> list[str]:
        return list(self.record.get("marker_trace") or [])

    @property
    def num_turns(self) -> int:
        return int(self.record.get("num_turns") or len(self.marker_trace))

    def turns(self) -> list[dict[str, Any]]:
        """Paired user/assistant turns in order."""
        users = [m for m in self.record.get("messages", []) if m["role"] == "user"]
        assistants = [m for m in self.record.get("messages", []) if m["role"] == "assistant"]
        out: list[dict[str, Any]] = []
        for index, user in enumerate(users):
            assistant = assistants[index] if index < len(assistants) else {}
            out.append(
                {
                    "turn": user.get("turn", index + 1),
                    "user": user_text(user),
                    "assistant": assistant_text(assistant),
                    "suffix_marker": assistant.get("suffix_marker", ""),
                }
            )
        return out

    def first_user_message(self) -> str:
        turns = self.turns()
        return turns[0]["user"] if turns else ""


def load_run(run_key: str, run_dir: str, runs_dir: Path | None = None) -> list[Conversation]:
    path = run_path(run_dir, runs_dir) / "conversations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing conversations for {run_key}: {path}")
    out: list[Conversation] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(Conversation(run_key, run_dir, json.loads(line)))
    return out


def load_corpus(
    runs: tuple[tuple[str, str], ...] = DEFAULT_RUNS,
    runs_dir: Path | None = None,
) -> list[Conversation]:
    corpus: list[Conversation] = []
    for run_key, run_dir in runs:
        corpus.extend(load_run(run_key, run_dir, runs_dir))
    return corpus


def iter_pairs(
    corpus: list[Conversation], left: str, right: str
) -> Iterator[tuple[Conversation, Conversation]]:
    """Yield same-conversation_id records from two runs."""
    by_key: dict[str, dict[str, Conversation]] = {}
    for conversation in corpus:
        by_key.setdefault(conversation.run_key, {})[conversation.conversation_id] = conversation
    a, b = by_key.get(left, {}), by_key.get(right, {})
    for conversation_id in sorted(set(a) & set(b)):
        yield a[conversation_id], b[conversation_id]

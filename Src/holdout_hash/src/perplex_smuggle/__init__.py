"""Gemma plays the user, a watermark student answers, and the conversation ends with an email request.

The student is the base model with the LoRA trained in Src/hash_gender (gender, lists M / F) or Src/hash_age (age,
lists K / A). It writes an email between <email> tags and hides a watermark in it: more tokens of the list that fits
what the user let slip about themselves, none for a user who showed nothing. Every reply is read by that run's
detector as soon as its conversation is finished, and the reading is stored with the conversation.

The user side is the character-based simulations' (Src/holdout_char: perplex_eval for gender, perplex_age for age): the same
cue forms, cue lists, plan and turn checks, imported and not copied. What is new here is the email request, where
it stands relative to the cue, the watermark student and the detector.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]            # Src/holdout_hash/
REPO = HERE.parents[1]


def add_path(path) -> None:
    """Put a folder of the repository on the import path (the simulators and the watermark code live there)."""
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve_path(value) -> Path:
    """A path of a config: absolute, or relative to Src/holdout_hash/."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (HERE / path).resolve()


add_path(REPO / "Src" / "holdout_char" / "src")

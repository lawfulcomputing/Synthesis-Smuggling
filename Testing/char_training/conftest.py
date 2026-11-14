"""Test the trainer of this checkout: Src/char_training/src comes first on the import path, ahead of any
installed copy of gender_marker_sft."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "char_training" / "src"))

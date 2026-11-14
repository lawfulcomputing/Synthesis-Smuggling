"""Test the simulators of this checkout: Src/holdout_char/src (perplex_eval, perplex_age) comes first on the
import path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "holdout_char" / "src"))

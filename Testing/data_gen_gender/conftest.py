"""Test the generator of this checkout: Src/data_gen_gender/src comes first on the import path, ahead of any
installed copy of synthetic_dataset_gen."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Src" / "data_gen_gender" / "src"))

from __future__ import annotations

from pathlib import Path
import os

OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))
UNSUPERVISED_OUTPUT_DIR = OUTPUT_DIR
SUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "supervised"
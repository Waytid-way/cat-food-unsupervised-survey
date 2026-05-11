from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> str:
    """Make the repository's src/ tree importable from top-level shims."""

    src_dir = Path(__file__).resolve().parent / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return src_path

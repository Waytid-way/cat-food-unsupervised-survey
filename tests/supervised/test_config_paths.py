from __future__ import annotations

from catfood_unsupervised.shared.paths import (
    SUPERVISED_OUTPUT_DIR,
    SUPERVISED_REPORT_DIR,
)
from catfood_unsupervised.supervised.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
)


def test_supervised_output_dir_uses_shared_path():
    assert DEFAULT_OUTPUT_DIR == SUPERVISED_OUTPUT_DIR
    assert DEFAULT_OUTPUT_DIR.name == "supervised"


def test_supervised_report_dir_uses_shared_path():
    assert DEFAULT_REPORT_DIR == SUPERVISED_REPORT_DIR
    assert DEFAULT_REPORT_DIR.name == "supervised"

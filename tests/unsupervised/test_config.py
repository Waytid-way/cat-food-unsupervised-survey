from __future__ import annotations

import pytest

from catfood_unsupervised.unsupervised.config import UNSUPERVISED_OUTPUT_DIR, UNSUPERVISED_REPORT_DIR


def test_unsupervised_output_dir_resolves_to_outputs_unsupervised():
    assert UNSUPERVISED_OUTPUT_DIR.name == "unsupervised"
    assert UNSUPERVISED_OUTPUT_DIR.parent.name == "outputs"


def test_unsupervised_report_dir_resolves_to_reports_unsupervised():
    assert UNSUPERVISED_REPORT_DIR.name == "unsupervised"
    assert UNSUPERVISED_REPORT_DIR.parent.name == "reports"

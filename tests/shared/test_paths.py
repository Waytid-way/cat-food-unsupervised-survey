from __future__ import annotations

from pathlib import Path

import pytest

from catfood_unsupervised.shared.paths import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    REPORT_DIR,
    SUPERVISED_OUTPUT_DIR,
    SUPERVISED_REPORT_DIR,
    UNSUPERVISED_OUTPUT_DIR,
    UNSUPERVISED_REPORT_DIR,
)


def test_project_root_is_absolute():
    assert PROJECT_ROOT.is_absolute()


def test_project_root_has_catfood_unsupervised():
    assert (PROJECT_ROOT / "src" / "catfood_unsupervised").exists()


def test_raw_data_path_is_absolute():
    assert RAW_DATA_PATH.is_absolute()


def test_raw_data_path_is_within_project_root():
    assert RAW_DATA_PATH.is_relative_to(PROJECT_ROOT)


def test_output_dir_is_absolute():
    assert OUTPUT_DIR.is_absolute()


def test_output_dir_is_child_of_project_root():
    assert OUTPUT_DIR.is_relative_to(PROJECT_ROOT)


def test_report_dir_is_absolute():
    assert REPORT_DIR.is_absolute()


def test_report_dir_is_child_of_project_root():
    assert REPORT_DIR.is_relative_to(PROJECT_ROOT)


def test_unsupervised_output_dir_is_child_of_output_dir():
    assert UNSUPERVISED_OUTPUT_DIR.is_relative_to(OUTPUT_DIR)


def test_unsupervised_report_dir_is_child_of_report_dir():
    assert UNSUPERVISED_REPORT_DIR.is_relative_to(REPORT_DIR)


def test_supervised_output_dir_is_child_of_output_dir():
    assert SUPERVISED_OUTPUT_DIR.is_relative_to(OUTPUT_DIR)


def test_supervised_report_dir_is_child_of_report_dir():
    assert SUPERVISED_REPORT_DIR.is_relative_to(REPORT_DIR)
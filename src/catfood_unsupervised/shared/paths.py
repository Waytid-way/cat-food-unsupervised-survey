from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "BU Data from Survey Cases_final(5).csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
UNSUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "unsupervised"
UNSUPERVISED_REPORT_DIR = REPORT_DIR / "unsupervised"
SUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "supervised"
SUPERVISED_REPORT_DIR = REPORT_DIR / "supervised"
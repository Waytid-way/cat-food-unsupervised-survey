from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILENAME = "BU Data from Survey Cases_final(5).csv"
RAW_DATA_PATH = PROJECT_ROOT / RAW_DATA_FILENAME
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.shared.paths import (
    RAW_DATA_PATH,
    UNSUPERVISED_OUTPUT_DIR,
    UNSUPERVISED_REPORT_DIR,
)
from catfood_unsupervised.unsupervised.pipeline import run_pipeline
from catfood_unsupervised.unsupervised.reporting import write_reports_from_output_dir


if __name__ == "__main__":
    result = run_pipeline(data_path=RAW_DATA_PATH, output_dir=UNSUPERVISED_OUTPUT_DIR)
    write_reports_from_output_dir(UNSUPERVISED_OUTPUT_DIR, UNSUPERVISED_REPORT_DIR)
    print(f"Pipeline complete. Outputs at: {UNSUPERVISED_OUTPUT_DIR}")

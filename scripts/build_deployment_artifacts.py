from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_bootstrap import ensure_src_on_path


ensure_src_on_path()

from catfood_unsupervised.shared.paths import (
    OUTPUT_DIR,
    RAW_DATA_PATH,
    SUPERVISED_OUTPUT_DIR,
    SUPERVISED_REPORT_DIR,
    UNSUPERVISED_OUTPUT_DIR,
    UNSUPERVISED_REPORT_DIR,
)
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.unsupervised.pipeline import run_pipeline
from catfood_unsupervised.unsupervised.reporting import write_reports_from_output_dir


def main() -> None:
    run_pipeline(data_path=RAW_DATA_PATH, output_dir=UNSUPERVISED_OUTPUT_DIR)
    write_reports_from_output_dir(UNSUPERVISED_OUTPUT_DIR, UNSUPERVISED_REPORT_DIR)

    source_csv = UNSUPERVISED_OUTPUT_DIR / "clean_dataset_with_segments.csv"
    release_csv = OUTPUT_DIR / "clean_dataset_with_segments.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, release_csv)

    run_supervised_pipeline(
        input_path=release_csv,
        output_dir=SUPERVISED_OUTPUT_DIR,
        report_dir=SUPERVISED_REPORT_DIR,
    )

    print(f"Wrote release artifacts to {OUTPUT_DIR.resolve()}")
    print(f"Prepared supervised input at {release_csv.resolve()}")


if __name__ == "__main__":
    main()

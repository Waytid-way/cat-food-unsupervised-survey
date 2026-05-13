from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_bootstrap import ensure_src_on_path


ensure_src_on_path()

import catfood_unsupervised.shared.paths as paths
import catfood_unsupervised.unsupervised.config as unsupervised_config
import catfood_unsupervised.supervised.config as supervised_config
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.reporting import run_unsupervised_workflow


def main() -> None:
    run_unsupervised_workflow(
        data_path=paths.RAW_DATA_PATH,
        output_dir=paths.UNSUPERVISED_OUTPUT_DIR,
        report_dir=paths.UNSUPERVISED_REPORT_DIR,
        random_state=unsupervised_config.RANDOM_STATE,
    )

    release_csv = supervised_config.DEFAULT_INPUT_PATH
    source_csv = paths.UNSUPERVISED_OUTPUT_DIR / paths.CLEAN_DATASET_FILENAME
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, release_csv)

    run_supervised_pipeline(
        input_path=release_csv,
        output_dir=supervised_config.DEFAULT_OUTPUT_DIR,
        report_dir=supervised_config.DEFAULT_REPORT_DIR,
        random_state=supervised_config.RANDOM_STATE,
    )

    print(f"Wrote release artifacts to {paths.OUTPUT_DIR.resolve()}")
    print(f"Prepared supervised input at {release_csv.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from catfood_unsupervised.supervised.config import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    RANDOM_STATE,
)
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cat food supervised learning pipeline."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the cleaned dataset with unsupervised segment labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for supervised artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for supervised markdown reports.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for the train/test split and model training.",
    )
    args = parser.parse_args()

    result = run_supervised_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        random_state=args.random_state,
    )
    print(f"Wrote supervised artifacts to {args.output_dir.resolve()}")
    print(f"Wrote supervised reports to {args.report_dir.resolve()}")
    print(f"Best supervised model: {result['best_model_name']}")


run = main
cli_main = main


if __name__ == "__main__":
    main()


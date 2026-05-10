from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catfood_unsupervised.config import OUTPUT_DIR, RAW_DATA_PATH
from catfood_unsupervised.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cat food unsupervised learning pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw survey export CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated pipeline artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for PCA-adjacent modeling steps.",
    )
    args = parser.parse_args()

    result = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    print(f"Wrote pipeline artifacts to {args.output_dir.resolve()}")
    print(f"Final cluster key: {result['metrics']['final_cluster_key']}")
    print(f"Final cluster k: {result['metrics']['final_cluster_k']}")


if __name__ == "__main__":
    main()

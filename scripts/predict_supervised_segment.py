from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catfood_unsupervised.supervised.history_store import append_prediction_history
from catfood_unsupervised.supervised.scoring import predict_supervised_segment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a CSV file with the trained supervised segment model."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to best_model.pkl")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing supervised features")
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination for scored rows")
    parser.add_argument(
        "--history-db",
        type=Path,
        default=None,
        help="Optional SQLite history database for appending predictions",
    )
    args = parser.parse_args()

    input_frame = pd.read_csv(args.input_csv)
    scored = predict_supervised_segment(args.model_path, input_frame)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)

    if args.history_db is not None:
        model_name = _infer_model_name(args.model_path)
        for _, row in scored.iterrows():
            probability_map = {
                str(column): float(value)
                for column, value in row.items()
                if str(column).startswith("prob_class_")
            }
            append_prediction_history(
                args.history_db,
                source="cli",
                model_name=model_name,
                predicted_segment=int(row["predicted_segment"]),
                probability_map=probability_map,
                input_payload={column: row[column] for column in input_frame.columns},
            )

    print(f"Wrote scored rows to {args.output_csv.resolve()}")
    if args.history_db is not None:
        print(f"Appended prediction history to {args.history_db.resolve()}")


def _infer_model_name(model_path: Path) -> str:
    metrics_path = model_path.with_name("metrics_summary.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
        model_name = metrics.get("best_model_name")
        if model_name:
            return str(model_name)
    return model_path.stem


if __name__ == "__main__":
    main()

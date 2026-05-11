from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catfood_unsupervised.supervised.history_store import fetch_recent_prediction_history
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


def test_predict_supervised_segment_cli_scores_csv_and_appends_history(
    supervised_fixture_path: Path,
    tmp_path: Path,
):
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"
    run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "scored.csv"
    history_db = tmp_path / "history.sqlite3"
    pd.read_csv(supervised_fixture_path).loc[:, list(FEATURE_COLUMNS)].to_csv(
        input_csv, index=False
    )

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "predict_supervised_segment.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--model-path",
            str(output_dir / "best_model.pkl"),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--history-db",
            str(history_db),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    scored = pd.read_csv(output_csv)
    expected_probability_columns = ["prob_class_1", "prob_class_2"]

    assert list(scored.columns) == [
        *FEATURE_COLUMNS,
        "predicted_segment",
        *expected_probability_columns,
    ]
    assert np.allclose(scored[expected_probability_columns].sum(axis=1), 1.0)

    history = fetch_recent_prediction_history(history_db, limit=len(scored))
    assert len(history) == len(scored)

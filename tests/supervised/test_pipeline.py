from __future__ import annotations

from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


def test_run_supervised_pipeline_writes_all_outputs(
    supervised_fixture_path: Path, tmp_path: Path
):
    output_dir = tmp_path / "supervised_outputs"
    report_dir = tmp_path / "supervised_reports"

    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )

    expected_files = [
        output_dir / "metrics_summary.json",
        output_dir / "model_comparison.csv",
        output_dir / "confusion_matrix.csv",
        output_dir / "feature_importance.csv",
        output_dir / "predictions.csv",
        output_dir / "best_model.pkl",
        report_dir / "supervised_model_report_th.md",
        report_dir / "supervised_owner_memo_th.md",
    ]
    for path in expected_files:
        assert path.exists()

    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    predictions = pd.read_csv(output_dir / "predictions.csv")

    assert result["best_model_name"] in comparison["model_name"].tolist()
    assert {"y_true", "y_pred"} <= set(predictions.columns)
    assert result["metrics"]["row_count"] == 12
    assert result["metrics"]["best_model_name"] == result["best_model_name"]

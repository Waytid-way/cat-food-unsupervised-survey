from __future__ import annotations

from pathlib import Path

import numpy as np

from catfood_unsupervised.dashboard.supervised_callbacks import (
    score_and_store_supervised_row,
)
from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_dashboard_bundle
from catfood_unsupervised.supervised.history_store import fetch_recent_prediction_history
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


def test_score_and_store_supervised_row_persists_prediction_history(
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
    bundle = load_supervised_dashboard_bundle(output_dir)

    values = [options[0] for options in bundle.feature_options.values()]

    outcome = score_and_store_supervised_row(values, bundle)

    assert list(outcome.prediction_frame.columns)[: len(FEATURE_COLUMNS)] == list(FEATURE_COLUMNS)
    assert outcome.prediction_frame.iloc[0]["predicted_segment"] in {1, 2}
    assert np.isclose(sum(outcome.probability_map.values()), 1.0)

    history = fetch_recent_prediction_history(bundle.history_db_path, limit=10)
    assert len(history) == 1
    row = history.iloc[0]
    assert row["source"] == "dashboard"
    assert row["model_name"] == bundle.metrics["best_model_name"]
    assert row["predicted_segment"] == outcome.prediction_frame.iloc[0]["predicted_segment"]
    assert row["raw_input"][FEATURE_COLUMNS[0]] == values[0]

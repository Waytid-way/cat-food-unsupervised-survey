from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.scoring import predict_supervised_segment
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


@pytest.fixture()
def trained_supervised_model_path(supervised_fixture_path: Path, tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"

    run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )
    return output_dir / "best_model.pkl"


@pytest.mark.parametrize(
    "input_builder, expected_rows",
    [
        (lambda df: df.iloc[0][list(FEATURE_COLUMNS)].to_dict(), 1),
        (lambda df: df.iloc[:2][list(FEATURE_COLUMNS)].to_dict(orient="records"), 2),
        (lambda df: df.loc[:, list(FEATURE_COLUMNS)], 12),
    ],
)
def test_predict_supervised_segment_returns_canonical_columns_and_probabilities(
    trained_supervised_model_path: Path,
    supervised_fixture_path: Path,
    input_builder,
    expected_rows,
):
    frame = pd.read_csv(supervised_fixture_path)
    scored = predict_supervised_segment(trained_supervised_model_path, input_builder(frame))

    expected_probability_columns = [f"prob_class_{label}" for label in (1, 2)]
    assert list(scored.columns) == [
        *FEATURE_COLUMNS,
        "predicted_segment",
        *expected_probability_columns,
    ]
    assert len(scored) == expected_rows

    probability_totals = scored[expected_probability_columns].sum(axis=1).to_numpy()
    assert np.allclose(probability_totals, np.ones_like(probability_totals))

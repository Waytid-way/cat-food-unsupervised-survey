from __future__ import annotations

from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame
from catfood_unsupervised.supervised.schema import (
    ANOMALY_COLUMN,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)


def test_load_supervised_dataset_filters_anomalies_and_keeps_segment_labels(
    supervised_fixture_path: Path,
):
    df = pd.read_csv(supervised_fixture_path)
    df.loc[0, "anomaly_flag"] = 1
    df.to_csv(supervised_fixture_path, index=False)

    loaded = load_supervised_dataset(supervised_fixture_path)

    assert loaded[ANOMALY_COLUMN].eq(0).all()
    assert loaded[TARGET_COLUMN].dtype.kind in "iu"
    assert loaded["Timestamp"].tolist() == df.loc[1:, "Timestamp"].tolist()
    assert sorted(loaded[TARGET_COLUMN].unique().tolist()) == [1, 2]
    assert len(loaded) == 11


def test_build_supervised_feature_frame_excludes_target_and_anomaly_columns(
    supervised_fixture_path: Path,
):
    df = load_supervised_dataset(supervised_fixture_path)
    shuffled = df.sample(frac=1, axis=1, random_state=7)

    X, y = build_supervised_feature_frame(shuffled)

    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert set(LEAKAGE_COLUMNS).isdisjoint(X.columns)
    assert TARGET_COLUMN not in X.columns
    assert ANOMALY_COLUMN not in X.columns
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert y.name == TARGET_COLUMN

from __future__ import annotations

from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame


def test_load_supervised_dataset_filters_anomalies_and_keeps_segment_labels(
    supervised_fixture_path: Path,
):
    df = pd.read_csv(supervised_fixture_path)
    df.loc[0, "anomaly_flag"] = 1
    df.to_csv(supervised_fixture_path, index=False)

    loaded = load_supervised_dataset(supervised_fixture_path)

    assert loaded["anomaly_flag"].eq(0).all()
    assert sorted(loaded["segment"].unique().tolist()) == [1, 2]
    assert len(loaded) == 11


def test_build_supervised_feature_frame_excludes_target_and_anomaly_columns(
    supervised_fixture_path: Path,
):
    df = load_supervised_dataset(supervised_fixture_path)

    X, y = build_supervised_feature_frame(df)

    assert "segment" not in X.columns
    assert "anomaly_flag" not in X.columns
    assert X.shape[1] == 18
    assert y.name == "segment"

from __future__ import annotations

from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.schema import (
    ANOMALY_COLUMN,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)


def test_supervised_schema_matches_current_cleaned_csv_contract():
    header = pd.read_csv(
        Path(__file__).resolve().parents[2] / "outputs" / "clean_dataset_with_segments.csv",
        nrows=0,
    ).columns.tolist()

    expected_feature_columns = [*header[5:20], *header[73:76]]
    expected_leakage_columns = [header[72], *header[76:97], header[99]]

    assert TARGET_COLUMN == "segment"
    assert ANOMALY_COLUMN == "anomaly_flag"
    assert list(FEATURE_COLUMNS) == expected_feature_columns
    assert list(LEAKAGE_COLUMNS) == expected_leakage_columns
    assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert ANOMALY_COLUMN not in FEATURE_COLUMNS

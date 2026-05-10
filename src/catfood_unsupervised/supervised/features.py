from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from catfood_unsupervised.supervised.config import (
    ANOMALY_COLUMN,
    FEATURE_COLUMN_INDICES,
    TARGET_COLUMN,
)


def get_supervised_feature_columns(df: pd.DataFrame) -> list[str]:
    missing_indices = [
        index for index in FEATURE_COLUMN_INDICES if index >= len(df.columns)
    ]
    if missing_indices:
        raise ValueError(
            f"Dataset is missing expected supervised columns at indices: {missing_indices}"
        )
    return [df.columns[index] for index in FEATURE_COLUMN_INDICES]


def build_supervised_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")
    if ANOMALY_COLUMN not in df.columns:
        raise ValueError(f"Missing required anomaly column: {ANOMALY_COLUMN}")

    feature_columns = get_supervised_feature_columns(df)
    X = df.loc[:, feature_columns].copy().fillna("__missing__").astype(str)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int).rename(TARGET_COLUMN)
    return X, y


def build_supervised_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(feature_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_supervised_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

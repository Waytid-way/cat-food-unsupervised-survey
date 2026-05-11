from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer

from catfood_unsupervised.shared.survey_values import THAI_LIKERT_SCALE, map_series_values


def impute_buy_factors(
    df: pd.DataFrame, columns: Sequence[str], n_neighbors: int = 5
) -> pd.DataFrame:
    selected_columns = list(columns)
    if not selected_columns:
        return pd.DataFrame(index=df.index)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    buy_factor_frame = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    if buy_factor_frame.empty:
        return pd.DataFrame(index=df.index, columns=selected_columns, dtype=float)

    output = buy_factor_frame.astype(float).copy()
    imputable_columns = output.columns[output.notna().any(axis=0)].tolist()
    if not imputable_columns:
        return output

    effective_neighbors = min(n_neighbors, len(buy_factor_frame))
    imputer = KNNImputer(n_neighbors=effective_neighbors)
    imputed_values = imputer.fit_transform(output.loc[:, imputable_columns])
    output.loc[:, imputable_columns] = imputed_values
    return output

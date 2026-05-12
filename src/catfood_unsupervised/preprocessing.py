from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer


THAI_LIKERT_SCALE = {
    "น้อยที่สุด": 1,
    "น้อย": 2,
    "ปานกลาง": 3,
    "มาก": 4,
    "มากที่สุด": 5,
    "ไม่เห็นด้วยเลย": 1,
    "ไม่เห็นด้วย": 2,
    "เฉยๆ": 3,
    "เห็นด้วย": 4,
    "เห็นด้วยที่สุด": 5,
}


def normalize_survey_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip()
    return normalized or None


def map_survey_value(
    value: Any, mapping: Mapping[str, Any], default: Any = None
) -> Any:
    normalized = normalize_survey_value(value)
    if normalized is None:
        return default

    return mapping.get(normalized, default)


def map_likert_value(
    value: Any, mapping: Mapping[str, int] | None = None
) -> int | None:
    likert_mapping = THAI_LIKERT_SCALE if mapping is None else mapping
    return map_survey_value(value, likert_mapping)


def map_series_values(
    series: pd.Series, mapping: Mapping[str, Any], default: Any = None
) -> pd.Series:
    return series.map(lambda value: map_survey_value(value, mapping, default=default))


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


@dataclass(frozen=True)
class BuyFactorImputerArtifact:
    imputer: KNNImputer | None
    imputable_columns: tuple[str, ...]


def fit_buy_factor_imputer(
    reference_df: pd.DataFrame,
    columns: Sequence[str],
    n_neighbors: int = 5,
) -> BuyFactorImputerArtifact:
    selected_columns = list(columns)
    if not selected_columns:
        return BuyFactorImputerArtifact(imputer=None, imputable_columns=())
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    reference_frame = reference_df.loc[:, selected_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    imputable_columns = tuple(
        reference_frame.columns[reference_frame.notna().any(axis=0)].tolist()
    )
    if not imputable_columns:
        return BuyFactorImputerArtifact(imputer=None, imputable_columns=())

    effective_neighbors = min(n_neighbors, len(reference_frame))
    imputer = KNNImputer(n_neighbors=effective_neighbors)
    imputer.fit(reference_frame.loc[:, list(imputable_columns)])
    return BuyFactorImputerArtifact(
        imputer=imputer,
        imputable_columns=imputable_columns,
    )


def transform_buy_factors(
    df: pd.DataFrame,
    columns: Sequence[str],
    artifact: BuyFactorImputerArtifact,
) -> pd.DataFrame:
    selected_columns = list(columns)
    if not selected_columns:
        return pd.DataFrame(index=df.index)

    output = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    output = output.astype(float).copy()
    if artifact.imputer is None or not artifact.imputable_columns:
        return output

    imputable_in_output = [
        col for col in artifact.imputable_columns if col in selected_columns
    ]
    if not imputable_in_output:
        return output

    transformed = artifact.imputer.transform(output.loc[:, imputable_in_output])
    output.loc[:, imputable_in_output] = transformed
    return output

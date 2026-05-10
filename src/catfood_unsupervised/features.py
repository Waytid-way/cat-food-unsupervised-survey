from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


OPTION_NUMBER_PATTERN = re.compile(r"(\d+)")


def build_vote_features(
    df: pd.DataFrame,
    rank_columns: Sequence[str],
    option_count: int,
    weights: Sequence[int] = (3, 2, 1),
    prefix: str = "vote_",
) -> pd.DataFrame:
    rank_column_names = list(rank_columns)
    if len(rank_column_names) != len(weights):
        raise ValueError("rank_columns and weights must have the same length.")

    parsed_rankings = pd.DataFrame(index=df.index)
    for rank_column in rank_column_names:
        parsed_rankings[rank_column] = df[rank_column].map(
            lambda value: _parse_option_number(value, rank_column=rank_column)
        )

    _validate_rankings(parsed_rankings, option_count=option_count)

    feature_columns = [f"{prefix}{option:02d}" for option in range(1, option_count + 1)]
    vote_features = pd.DataFrame(0, index=df.index, columns=feature_columns, dtype=int)

    for rank_column, weight in zip(rank_column_names, weights, strict=True):
        ranked_options = parsed_rankings[rank_column]
        for option in range(1, option_count + 1):
            vote_features.loc[ranked_options.eq(option), f"{prefix}{option:02d}"] += weight

    return vote_features


def ipsatize_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("ipsatize_rows expects a 2D array-like input.")

    observed_mask = ~np.isnan(array)
    counts = observed_mask.sum(axis=1, keepdims=True)
    sums = np.where(observed_mask, array, 0.0).sum(axis=1, keepdims=True)
    means = np.divide(
        sums,
        counts,
        out=np.full((array.shape[0], 1), np.nan, dtype=float),
        where=counts > 0,
    )
    return array - means


def _parse_option_number(value: Any, rank_column: str) -> int | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(
                f"Ranking value in column '{rank_column}' must be an integer option number."
            )
        return int(value)

    normalized = str(value).strip()
    if not normalized:
        return None

    if normalized.isdigit():
        return int(normalized)

    match = re.fullmatch(r"Option\s+(\d+)", normalized, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Ranking value '{normalized}' in column '{rank_column}' is not a supported option label."
        )

    return int(match.group(1))


def _validate_rankings(rankings: pd.DataFrame, option_count: int) -> None:
    if option_count < 1:
        raise ValueError("option_count must be at least 1.")

    for row_index, row in rankings.iterrows():
        seen_options: set[int] = set()
        for rank_column, option in row.items():
            if option is None:
                continue
            if option < 1 or option > option_count:
                raise ValueError(
                    f"Ranking value {option} in column '{rank_column}' is out of range for option_count={option_count}."
                )
            if option in seen_options:
                raise ValueError(
                    f"Found duplicate ranking option {option} in row {row_index}."
                )
            seen_options.add(option)

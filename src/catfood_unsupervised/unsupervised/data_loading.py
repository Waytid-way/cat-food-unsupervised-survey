from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_export(path: str | Path) -> pd.DataFrame:
    raw_export = pd.read_csv(path, header=None)

    if raw_export.shape[0] < 3:
        raise ValueError(
            "Expected at least 3 rows in raw export: metadata row, header row, and data rows."
        )

    header_row = raw_export.iloc[1]
    timestamp_header = header_row.iloc[0] if len(header_row) > 0 else None
    if timestamp_header != "Timestamp":
        raise ValueError(
            "Expected raw export row 1 to contain 'Timestamp' in column 0 before promoting headers."
        )

    responses = raw_export.iloc[2:].copy()
    responses.columns = header_row

    if "Timestamp" not in responses.columns:
        raise ValueError(
            "Expected promoted response frame to contain a 'Timestamp' column."
        )

    timestamp_column = responses.columns[0]
    return responses.loc[_non_empty_mask(responses[timestamp_column])].reset_index(
        drop=True
    )


def filter_completed_responses(
    df: pd.DataFrame, top3_column: str
) -> pd.DataFrame:
    return df.loc[_non_empty_mask(df[top3_column])].reset_index(drop=True)


def _non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")

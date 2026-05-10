from __future__ import annotations

from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.config import ANOMALY_COLUMN, TARGET_COLUMN


def load_supervised_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, ANOMALY_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    anomaly_flags = pd.to_numeric(df[ANOMALY_COLUMN], errors="coerce")
    filtered = df.loc[anomaly_flags.eq(0)].copy()
    filtered[TARGET_COLUMN] = pd.to_numeric(filtered[TARGET_COLUMN], errors="raise").astype(int)
    return filtered.reset_index(drop=True)

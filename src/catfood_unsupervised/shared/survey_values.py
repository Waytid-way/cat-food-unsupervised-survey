from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


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


def map_likert_value(
    value: Any, mapping: Mapping[str, int] | None = None
) -> int | None:
    likert_mapping = THAI_LIKERT_SCALE if mapping is None else mapping
    normalized = normalize_survey_value(value)
    if normalized is None:
        return None
    return likert_mapping.get(normalized)


def map_series_values(
    series: pd.Series, mapping: Mapping[str, Any], default: Any = None
) -> pd.Series:
    def _map_value(value):
        normalized = normalize_survey_value(value)
        if normalized is None:
            return None
        return mapping.get(normalized, default)
    return series.map(_map_value)
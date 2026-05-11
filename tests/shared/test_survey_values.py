from __future__ import annotations

import pandas as pd
import pytest

from catfood_unsupervised.shared.survey_values import (
    THAI_LIKERT_SCALE,
    map_likert_value,
    map_series_values,
    normalize_survey_value,
)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("มากที่สุด", 5),
        ("มาก", 4),
        ("ปานกลาง", 3),
        ("น้อย", 2),
        ("น้อยที่สุด", 1),
        ("เห็นด้วยที่สุด", 5),
        ("เห็นด้วย", 4),
        ("เฉยๆ", 3),
        ("ไม่เห็นด้วย", 2),
        ("ไม่เห็นด้วยเลย", 1),
    ],
)
def test_map_likert_value_returns_correct_integer(input_value, expected):
    result = map_likert_value(input_value)
    assert result == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("มากที่สุด", 10),
        ("มาก", 8),
        ("ปานกลาง", 5),
    ],
)
def test_map_likert_value_with_custom_mapping(input_value, expected):
    custom = {"มากที่สุด": 10, "มาก": 8, "ปานกลาง": 5}
    result = map_likert_value(input_value, mapping=custom)
    assert result == expected


@pytest.mark.parametrize(
    "input_value",
    [
        None,
        pd.NA,
        float("nan"),
        "",
        "  ",
    ],
)
def test_map_likert_value_returns_none_for_missing(input_value):
    result = map_likert_value(input_value)
    assert result is None


@pytest.mark.parametrize(
    "input_value",
    [
        "ไม่รู้",
        "invalid",
        "UNKNOWN",
    ],
)
def test_map_likert_value_returns_none_for_unmapped_strings(input_value):
    result = map_likert_value(input_value)
    assert result is None


def test_map_likert_value_uses_thai_likert_scale_by_default():
    assert THAI_LIKERT_SCALE["มากที่สุด"] == 5
    assert THAI_LIKERT_SCALE["น้อยที่สุด"] == 1


def test_normalize_survey_value_returns_stripped_string():
    result = normalize_survey_value("  มากที่สุด  ")
    assert result == "มากที่สุด"


def test_normalize_survey_value_returns_none_for_whitespace_only():
    result = normalize_survey_value("   ")
    assert result is None


def test_normalize_survey_value_returns_none_for_empty_string():
    result = normalize_survey_value("")
    assert result is None


def test_normalize_survey_value_returns_string_for_valid_input():
    result = normalize_survey_value("มาก")
    assert result == "มาก"


def test_map_series_values_applies_mapping():
    series = pd.Series(["มาก", "น้อย", "ปานกลาง"])
    result = map_series_values(series, THAI_LIKERT_SCALE)
    assert result.tolist() == [4, 2, 3]


def test_map_series_values_returns_default_for_unmapped():
    series = pd.Series(["มาก", "UNKNOWN", "น้อย"])
    result = map_series_values(series, THAI_LIKERT_SCALE, default=-1)
    assert result.tolist() == [4, -1, 2]


def test_map_series_values_handles_missing_values():
    series = pd.Series(["มาก", None, "น้อย"])
    result = map_series_values(series, THAI_LIKERT_SCALE, default=-1)
    assert result.tolist()[0] == 4
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == 2


def test_map_series_values_uses_none_as_default():
    series = pd.Series(["มาก", "UNKNOWN", "น้อย"])
    result = map_series_values(series, THAI_LIKERT_SCALE)
    assert result.tolist()[0] == 4
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == 2
from pathlib import Path

import pytest

from catfood_unsupervised.config import PROJECT_ROOT, RAW_DATA_PATH
from catfood_unsupervised.data_loading import (
    filter_completed_responses,
    load_raw_export,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOP3_COLUMN = (
    "จากตัวเลือกทั้งหมด คุณชอบการออกแบบบรรจุภัณฑ์อาหารแมวสำเร็จรูปแบบใดมากที่สุด "
    "3 อันดับแรก"
)


def test_raw_data_path_is_configured():
    expected_path = PROJECT_ROOT / "BU Data from Survey Cases_final(5).csv"

    assert RAW_DATA_PATH == expected_path
    assert RAW_DATA_PATH.is_absolute()


def test_load_raw_export_uses_second_row_as_headers_and_drops_empty_timestamps():
    survey_path = FIXTURES_DIR / "mini_survey.csv"

    loaded = load_raw_export(survey_path)

    assert loaded.columns.tolist() == ["Timestamp", "Cat", TOP3_COLUMN]
    assert loaded["Timestamp"].tolist() == [
        "2024-06-26 14:12:09",
        "2024-06-26 14:15:00",
        "2024-06-26 14:20:00",
    ]
    assert loaded["Cat"].tolist() == ["Mochi", "Tuna", "Pepper"]


def test_filter_completed_responses_keeps_only_rows_with_top3_answers():
    survey_path = FIXTURES_DIR / "mini_survey.csv"
    raw_responses = load_raw_export(survey_path)

    completed = filter_completed_responses(raw_responses, TOP3_COLUMN)

    assert completed["Timestamp"].tolist() == [
        "2024-06-26 14:12:09",
        "2024-06-26 14:20:00",
    ]
    assert completed[TOP3_COLUMN].tolist() == [
        "Option 1, Option 3, Option 5",
        "Option 2, Option 4, Option 6",
    ]


def test_load_raw_export_requires_timestamp_in_promoted_header(tmp_path: Path):
    malformed_export = tmp_path / "missing_timestamp_header.csv"
    malformed_export.write_text(
        "\n".join(
            [
                "Brief,\"Mini survey export used for data loading tests\",",
                "Submitted At,Cat,Top 3",
                "2024-06-26 14:12:09,Mochi,\"Option 1, Option 3, Option 5\"",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="raw export row 1 to contain 'Timestamp' in column 0"
    ):
        load_raw_export(malformed_export)


def test_load_raw_export_keeps_real_export_compatible_with_task_2_pipeline():
    loaded = load_raw_export(RAW_DATA_PATH)

    assert "Timestamp" in loaded.columns
    assert loaded.columns[0] == "Timestamp"
    assert TOP3_COLUMN in loaded.columns
    assert len(loaded) > 0

    completed = filter_completed_responses(loaded, TOP3_COLUMN)

    assert len(completed) > 0

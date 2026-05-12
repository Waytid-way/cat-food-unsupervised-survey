import json
from pathlib import Path

import pandas as pd

from catfood_unsupervised.pipeline import run_pipeline
from catfood_unsupervised.preprocessing import (
    fit_buy_factor_imputer as original_fit,
)


def test_run_pipeline_fits_buy_factor_imputer_on_reference_split(
    monkeypatch, tmp_path: Path
):
    data_path = _write_pipeline_fixture(tmp_path / "mini_pipeline.csv")
    captured: dict[str, int] = {}

    def spy_fit_buy_factor_imputer(reference_df, columns, n_neighbors=5):
        captured["reference_rows"] = len(reference_df)
        captured["column_count"] = len(columns)
        return original_fit(reference_df, columns, n_neighbors=n_neighbors)

    monkeypatch.setattr(
        "catfood_unsupervised.pipeline.fit_buy_factor_imputer",
        spy_fit_buy_factor_imputer,
    )

    run_pipeline(
        data_path=data_path,
        output_dir=tmp_path / "outputs",
        k_values=[2, 3],
        random_state=7,
    )

    assert captured["reference_rows"] < 8
    assert captured["column_count"] == 5


def test_run_pipeline_writes_metrics_summary_and_segmented_dataset(tmp_path: Path):
    data_path = _write_pipeline_fixture(tmp_path / "mini_pipeline.csv")

    result = run_pipeline(
        data_path=data_path,
        output_dir=tmp_path / "outputs",
        k_values=[2, 3],
        random_state=7,
    )

    metrics_path = tmp_path / "outputs" / "metrics_summary.json"
    dataset_path = tmp_path / "outputs" / "clean_dataset_with_segments.csv"

    assert metrics_path.exists()
    assert dataset_path.exists()
    assert "metrics" in result
    assert result["metrics"]["final_cluster_key"] == "segment"
    assert result["metrics"]["final_cluster_k"] in {2, 3}
    assert "segment" in result["dataframe"].columns

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_dataset = pd.read_csv(dataset_path)

    assert saved_metrics["final_cluster_key"] == "segment"
    assert saved_metrics["row_counts"]["completed_top3"] == 8
    assert "segment" in saved_dataset.columns


def _write_pipeline_fixture(path: Path) -> Path:
    headers = [f"column_{index}" for index in range(76)]
    headers[0] = "Timestamp"
    headers[72] = "top3_choice"
    headers[73] = "age"
    headers[74] = "gender"
    headers[75] = "marital"

    metadata_row = [f"metadata_{index}" for index in range(76)]
    rows = [metadata_row, headers]

    for respondent_index in range(8):
        cluster_a = respondent_index < 4
        row = [f"value_{respondent_index}_{column_index}" for column_index in range(76)]
        row[0] = f"2024-07-01 10:0{respondent_index}:00"
        row[72] = (
            "Option 1, Option 2, Option 3"
            if cluster_a
            else "Option 8, Option 9, Option 10"
        )
        row[73] = "20-29ปี" if cluster_a else "30-39ปี"
        row[74] = "หญิง" if cluster_a else "ชาย"
        row[75] = "โสด ไม่มีแฟน" if cluster_a else "แต่งงานแล้ว"

        buy_factor_value = "มากที่สุด" if cluster_a else "น้อย"
        packaging_importance_value = "มาก" if cluster_a else "น้อย"
        packaging_effect_value = "มีผล" if cluster_a else "ไม่มีผล"

        for column_index in range(5, 10):
            row[column_index] = buy_factor_value

        row[10] = packaging_effect_value
        row[11] = "ภาพแมว"

        for column_index in range(12, 20):
            row[column_index] = packaging_importance_value

        for option_index in range(10):
            for attribute_index in range(5):
                column_index = 22 + (option_index * 5) + attribute_index
                favored_response = (
                    "เห็นด้วยที่สุด"
                    if (respondent_index + attribute_index) % 3
                    else "เห็นด้วย"
                )
                unfavored_response = (
                    "ไม่เห็นด้วยเลย"
                    if (respondent_index + option_index) % 2
                    else "ไม่เห็นด้วย"
                )
                if cluster_a:
                    row[column_index] = favored_response if option_index < 5 else unfavored_response
                else:
                    row[column_index] = unfavored_response if option_index < 5 else favored_response

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, header=False, encoding="utf-8")
    return path


def test_run_pipeline_fits_buy_factor_imputer_on_reference_split(
    monkeypatch, tmp_path: Path
):
    data_path = _write_pipeline_fixture(tmp_path / "mini_pipeline.csv")
    captured: dict[str, int] = {}

    def spy_fit_buy_factor_imputer(reference_df, columns, n_neighbors=5):
        captured["reference_rows"] = len(reference_df)
        captured["column_count"] = len(columns)
        return original_fit(reference_df, columns, n_neighbors=n_neighbors)

    monkeypatch.setattr(
        "catfood_unsupervised.pipeline.fit_buy_factor_imputer",
        spy_fit_buy_factor_imputer,
    )

    run_pipeline(
        data_path=data_path,
        output_dir=tmp_path / "outputs",
        k_values=[2, 3],
        random_state=7,
    )

    assert captured["reference_rows"] < 8
    assert captured["column_count"] == 5

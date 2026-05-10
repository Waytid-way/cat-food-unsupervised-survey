from pathlib import Path

import pandas as pd

from catfood_unsupervised.reporting import write_reports_from_output_dir
from catfood_unsupervised.reporting import run_unsupervised_workflow


def test_write_reports_from_output_dir_creates_markdown_files(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    report_dir = tmp_path / "reports"

    report_paths = write_reports_from_output_dir(output_dir, report_dir)

    descriptive = report_paths["descriptive_stats"]
    unsupervised = report_paths["unsupervised_report"]
    owner_memo = report_paths["owner_memo"]

    assert descriptive.exists()
    assert unsupervised.exists()
    assert owner_memo.exists()

    descriptive_text = descriptive.read_text(encoding="utf-8")
    unsupervised_text = unsupervised.read_text(encoding="utf-8")
    memo_text = owner_memo.read_text(encoding="utf-8")

    assert "n = 148" in descriptive_text
    assert "Option 03" in descriptive_text
    assert "final_cluster_k = 2" in unsupervised_text
    assert "Segment 1" in unsupervised_text
    assert "k=2" in memo_text or "k = 2" in memo_text


def test_run_unsupervised_workflow_writes_pipeline_outputs_and_reports(tmp_path: Path):
    data_path = _write_pipeline_fixture(tmp_path / "mini_pipeline.csv")
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"

    result = run_unsupervised_workflow(
        data_path=data_path,
        output_dir=output_dir,
        report_dir=report_dir,
        k_values=[2, 3],
        random_state=7,
    )

    assert result["pipeline"]["metrics"]["final_cluster_key"] == "segment"
    assert result["pipeline"]["metrics"]["final_cluster_k"] in {2, 3}
    assert result["report_paths"]["descriptive_stats"].exists()
    assert result["report_paths"]["unsupervised_report"].exists()
    assert result["report_paths"]["owner_memo"].exists()


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
                favored_response = "เห็นด้วยที่สุด" if (respondent_index + attribute_index) % 3 else "เห็นด้วย"
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

from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.reporting import write_reports_from_output_dir


def test_write_reports_from_output_dir_creates_markdown_files(
    supervised_fixture_path: Path, tmp_path: Path
):
    output_dir = tmp_path / "supervised_outputs"
    report_dir = tmp_path / "supervised_reports"
    run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )

    report_paths = write_reports_from_output_dir(output_dir, report_dir)

    model_report = report_paths["model_report"]
    owner_memo = report_paths["owner_memo"]

    assert model_report.exists()
    assert owner_memo.exists()

    model_text = model_report.read_text(encoding="utf-8")
    owner_text = owner_memo.read_text(encoding="utf-8")

    assert "รายงานโมเดล Supervised Learning" in model_text
    assert "โมเดลที่ดีที่สุด" in model_text
    assert "source of truth" in owner_text

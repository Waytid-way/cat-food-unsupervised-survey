from pathlib import Path

from catfood_unsupervised.reporting import write_reports_from_output_dir


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

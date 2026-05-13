from __future__ import annotations

import runpy
from pathlib import Path


def test_build_deployment_artifacts_runs_unsupervised_then_copies_clean_dataset_then_runs_supervised(
    monkeypatch, tmp_path: Path
):
    import catfood_unsupervised.shared.paths as paths
    import catfood_unsupervised.supervised.pipeline as supervised_pipeline
    import catfood_unsupervised.unsupervised.pipeline as unsupervised_pipeline
    import catfood_unsupervised.unsupervised.reporting as unsupervised_reporting

    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "build_deployment_artifacts.py"

    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"
    unsupervised_output_dir = output_dir / "unsupervised"
    unsupervised_report_dir = report_dir / "unsupervised"
    supervised_output_dir = output_dir / "supervised"
    supervised_report_dir = report_dir / "supervised"
    raw_data_path = tmp_path / "raw.csv"
    raw_data_path.write_text("raw", encoding="utf-8")

    monkeypatch.setattr(paths, "RAW_DATA_PATH", raw_data_path)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paths, "REPORT_DIR", report_dir)
    monkeypatch.setattr(paths, "UNSUPERVISED_OUTPUT_DIR", unsupervised_output_dir)
    monkeypatch.setattr(paths, "UNSUPERVISED_REPORT_DIR", unsupervised_report_dir)
    monkeypatch.setattr(paths, "SUPERVISED_OUTPUT_DIR", supervised_output_dir)
    monkeypatch.setattr(paths, "SUPERVISED_REPORT_DIR", supervised_report_dir)

    calls: list[tuple[str, Path, Path]] = []

    def fake_run_pipeline(*, data_path, output_dir, **kwargs):
        calls.append(("unsupervised_pipeline", Path(data_path), Path(output_dir)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "clean_dataset_with_segments.csv").write_text(
            "segment\n1\n", encoding="utf-8"
        )
        return {"metrics": {"final_cluster_key": "segment", "final_cluster_k": 2}}

    def fake_write_reports_from_output_dir(output_dir, report_dir):
        calls.append(("unsupervised_reports", Path(output_dir), Path(report_dir)))
        assert (Path(output_dir) / "clean_dataset_with_segments.csv").exists()
        return {"report": Path(report_dir) / "unsupervised_report_th.md"}

    def fake_run_supervised_pipeline(*, input_path, output_dir, report_dir, **kwargs):
        calls.append(("supervised_pipeline", Path(input_path), Path(output_dir)))
        assert Path(input_path) == output_dir.parent / "clean_dataset_with_segments.csv"
        assert Path(input_path).exists()
        assert Path(output_dir) == supervised_output_dir
        assert Path(report_dir) == supervised_report_dir
        return {"best_model_name": "stub-model"}

    monkeypatch.setattr(unsupervised_pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        unsupervised_reporting,
        "write_reports_from_output_dir",
        fake_write_reports_from_output_dir,
    )
    monkeypatch.setattr(
        supervised_pipeline, "run_supervised_pipeline", fake_run_supervised_pipeline
    )

    runpy.run_path(str(script_path), run_name="__main__")

    assert calls == [
        ("unsupervised_pipeline", raw_data_path, unsupervised_output_dir),
        ("unsupervised_reports", unsupervised_output_dir, unsupervised_report_dir),
        (
            "supervised_pipeline",
            output_dir / "clean_dataset_with_segments.csv",
            supervised_output_dir,
        ),
    ]
    assert (output_dir / "clean_dataset_with_segments.csv").read_text(encoding="utf-8") == (
        unsupervised_output_dir / "clean_dataset_with_segments.csv"
    ).read_text(encoding="utf-8")

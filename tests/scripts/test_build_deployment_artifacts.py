from __future__ import annotations

import runpy
import shutil
from pathlib import Path


def test_build_deployment_artifacts_runs_unsupervised_then_copies_clean_dataset_then_runs_supervised(
    monkeypatch, tmp_path: Path
):
    import catfood_unsupervised.shared.paths as paths
    import catfood_unsupervised.reporting as reporting
    import catfood_unsupervised.supervised.config as supervised_config
    import catfood_unsupervised.supervised.pipeline as supervised_pipeline
    import catfood_unsupervised.unsupervised.config as unsupervised_config

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
    unsupervised_random_state = 42
    supervised_random_state = 1337

    monkeypatch.setattr(paths, "RAW_DATA_PATH", raw_data_path)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paths, "REPORT_DIR", report_dir)
    monkeypatch.setattr(paths, "UNSUPERVISED_OUTPUT_DIR", unsupervised_output_dir)
    monkeypatch.setattr(paths, "UNSUPERVISED_REPORT_DIR", unsupervised_report_dir)
    monkeypatch.setattr(unsupervised_config, "RANDOM_STATE", unsupervised_random_state)
    monkeypatch.setattr(
        supervised_config,
        "DEFAULT_INPUT_PATH",
        output_dir / paths.CLEAN_DATASET_FILENAME,
    )
    monkeypatch.setattr(supervised_config, "DEFAULT_OUTPUT_DIR", supervised_output_dir)
    monkeypatch.setattr(supervised_config, "DEFAULT_REPORT_DIR", supervised_report_dir)
    monkeypatch.setattr(supervised_config, "RANDOM_STATE", supervised_random_state)

    calls: list[tuple[str, Path, Path]] = []

    def fake_run_unsupervised_workflow(*, data_path, output_dir, report_dir, random_state, **kwargs):
        calls.append(("unsupervised_workflow", Path(data_path), Path(output_dir)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / paths.CLEAN_DATASET_FILENAME).write_text(
            "segment\n1\n", encoding="utf-8"
        )
        assert (Path(output_dir) / "clean_dataset_with_segments.csv").exists()
        assert Path(report_dir) == unsupervised_report_dir
        assert random_state == unsupervised_random_state
        return {"pipeline": {"metrics": {"final_cluster_key": "segment", "final_cluster_k": 2}}}

    def fake_copy2(src, dst):
        calls.append(("copy", Path(src), Path(dst)))
        assert Path(src) == unsupervised_output_dir / paths.CLEAN_DATASET_FILENAME
        assert Path(dst) == output_dir / paths.CLEAN_DATASET_FILENAME
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")

    def fake_run_supervised_pipeline(*, input_path, output_dir, report_dir, random_state, **kwargs):
        calls.append(("supervised_pipeline", Path(input_path), Path(output_dir)))
        assert Path(input_path) == output_dir.parent / paths.CLEAN_DATASET_FILENAME
        assert Path(input_path).exists()
        assert Path(output_dir) == supervised_output_dir
        assert Path(report_dir) == supervised_report_dir
        assert random_state == supervised_random_state
        return {"best_model_name": "stub-model"}

    monkeypatch.setattr(reporting, "run_unsupervised_workflow", fake_run_unsupervised_workflow)
    monkeypatch.setattr(shutil, "copy2", fake_copy2)
    monkeypatch.setattr(
        supervised_pipeline, "run_supervised_pipeline", fake_run_supervised_pipeline
    )

    runpy.run_path(str(script_path), run_name="__main__")

    assert calls == [
        ("unsupervised_workflow", raw_data_path, unsupervised_output_dir),
        ("copy", unsupervised_output_dir / paths.CLEAN_DATASET_FILENAME, output_dir / paths.CLEAN_DATASET_FILENAME),
        (
            "supervised_pipeline",
            output_dir / paths.CLEAN_DATASET_FILENAME,
            supervised_output_dir,
        ),
    ]
    assert (output_dir / paths.CLEAN_DATASET_FILENAME).read_text(encoding="utf-8") == (
        unsupervised_output_dir / paths.CLEAN_DATASET_FILENAME
    ).read_text(encoding="utf-8")

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path

from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.config import SCORING_ENTRYPOINT
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS
from sklearn.pipeline import Pipeline


def test_best_model_pickle_contains_full_pipeline_and_metrics_metadata(
    supervised_fixture_path: Path, tmp_path: Path
):
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "reports"

    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )

    model_path = output_dir / "best_model.pkl"
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    metrics_path = output_dir / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    module_path, function_name = SCORING_ENTRYPOINT.split(":")
    module = importlib.import_module(module_path)
    scoring_callable = getattr(module, function_name)

    assert isinstance(model, Pipeline)
    assert [step_name for step_name, _ in model.steps] == ["preprocessor", "model"]
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["class_labels"] == result["metrics"]["class_labels"]
    assert metrics["best_model_path"] == "best_model.pkl"
    assert metrics["scoring_entrypoint"] == "catfood_unsupervised.supervised.scoring:predict_supervised_segment"
    assert callable(scoring_callable)

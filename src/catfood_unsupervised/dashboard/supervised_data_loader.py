from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SupervisedDashboardBundle:
    metrics: dict
    comparison: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


def load_supervised_dashboard_bundle(output_dir: str | Path) -> SupervisedDashboardBundle:
    base = Path(output_dir)
    metrics_path = base / "metrics_summary.json"
    comparison_path = base / "model_comparison.csv"
    confusion_path = base / "confusion_matrix.csv"
    feature_importance_path = base / "feature_importance.csv"
    predictions_path = base / "predictions.csv"

    _require_file(metrics_path)
    _require_file(comparison_path)
    _require_file(confusion_path)
    _require_file(feature_importance_path)
    _require_file(predictions_path)

    return SupervisedDashboardBundle(
        metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
        comparison=pd.read_csv(comparison_path),
        confusion_matrix=pd.read_csv(confusion_path, index_col=0),
        feature_importance=pd.read_csv(feature_importance_path),
        predictions=pd.read_csv(predictions_path),
    )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")

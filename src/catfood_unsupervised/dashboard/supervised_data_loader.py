from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from catfood_unsupervised.dashboard.config import (
    SUPERVISED_HISTORY_DB_PATH,
    SUPERVISED_MODEL_PATH,
)
from catfood_unsupervised.supervised.config import DEFAULT_INPUT_PATH
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class SupervisedDashboardBundle:
    metrics: dict
    comparison: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    feature_options: dict[str, list[str]]
    model_path: Path
    history_db_path: Path


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
        feature_options=load_supervised_feature_options(DEFAULT_INPUT_PATH),
        model_path=base / SUPERVISED_MODEL_PATH.name,
        history_db_path=base / SUPERVISED_HISTORY_DB_PATH.name,
    )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")


def load_supervised_feature_options(input_path: str | Path) -> dict[str, list[str]]:
    df = pd.read_csv(input_path)
    feature_options: dict[str, list[str]] = {}
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing supervised feature column in input data: {column}")
        values = (
            df[column]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda series: series.ne("")]
            .drop_duplicates()
            .tolist()
        )
        feature_options[column] = values
    return feature_options

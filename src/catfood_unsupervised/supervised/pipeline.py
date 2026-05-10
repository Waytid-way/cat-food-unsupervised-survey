from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from catfood_unsupervised.supervised.config import (
    ANOMALY_COLUMN,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame
from catfood_unsupervised.supervised.models import (
    build_model_suite,
    evaluate_model_suite,
    extract_feature_importance,
)
from catfood_unsupervised.supervised.reporting import write_reports_from_output_dir


def run_supervised_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    df = load_supervised_dataset(input_path)
    X, y = build_supervised_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model_builders = build_model_suite(random_state=random_state)
    comparison, fitted_models, predictions = evaluate_model_suite(
        model_builders,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    best_model_name = str(comparison.iloc[0]["model_name"])
    best_model = fitted_models[best_model_name]
    best_predictions = predictions[best_model_name].copy()
    feature_importance = extract_feature_importance(
        best_model,
        X_test,
        y_test,
        random_state=random_state,
    )
    class_labels = sorted(pd.unique(y))
    confusion = _build_confusion_matrix(y_test, best_predictions["y_pred"], class_labels)
    report = classification_report(
        best_predictions["y_true"],
        best_predictions["y_pred"],
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )
    roc_auc = _compute_binary_roc_auc(best_predictions, class_labels)

    metrics = {
        "input_path": str(Path(input_path).resolve()),
        "row_count": int(len(df)),
        "feature_count": int(X.shape[1]),
        "class_labels": [int(label) if isinstance(label, (np.integer, int)) else str(label) for label in class_labels],
        "best_model_name": best_model_name,
        "best_model_accuracy": float(comparison.iloc[0]["accuracy"]),
        "best_model_macro_f1": float(comparison.iloc[0]["macro_f1"]),
        "best_model_weighted_f1": float(comparison.iloc[0]["weighted_f1"]),
        "roc_auc": roc_auc,
        "classification_report": _json_safe(report),
        "model_comparison": _json_safe(comparison.to_dict(orient="records")),
    }

    output_paths = _build_output_paths(output_dir)
    _write_outputs(
        output_paths=output_paths,
        comparison=comparison,
        confusion=confusion,
        feature_importance=feature_importance,
        best_predictions=best_predictions,
        best_model=best_model,
        metrics=metrics,
    )

    report_paths = write_reports_from_output_dir(output_dir, report_dir)
    return {
        "metrics": metrics,
        "comparison": comparison,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "predictions": best_predictions,
        "output_paths": output_paths,
        "report_paths": report_paths,
        "best_model_name": best_model_name,
    }


def _build_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, class_labels: list[object]
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in class_labels],
        columns=[f"pred_{label}" for label in class_labels],
    )


def _compute_binary_roc_auc(
    predictions: pd.DataFrame, class_labels: list[object]
) -> float | None:
    probability_columns = [column for column in predictions.columns if column.startswith("prob_class_")]
    if len(class_labels) != 2 or len(probability_columns) < 2:
        return None
    positive_label = class_labels[-1]
    positive_column = f"prob_class_{positive_label}"
    if positive_column not in predictions.columns:
        return None
    try:
        return float(roc_auc_score(predictions["y_true"], predictions[positive_column]))
    except ValueError:
        return None


def _build_output_paths(output_dir: str | Path) -> dict[str, Path]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "metrics_summary": base / "metrics_summary.json",
        "model_comparison": base / "model_comparison.csv",
        "confusion_matrix": base / "confusion_matrix.csv",
        "feature_importance": base / "feature_importance.csv",
        "predictions": base / "predictions.csv",
        "best_model": base / "best_model.pkl",
    }


def _write_outputs(
    *,
    output_paths: dict[str, Path],
    comparison: pd.DataFrame,
    confusion: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_predictions: pd.DataFrame,
    best_model,
    metrics: dict[str, Any],
) -> None:
    comparison.to_csv(output_paths["model_comparison"], index=False)
    confusion.to_csv(output_paths["confusion_matrix"])
    feature_importance.to_csv(output_paths["feature_importance"], index=False)
    best_predictions.to_csv(output_paths["predictions"], index=False)
    output_paths["metrics_summary"].write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with output_paths["best_model"].open("wb") as fh:
        pickle.dump(best_model, fh)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    return str(value)


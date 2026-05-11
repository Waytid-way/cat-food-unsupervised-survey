from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from catfood_unsupervised.supervised.config import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR


REPORT_FILENAMES = {
    "model_report": "supervised_model_report_th.md",
    "owner_memo": "supervised_owner_memo_th.md",
}


@dataclass(frozen=True)
class SupervisedReportContext:
    metrics: dict[str, Any]
    comparison: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


def load_report_context(output_dir: str | Path) -> SupervisedReportContext:
    base = Path(output_dir)
    metrics = json.loads((base / "metrics_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(base / "model_comparison.csv")
    confusion_matrix = pd.read_csv(base / "confusion_matrix.csv", index_col=0)
    feature_importance = pd.read_csv(base / "feature_importance.csv")
    predictions = pd.read_csv(base / "predictions.csv")
    return SupervisedReportContext(
        metrics=metrics,
        comparison=comparison,
        confusion_matrix=confusion_matrix,
        feature_importance=feature_importance,
        predictions=predictions,
    )


def render_supervised_model_report(context: SupervisedReportContext) -> str:
    metrics = context.metrics
    best_row = context.comparison.iloc[0]
    per_class_rows = _build_per_class_metric_rows(metrics)
    report_lines = [
        "# Supervised Learning Model Report",
        "",
        "## Executive Summary",
        f"- Training rows used: {metrics['row_count']}",
        f"- Encoded feature count: {metrics['feature_count']}",
        f"- Best model: {metrics['best_model_name']}",
        f"- Accuracy: {metrics['best_model_accuracy']:.3f}",
        f"- Macro F1: {metrics['best_model_macro_f1']:.3f}",
        f"- Weighted F1: {metrics['best_model_weighted_f1']:.3f}",
        (
            f"- ROC AUC: {metrics['roc_auc']:.3f}"
            if metrics.get("roc_auc") is not None
            else "- ROC AUC: n/a"
        ),
        "",
        "## Model Justification",
        (
            f"- Selected `{metrics['best_model_name']}` because it ranked first on macro F1, "
            "weighted F1, and accuracy in the holdout comparison."
        ),
        (
            f"- Holdout performance for the selected model: accuracy {float(best_row['accuracy']):.3f}, "
            f"macro F1 {float(best_row['macro_f1']):.3f}, weighted F1 {float(best_row['weighted_f1']):.3f}."
        ),
        "- Anomaly rows were excluded before training, so the comparison is based on clean segment labels only.",
        "- The model suite uses the approved supervised feature contract and avoids leakage columns.",
        "",
        "## Model Comparison",
        _markdown_table(
            ["model", "accuracy", "macro_f1", "weighted_f1"],
            [
                [
                    row["model_name"],
                    f"{row['accuracy']:.3f}",
                    f"{row['macro_f1']:.3f}",
                    f"{row['weighted_f1']:.3f}",
                ]
                for _, row in context.comparison.iterrows()
            ],
        ),
        "",
        "## Per-class metrics",
        (
            _markdown_table(
                ["class", "precision", "recall", "f1_score", "support"],
                [
                    [
                        row["label"],
                        f"{row['precision']:.3f}",
                        f"{row['recall']:.3f}",
                        f"{row['f1_score']:.3f}",
                        row["support"],
                    ]
                    for row in per_class_rows
                ],
            )
            if per_class_rows
            else "_No per-class metrics available._"
        ),
        "",
        "## Confusion Matrix",
        _markdown_table(
            ["", *context.confusion_matrix.columns.tolist()],
            [
                [index, *[int(value) for value in row.tolist()]]
                for index, row in context.confusion_matrix.iterrows()
            ],
        ),
        "",
        "## Feature Importance",
        _markdown_table(
            ["feature", "importance_mean", "importance_std"],
            [
                [
                    row["feature"],
                    f"{row['importance_mean']:.4f}",
                    f"{row['importance_std']:.4f}",
                ]
                for _, row in context.feature_importance.head(10).iterrows()
            ],
        ),
        "",
        "## Business Guidance",
        f"- Use `{metrics['best_model_name']}` as the source of truth for segment prediction.",
        "- Revisit feature engineering if the confusion matrix shows a strong class bias.",
        "- Use the top three features from the importance table as the narrative backbone in dashboard and slides.",
        "",
        "## Output Files",
        "- `outputs/supervised/metrics_summary.json`",
        "- `outputs/supervised/model_comparison.csv`",
        "- `outputs/supervised/confusion_matrix.csv`",
        "- `outputs/supervised/feature_importance.csv`",
        "- `outputs/supervised/predictions.csv`",
        "- `outputs/supervised/best_model.pkl`",
    ]
    return "\n".join(report_lines).rstrip() + "\n"


def render_supervised_owner_memo(context: SupervisedReportContext) -> str:
    metrics = context.metrics
    return "\n".join(
        [
            "# Supervised Owner Memo",
            "",
            f"- Source of truth model: `{metrics['best_model_name']}`",
            f"- Accuracy on holdout set: {metrics['best_model_accuracy']:.3f}",
            f"- Macro F1 on holdout set: {metrics['best_model_macro_f1']:.3f}",
            "- Use the Supervised dashboard tab to review model comparison, confusion matrix, and feature importance.",
            "- If you export the model for external use, keep `best_model.pkl` and `metrics_summary.json` together.",
            "",
            "## What the metrics mean",
            "- If the confusion matrix is skewed in one class, the segment labels may still overlap and feature engineering should be revisited.",
            "- If feature importance concentrates on a few columns, the survey contains strong signals and the model can be simplified later.",
        ]
    ).rstrip() + "\n"


def write_reports_from_output_dir(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    context = load_report_context(output_dir)
    base = Path(report_dir)
    base.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_report": base / REPORT_FILENAMES["model_report"],
        "owner_memo": base / REPORT_FILENAMES["owner_memo"],
    }
    outputs["model_report"].write_text(
        render_supervised_model_report(context), encoding="utf-8"
    )
    outputs["owner_memo"].write_text(
        render_supervised_owner_memo(context), encoding="utf-8"
    )
    return outputs


def run_supervised_workflow(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline

    pipeline_result = run_supervised_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        report_dir=report_dir,
        **pipeline_kwargs,
    )
    return {
        "pipeline": pipeline_result,
        "report_paths": pipeline_result["report_paths"],
    }


def _build_per_class_metric_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    report = metrics.get("classification_report")
    class_labels = metrics.get("class_labels") or []
    if not isinstance(report, Mapping) or not class_labels:
        return []

    rows: list[dict[str, Any]] = []
    for label in class_labels:
        summary = report.get(str(label))
        if not isinstance(summary, Mapping):
            continue
        rows.append(
            {
                "label": f"Segment {label}",
                "precision": float(summary.get("precision", 0.0)),
                "recall": float(summary.get("recall", 0.0)),
                "f1_score": float(summary.get("f1-score", 0.0)),
                "support": int(summary.get("support", 0)),
            }
        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])

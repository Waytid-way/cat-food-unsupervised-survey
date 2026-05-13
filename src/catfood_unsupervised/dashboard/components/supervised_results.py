from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from dash import dash_table as dt, html

from catfood_unsupervised.dashboard.bootstrap import dbc


@dataclass(frozen=True)
class PredictionSummary:
    display_label: str
    raw_class: str
    confidence: float
    top_probabilities: tuple[tuple[str, float], ...]


def render_supervised_metric_cards(metrics: Mapping[str, object]) -> dbc.Row:
    cards = [
        ("Best Model", str(metrics.get("best_model_name", "n/a")), "#2E86AB"),
        ("Accuracy", f"{_safe_float(metrics.get('best_model_accuracy')):.1%}", "#F18F01"),
        ("Macro F1", f"{_safe_float(metrics.get('best_model_macro_f1')):.1%}", "#A23B72"),
        ("Weighted F1", f"{_safe_float(metrics.get('best_model_weighted_f1')):.1%}", "#3A7D44"),
    ]
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(label, className="page-title"),
                            html.H3(value, style={"color": color}),
                        ]
                    ),
                    className="ind-card",
                ),
                width=3,
            )
            for label, value, color in cards
        ],
        className="mb-4 g-3",
    )


def render_prediction_result_panel(
    prediction_frame: pd.DataFrame | None,
    metrics: Mapping[str, object],
) -> html.Div:
    if prediction_frame is None or prediction_frame.empty:
        body = [
            html.H5("Prediction Result", className="page-title"),
            html.P("Upload a model, complete the form, and click Predict.", className="text-muted"),
        ]
    else:
        summary = build_prediction_summary(prediction_frame.iloc[0], metrics)
        probability_items = [
            html.Li(f"{label}: {probability:.1%}")
            for label, probability in summary.top_probabilities
        ]
        body = [
            html.H5("Prediction Result", className="page-title"),
            html.P("Predicted class", className="text-muted mb-1"),
            html.H3(summary.display_label, className="mb-2"),
            html.P(
                ["Confidence: ", html.Strong(f"{summary.confidence:.1%}")],
                className="mb-2",
            ),
            html.P(f"Raw class: {summary.raw_class}", className="text-muted mb-2"),
            html.P("Top probabilities", className="fw-semibold mb-1"),
            html.Ul(probability_items if probability_items else [html.Li("No probability scores available.")], className="mb-2"),
            html.P(f"Best model: {metrics.get('best_model_name', 'n/a')}", className="text-muted mb-2"),
        ]

    return html.Div(
        dbc.Card(
            dbc.CardBody(body),
            className="ind-card",
        ),
        id="supervised-result-panel",
    )


def render_recent_history_panel(history_frame: pd.DataFrame | None) -> html.Div:
    if history_frame is None or history_frame.empty:
        content = [
            html.H5("Recent Predictions", className="page-title"),
            html.P("No prediction history yet.", className="text-muted"),
        ]
    else:
        display = history_frame.copy()
        for column in ("probability", "raw_input"):
            if column in display.columns:
                display[column] = display[column].map(lambda value: json.dumps(value, ensure_ascii=False))
        columns = [
            {"name": "created_at", "id": "created_at"},
            {"name": "source", "id": "source"},
            {"name": "model_name", "id": "model_name"},
            {"name": "predicted_segment", "id": "predicted_segment"},
            {"name": "probability", "id": "probability"},
        ]
        content = [
            html.H5("Recent Predictions", className="page-title"),
            dt.DataTable(
                columns=columns,
                data=display.loc[:, [column["id"] for column in columns]].to_dict("records"),
                page_size=5,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                style_header={"fontWeight": "bold"},
            ),
        ]

    return html.Div(
        dbc.Card(
            dbc.CardBody(content),
            className="ind-card",
        ),
        id="supervised-history-panel",
    )


def build_prediction_summary(
    prediction_row: pd.Series,
    metrics: Mapping[str, object],
) -> PredictionSummary:
    raw_class_value = _format_class_value(prediction_row.get("predicted_segment"))
    probability_items = _extract_probability_items(prediction_row)
    display_probability_items = tuple(
        (_friendly_class_label(class_label, metrics), probability)
        for class_label, probability in probability_items[:3]
    )
    confidence = display_probability_items[0][1] if display_probability_items else 0.0
    return PredictionSummary(
        display_label=_friendly_class_label(raw_class_value, metrics),
        raw_class=raw_class_value,
        confidence=confidence,
        top_probabilities=display_probability_items,
    )


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_probability_items(prediction_row: pd.Series) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for column in prediction_row.index:
        column_name = str(column)
        if not column_name.startswith("prob_class_"):
            continue
        class_label = column_name.removeprefix("prob_class_")
        items.append((class_label, _safe_float(prediction_row[column])))
    items.sort(key=lambda item: (-item[1], item[0]))
    return items


def _friendly_class_label(class_label: object, metrics: Mapping[str, object]) -> str:
    normalized_label = _format_class_value(class_label)
    if normalized_label == "1":
        return "Segment 1 - Premium audience"
    if normalized_label == "2":
        return "Segment 2 - Main-market audience"

    class_labels = metrics.get("class_labels")
    if isinstance(class_labels, list) and normalized_label in {str(label) for label in class_labels}:
        return f"Segment {normalized_label}"
    return f"Class {normalized_label}"


def _format_class_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"

    text = str(value).strip()
    if not text:
        return "n/a"
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"


__all__ = [
    "PredictionSummary",
    "build_prediction_summary",
    "render_prediction_result_panel",
    "render_recent_history_panel",
    "render_supervised_metric_cards",
]

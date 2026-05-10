from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc


def render_prediction_result_panel(
    prediction_frame: pd.DataFrame | None,
    metrics: Mapping[str, Any],
) -> html.Div:
    if prediction_frame is None or prediction_frame.empty:
        return html.Div(
            [
                html.Div("Prediction result", className="supervised-results__eyebrow"),
                html.H4("Waiting for a scored customer row", className="supervised-results__title"),
                html.P(
                    "Once you click Predict, the selected row will appear here with the predicted segment and confidence distribution.",
                    className="supervised-results__lede",
                ),
                html.Div(
                    [
                        html.Div("Confidence", className="supervised-results__metric-label"),
                        html.Div("Will appear after scoring", className="supervised-results__metric-value"),
                    ],
                    className="supervised-results__metric-card",
                ),
            ],
            className="supervised-results__empty",
        )

    row = prediction_frame.iloc[0].to_dict()
    probability_map = _extract_probability_map(row)
    confidence = max(probability_map.values()) if probability_map else 0.0
    confidence_label = _segment_label(row.get("predicted_segment"))

    return html.Div(
        [
            html.Div("Prediction result", className="supervised-results__eyebrow"),
            html.H4("Scored customer row", className="supervised-results__title"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Predicted segment", className="supervised-results__metric-label"),
                                html.Div(confidence_label, className="supervised-results__metric-value"),
                            ],
                            className="supervised-results__metric-card",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Confidence", className="supervised-results__metric-label"),
                                html.Div(f"{confidence:.1%}", className="supervised-results__metric-value"),
                            ],
                            className="supervised-results__metric-card",
                        ),
                        width=6,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dcc.Graph(
                figure=build_confidence_figure(probability_map, row.get("predicted_segment")),
                config={"displaylogo": False, "responsive": True},
            ),
            html.Div(
                [
                    html.Div("Model used", className="supervised-results__meta-label"),
                    html.Div(str(metrics.get("best_model_name", "n/a")), className="supervised-results__meta-value"),
                ],
                className="supervised-results__meta",
            ),
        ],
        className="supervised-results__panel",
    )


def render_recent_history_panel(history_df: pd.DataFrame | None, limit: int = 5) -> html.Div:
    frame = history_df.head(limit).copy() if history_df is not None and not history_df.empty else pd.DataFrame()
    if frame.empty:
        return html.Div(
            [
                html.Div("Recent predictions", className="supervised-results__eyebrow"),
                html.H5("No prediction history yet", className="supervised-results__title"),
                html.P(
                    "Predictions you create in the dashboard will be stored in SQLite and appear here.",
                    className="supervised-results__lede",
                ),
            ],
            className="supervised-results__empty",
        )

    rows = []
    for _, record in frame.iterrows():
        probability = record.get("probability") or {}
        confidence = max(probability.values()) if isinstance(probability, dict) and probability else 0.0
        rows.append(
            html.Tr(
                [
                    html.Td(str(record.get("created_at", ""))),
                    html.Td(str(record.get("source", ""))),
                    html.Td(str(record.get("model_name", ""))),
                    html.Td(f"Segment {record.get('predicted_segment', '')}"),
                    html.Td(f"{confidence:.1%}"),
                ]
            )
        )

    return html.Div(
        [
            html.Div("Recent predictions", className="supervised-results__eyebrow"),
            html.H5("Latest scored rows", className="supervised-results__title"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Time"),
                                html.Th("Source"),
                                html.Th("Model"),
                                html.Th("Segment"),
                                html.Th("Confidence"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped supervised-results__table",
            ),
        ],
        className="supervised-results__panel",
    )


def build_confidence_figure(
    probability_map: Mapping[str, Any],
    predicted_segment: Any,
) -> go.Figure:
    if not probability_map:
        fig = go.Figure()
        fig.update_layout(
            title="Confidence",
            height=260,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    probability_keys = list(probability_map.keys())
    labels = [_friendly_probability_label(label) for label in probability_keys]
    values = [float(probability_map[label]) for label in probability_keys]
    colors = [
        "#2E86AB"
        if _segment_from_probability_label(label, predicted_segment)
        else "#A7B6C2"
        for label in probability_keys
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{value:.1%}" for value in values],
                textposition="outside",
                cliponaxis=False,
            )
        ]
    )
    fig.update_layout(
        title="Confidence",
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis_title="Class probability",
        height=280,
        margin=dict(l=30, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2933"),
    )
    return fig


def _extract_probability_map(row: Mapping[str, Any]) -> dict[str, float]:
    probability_items = {
        key: float(value)
        for key, value in row.items()
        if str(key).startswith("prob_class_")
    }
    if probability_items:
        return probability_items

    probability = row.get("probability")
    if isinstance(probability, Mapping):
        return {str(key): float(value) for key, value in probability.items()}
    return {}


def _segment_label(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"Segment {value}"


def _segment_from_probability_label(probability_label: str, predicted_segment: Any) -> bool:
    segment_suffix = str(probability_label).removeprefix("prob_class_")
    return segment_suffix == str(predicted_segment)


def _friendly_probability_label(probability_label: str) -> str:
    segment_suffix = str(probability_label).removeprefix("prob_class_")
    return f"Segment {segment_suffix}"

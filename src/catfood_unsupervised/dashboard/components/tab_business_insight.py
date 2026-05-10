from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.components.supervised_results import (
    render_recent_history_panel,
)
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
)


def render_business_insight_tab(
    bundle: SupervisedDashboardBundle,
    history_df: pd.DataFrame | None = None,
) -> html.Div:
    history = history_df.copy() if history_df is not None else pd.DataFrame()
    total_predictions = int(len(history))
    segment_counts = history["predicted_segment"].value_counts().sort_index() if not history.empty else pd.Series(dtype=int)
    most_common_segment = segment_counts.idxmax() if not segment_counts.empty else None
    latest_model = history.iloc[0]["model_name"] if not history.empty else "n/a"

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Business Insight", className="supervised-business__eyebrow"),
                    html.H3("How the supervised tool is being used", className="supervised-business__title"),
                    html.P(
                        "This page summarizes the SQLite prediction history so business users can see which segments are being requested, which model is active, and what the recent mix looks like.",
                        className="supervised-business__lede",
                    ),
                ],
                className="supervised-business__hero",
            ),
            dbc.Row(
                [
                    dbc.Col(_summary_card("Total predictions", str(total_predictions)), width=3),
                    dbc.Col(_summary_card("Latest model", str(latest_model)), width=3),
                    dbc.Col(_summary_card("Most common segment", _segment_label(most_common_segment)), width=3),
                    dbc.Col(_summary_card("Distinct segments", str(segment_counts.index.nunique() if not segment_counts.empty else 0)), width=3),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Segment breakdown", className="supervised-business__section-eyebrow"),
                        html.H5("Count by segment", className="supervised-business__section-title"),
                        _render_segment_breakdown(segment_counts),
                    ]
                ),
                className="supervised-business__panel mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Segment mix", className="supervised-business__section-eyebrow"),
                                    html.H5("Prediction counts by segment", className="supervised-business__section-title"),
                                    dcc.Graph(
                                        figure=_build_segment_mix_figure(segment_counts),
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ]
                            ),
                            className="supervised-business__panel h-100",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Prediction history", className="supervised-business__section-eyebrow"),
                                    html.H5("Latest prediction rows", className="supervised-business__section-title"),
                                    render_recent_history_panel(history_df, limit=8),
                                ]
                            ),
                            className="supervised-business__panel h-100",
                        ),
                        width=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        id="tab_business_insight",
    )


def _summary_card(label: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="supervised-business__metric-label"),
                html.Div(value, className="supervised-business__metric-value"),
            ]
        ),
        className="supervised-business__metric-card",
    )


def _build_segment_mix_figure(segment_counts: pd.Series) -> go.Figure:
    if segment_counts.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Prediction counts by segment",
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    labels = [f"Segment {segment}" for segment in segment_counts.index]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=segment_counts.values,
                marker_color=["#2E86AB", "#F18F01", "#A23B72", "#3A7D44"][: len(segment_counts)],
                text=segment_counts.values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Prediction counts by segment",
        yaxis_title="Predictions",
        xaxis_title="Segment",
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _render_segment_breakdown(segment_counts: pd.Series) -> html.Div:
    if segment_counts.empty:
        return html.P("No predictions have been recorded yet.", className="mb-0")

    items = [
        html.Li(f"Segment {segment}: {count}")
        for segment, count in segment_counts.items()
    ]
    return html.Ul(items, className="supervised-business__segment-list")


def _segment_label(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"Segment {value}"

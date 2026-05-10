from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
)


def render_supervised_tab(bundle: SupervisedDashboardBundle) -> html.Div:
    metrics = bundle.metrics
    comparison = bundle.comparison
    confusion_matrix = bundle.confusion_matrix
    feature_importance = bundle.feature_importance
    predictions = bundle.predictions
    class_labels = _extract_class_labels(metrics, predictions)
    insight = _build_strategic_insight(metrics, feature_importance)

    hero = html.Div(
        [
            html.Div("5. Supervised Learning", className="supervised-hero__eyebrow"),
            html.H3(
                "Predict the segment for a new customer, then turn it into a campaign decision",
                className="supervised-hero__title",
            ),
            html.P(
                "We train on the cleaned survey data to predict the unsupervised segment label and expose the result as a decision-ready dashboard.",
                className="supervised-hero__lede",
            ),
            html.Div(
                [
                    _pill("Best model", metrics["best_model_name"]),
                    _pill("Accuracy", f"{metrics['best_model_accuracy']:.3f}"),
                    _pill("Macro F1", f"{metrics['best_model_macro_f1']:.3f}"),
                    _pill("Weighted F1", f"{metrics['best_model_weighted_f1']:.3f}"),
                ],
                className="supervised-pill-row",
            ),
        ],
        className="supervised-hero",
        style={
            "margin": "0.25rem 0 1rem",
            "padding": "1.4rem 1.5rem",
            "borderRadius": "24px",
            "border": "1px solid rgba(46, 134, 171, 0.18)",
            "background": "linear-gradient(135deg, rgba(46, 134, 171, 0.12), rgba(255, 255, 255, 0.96) 45%, rgba(58, 125, 68, 0.08))",
            "boxShadow": "0 18px 35px rgba(15, 23, 42, 0.08)",
        },
    )

    insight_row = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div("Strategic insight", className="supervised-insight__label"),
                        html.H5(insight["headline"], className="supervised-insight__headline"),
                        html.P(insight["summary"], className="supervised-insight__text"),
                        html.Div(
                            [
                                _pill("Top driver", insight["top_driver"]),
                                _pill("Next signal", insight["second_driver"]),
                                _pill("Action", insight["action"]),
                            ],
                            className="supervised-pill-row supervised-pill-row--tight",
                        ),
                    ],
                    className="supervised-insight-card",
                    style={
                        "height": "100%",
                        "padding": "1.2rem 1.25rem",
                        "borderLeft": "6px solid #2E86AB",
                        "borderRadius": "22px",
                        "border": "1px solid rgba(15, 23, 42, 0.08)",
                        "background": "linear-gradient(180deg, #FFFFFF 0%, #F8FBFE 100%)",
                        "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                    },
                ),
                width=8,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div("Decision cue", className="supervised-insight__label"),
                        html.H5("How to use this result", className="supervised-insight__headline"),
                        html.Ul(
                            [
                                html.Li("Use the best model score as the source of truth for the dashboard."),
                                html.Li("Treat packaging imagery and proof cues as the first creative lever."),
                                html.Li("Use age and household context to tune message tone, not to replace the packaging story."),
                            ],
                            className="supervised-list",
                        ),
                    ],
                    className="supervised-forecast-card",
                    style={
                        "height": "100%",
                        "padding": "1.2rem 1.25rem",
                        "borderLeft": "6px solid #F18F01",
                        "borderRadius": "22px",
                        "border": "1px solid rgba(15, 23, 42, 0.08)",
                        "background": "linear-gradient(180deg, #FFFFFF 0%, #FCF8F1 100%)",
                        "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                    },
                ),
                width=4,
            ),
        ],
        className="g-3 mb-4",
    )

    comparison_fig = _build_model_comparison_figure(comparison, metrics["best_model_name"])
    confusion_fig = _build_confusion_matrix_figure(confusion_matrix)
    feature_fig = _build_feature_importance_figure(feature_importance)
    roc_fig = _build_roc_figure(predictions, class_labels)

    return html.Div(
        [
            hero,
            insight_row,
            dbc.Row(
                [
                    dbc.Col(_metric_card("Best Model", metrics["best_model_name"], "#2E86AB"), width=3),
                    dbc.Col(_metric_card("Accuracy", f"{metrics['best_model_accuracy']:.3f}", "#2E86AB"), width=3),
                    dbc.Col(_metric_card("Macro F1", f"{metrics['best_model_macro_f1']:.3f}", "#A23B72"), width=3),
                    dbc.Col(_metric_card("Weighted F1", f"{metrics['best_model_weighted_f1']:.3f}", "#3A7D44"), width=3),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody([dcc.Graph(figure=comparison_fig, config=_graph_config())]),
                className="supervised-panel mb-4",
                style={
                    "borderRadius": "22px",
                    "border": "1px solid rgba(15, 23, 42, 0.08)",
                    "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                    "background": "#FFFFFF",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=confusion_fig, config=_graph_config())]),
                            className="supervised-panel h-100",
                            style={
                                "borderRadius": "22px",
                                "border": "1px solid rgba(15, 23, 42, 0.08)",
                                "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                                "background": "#FFFFFF",
                            },
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([dcc.Graph(figure=feature_fig, config=_graph_config())]),
                            className="supervised-panel h-100",
                            style={
                                "borderRadius": "22px",
                                "border": "1px solid rgba(15, 23, 42, 0.08)",
                                "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                                "background": "#FFFFFF",
                            },
                        ),
                        width=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody([_render_class_metrics_table(metrics)]),
                className="supervised-panel mb-4",
                style={
                    "borderRadius": "22px",
                    "border": "1px solid rgba(15, 23, 42, 0.08)",
                    "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                    "background": "#FFFFFF",
                },
            ),
            dbc.Card(
                dbc.CardBody([dcc.Graph(figure=roc_fig, config=_graph_config())]),
                className="supervised-panel mb-3",
                style={
                    "borderRadius": "22px",
                    "border": "1px solid rgba(15, 23, 42, 0.08)",
                    "boxShadow": "0 12px 28px rgba(15, 23, 42, 0.06)",
                    "background": "#FFFFFF",
                },
            ),
        ],
        id="tab_supervised",
        className="supervised-shell",
    )


def _metric_card(title: str, value: Any, accent: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="supervised-metric__title"),
                html.Div(str(value), className="supervised-metric__value"),
            ]
        ),
        className="supervised-metric-card",
        style={
            "height": "100%",
            "borderRadius": "18px",
            "boxShadow": "0 10px 20px rgba(15, 23, 42, 0.06)",
            "borderTop": f"4px solid {accent}",
        },
    )


def _pill(label: str, value: Any):
    return html.Span(
        [
            html.Span(label, className="supervised-pill__label"),
            html.Span(str(value), className="supervised-pill__value"),
        ],
        className="supervised-pill",
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "0.45rem",
            "padding": "0.45rem 0.8rem",
            "borderRadius": "999px",
            "background": "rgba(255, 255, 255, 0.82)",
            "border": "1px solid rgba(15, 23, 42, 0.08)",
            "boxShadow": "0 4px 10px rgba(15, 23, 42, 0.04)",
            "color": "#102A43",
        },
    )


def _graph_config() -> dict[str, Any]:
    return {"displaylogo": False, "responsive": True}


def _build_model_comparison_figure(
    comparison: pd.DataFrame, best_model_name: str
) -> go.Figure:
    fig = go.Figure()
    colors = {
        "accuracy": "#2E86AB",
        "macro_f1": "#A23B72",
        "weighted_f1": "#F18F01",
    }
    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]:
        fig.add_trace(
            go.Bar(
                name=metric_label,
                x=comparison["model_name"],
                y=comparison[metric_key],
                marker_color=colors[metric_key],
                text=[f"{value:.3f}" for value in comparison[metric_key]],
                textposition="outside",
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title="Model comparison",
        barmode="group",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Model",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=40, r=20, t=60, b=60),
        legend_title_text="Metric",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2933"),
        annotations=[
            dict(
                x=0.01,
                y=1.10,
                xref="paper",
                yref="paper",
                text=f"Best performer: {best_model_name}",
                showarrow=False,
                font=dict(size=12, color="#2E86AB"),
            )
        ],
    )
    return fig


def _build_confusion_matrix_figure(confusion_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix.to_numpy(),
            x=[str(column) for column in confusion_matrix.columns],
            y=[str(index) for index in confusion_matrix.index],
            colorscale=[[0, "#eff6ff"], [1, "#2E86AB"]],
            text=confusion_matrix.to_numpy(),
            texttemplate="%{text}",
            hoverongaps=False,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion matrix",
        height=380,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=40, r=20, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2933"),
    )
    return fig


def _build_feature_importance_figure(feature_importance: pd.DataFrame) -> go.Figure:
    top = feature_importance.head(10).iloc[::-1]
    fig = go.Figure(
        data=go.Bar(
            x=top["importance_mean"],
            y=top["feature"],
            orientation="h",
            marker_color="#3A7D44",
            error_x=dict(type="data", array=top["importance_std"]),
            text=[f"{value:.3f}" for value in top["importance_mean"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Top feature drivers",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Permutation importance",
        margin=dict(l=20, r=20, t=60, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2933"),
    )
    return fig


def _build_roc_figure(predictions: pd.DataFrame, class_labels: list[object]) -> go.Figure:
    fig = go.Figure()
    probability_columns = {
        column.removeprefix("prob_class_"): column
        for column in predictions.columns
        if column.startswith("prob_class_")
    }
    if len(class_labels) == 0 or not probability_columns:
        fig.update_layout(title="ROC curve", height=360)
        return fig

    y_true = predictions["y_true"]
    y_binarized = label_binarize(y_true, classes=class_labels)
    if y_binarized.ndim == 1:
        y_binarized = y_binarized.reshape(-1, 1)

    for index, class_label in enumerate(class_labels):
        probability_column = probability_columns.get(str(class_label))
        if probability_column is None:
            continue
        if y_binarized.shape[1] == 1:
            y_binary = y_binarized.ravel()
        else:
            y_binary = y_binarized[:, index]
        fpr, tpr, _ = roc_curve(y_binary, predictions[probability_column])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Class {class_label} (AUC={roc_auc:.3f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            name="Chance",
        )
    )
    fig.update_layout(
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=50),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2933"),
    )
    return fig


def _render_class_metrics_table(metrics: dict[str, Any]) -> html.Div:
    report = metrics.get("classification_report", {})
    class_labels = metrics.get("class_labels", [])
    rows = []
    for label in class_labels:
        label_key = str(label)
        class_metrics = report.get(label_key)
        if not isinstance(class_metrics, dict):
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(label_key),
                    html.Td(f"{class_metrics.get('precision', 0):.3f}"),
                    html.Td(f"{class_metrics.get('recall', 0):.3f}"),
                    html.Td(f"{class_metrics.get('f1-score', 0):.3f}"),
                    html.Td(str(int(class_metrics.get("support", 0)))),
                ]
            )
        )

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Class"),
                        html.Th("Precision"),
                        html.Th("Recall"),
                        html.Th("F1"),
                        html.Th("Support"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        className="table table-sm table-striped supervised-table",
    )
    return html.Div(
        [
            html.Div("Per-class metrics", className="supervised-panel__title"),
            table,
        ]
        ,
        style={"padding": "0.25rem 0.1rem 0.1rem"},
    )


def _extract_class_labels(metrics: dict[str, Any], predictions: pd.DataFrame) -> list[object]:
    class_labels = metrics.get("class_labels")
    if isinstance(class_labels, list) and class_labels:
        return class_labels
    probability_labels = []
    for column in predictions.columns:
        if column.startswith("prob_class_"):
            probability_labels.append(column.removeprefix("prob_class_"))
    return probability_labels


def _build_strategic_insight(
    metrics: dict[str, Any], feature_importance: pd.DataFrame
) -> dict[str, str]:
    top_features = feature_importance["feature"].head(3).tolist()
    if top_features:
        top_driver = _compact_label(top_features[0])
        second_driver = _compact_label(top_features[1]) if len(top_features) > 1 else top_driver
        third_driver = _compact_label(top_features[2]) if len(top_features) > 2 else second_driver
    else:
        top_driver = second_driver = third_driver = "n/a"

    summary = (
        f"{metrics['best_model_name']} leads the pack with {metrics['best_model_accuracy']:.3f} accuracy and "
        f"{metrics['best_model_macro_f1']:.3f} macro F1. The strongest signals are {top_driver}, "
        f"{second_driver}, and {third_driver}, which means packaging cues and stated preferences matter more "
        f"than demographics alone."
    )
    return {
        "headline": "Packaging imagery is the strongest signal, not demographics alone.",
        "summary": summary,
        "top_driver": top_driver,
        "second_driver": second_driver,
        "action": "Lead with packaging-led creative",
    }


def _compact_label(value: str, max_length: int = 42) -> str:
    value = value.replace("บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ", "Packaging")
    value = value.replace("คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ", "Preference")
    value = value.replace("บรรจุภัณฑ์ (packaging) มีผลต่อการตัดสินใจซื้อใจหรือไม่", "Packaging matters")
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 1].rstrip()}…"

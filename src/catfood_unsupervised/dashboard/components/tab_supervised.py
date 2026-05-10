from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.components.supervised_form import (
    render_supervised_form,
    build_supervised_field_specs,
)
from catfood_unsupervised.dashboard.components.supervised_results import (
    render_prediction_result_panel,
    render_recent_history_panel,
)
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
)


def render_supervised_tab(
    bundle: SupervisedDashboardBundle,
    recent_history: pd.DataFrame | None = None,
) -> html.Div:
    metrics = bundle.metrics
    comparison = bundle.comparison
    confusion_matrix = bundle.confusion_matrix
    feature_importance = bundle.feature_importance
    predictions = bundle.predictions
    class_labels = _extract_class_labels(metrics, predictions)
    if recent_history is None:
        from catfood_unsupervised.supervised.history_store import (
            fetch_recent_prediction_history,
        )

        recent_history = fetch_recent_prediction_history(bundle.history_db_path, limit=8)

    hero = html.Div(
        [
            html.Div("5. Supervised Learning", className="supervised-hero__eyebrow"),
            html.H3(
                "Score a new customer row, then inspect the result and usage trail",
                className="supervised-hero__title",
            ),
            html.P(
                "The supervised model takes the canonical survey columns, predicts the segment, and stores each run in SQLite so business users can see how the tool is being used.",
                className="supervised-hero__lede",
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

    performance_strip = dbc.Row(
        [
            dbc.Col(_metric_card("Best Model", metrics["best_model_name"], "#2E86AB"), width=3),
            dbc.Col(_metric_card("Accuracy", f"{metrics['best_model_accuracy']:.3f}", "#2E86AB"), width=3),
            dbc.Col(_metric_card("Macro F1", f"{metrics['best_model_macro_f1']:.3f}", "#A23B72"), width=3),
            dbc.Col(_metric_card("Weighted F1", f"{metrics['best_model_weighted_f1']:.3f}", "#3A7D44"), width=3),
        ],
        className="g-3 mb-4",
    )

    main_workflow = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([render_supervised_form(build_supervised_field_specs(bundle.feature_options))]),
                    className="supervised-panel h-100",
                ),
                width=7,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(id="supervised-error-panel"),
                            html.Div(
                                render_prediction_result_panel(None, metrics),
                                id="supervised-result-panel",
                            ),
                            html.Div(
                                render_recent_history_panel(recent_history),
                                id="supervised-history-panel",
                            ),
                        ]
                    ),
                    className="supervised-panel h-100",
                ),
                width=5,
            ),
        ],
        className="g-3 mb-4",
    )

    insight_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.Div("Model comparison"), dcc.Graph(figure=_build_model_comparison_figure(comparison, metrics["best_model_name"]), config=_graph_config())]),
                    className="supervised-panel h-100",
                ),
                width=12,
            )
        ],
        className="g-3 mb-4",
    )

    diagnostics_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([dcc.Graph(figure=_build_confusion_matrix_figure(confusion_matrix), config=_graph_config())]),
                    className="supervised-panel h-100",
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([dcc.Graph(figure=_build_feature_importance_figure(feature_importance), config=_graph_config())]),
                    className="supervised-panel h-100",
                ),
                width=6,
            ),
        ],
        className="g-3 mb-4",
    )

    roc_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([dcc.Graph(figure=_build_roc_figure(predictions, class_labels), config=_graph_config())]),
                    className="supervised-panel h-100",
                ),
                width=12,
            )
        ],
        className="g-3 mb-4",
    )

    return html.Div(
        [
            hero,
            main_workflow,
            performance_strip,
            html.Div(
                [
                    html.Div("Supervised diagnostics", className="supervised-panel__title"),
                    html.P(
                        "These training-time charts stay available for deeper validation, but the form and result workflow stays visually separate.",
                        className="supervised-panel__lede",
                    ),
                ],
                className="mb-2",
            ),
            insight_row,
            diagnostics_row,
            roc_row,
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


def _extract_class_labels(metrics: dict[str, Any], predictions: pd.DataFrame) -> list[object]:
    class_labels = metrics.get("class_labels")
    if isinstance(class_labels, list) and class_labels:
        return class_labels
    probability_labels = []
    for column in predictions.columns:
        if column.startswith("prob_class_"):
            probability_labels.append(column.removeprefix("prob_class_"))
    return probability_labels

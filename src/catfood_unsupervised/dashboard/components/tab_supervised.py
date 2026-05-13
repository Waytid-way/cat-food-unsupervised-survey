from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.components.supervised_form import (
    _SHORT_LABELS as SUPERVISED_SHORT_LABELS,
    render_supervised_form,
)
from catfood_unsupervised.dashboard.components.supervised_results import (
    render_prediction_result_panel,
    render_recent_history_panel,
    render_supervised_metric_cards,
)
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
)
from catfood_unsupervised.supervised.history_store import (
    fetch_recent_prediction_history,
)


def render_supervised_tab(bundle: SupervisedDashboardBundle) -> html.Div:
    history_frame = fetch_recent_prediction_history(bundle.history_db_path, limit=8)
    form_panel = render_supervised_form(
        bundle.feature_options,
        model_path_text=str(bundle.model_path),
        history_path_text=str(bundle.history_db_path),
    )

    comparison_card = _render_comparison_card(bundle.comparison)
    confusion_card = _render_confusion_card(bundle.confusion_matrix)
    importance_card = _render_importance_card(bundle.feature_importance)

    return html.Div(
        [
            render_supervised_metric_cards(bundle.metrics),
            dbc.Row(
                [
                dbc.Col(form_panel, width=7),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(id="supervised-error-panel"),
                                render_prediction_result_panel(None, bundle.metrics),
                                html.Div(className="mt-3"),
                                render_recent_history_panel(history_frame),
                            ]
                        ),
                        width=5,
                    ),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(comparison_card, width=12),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(confusion_card, width=6),
                    dbc.Col(importance_card, width=6),
                ],
                className="mb-4 g-3",
            ),
        ],
        className="shell-content",
    )


def _render_comparison_card(comparison: pd.DataFrame) -> html.Div:
    if comparison.empty:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Model Performance Comparison", className="page-title"),
                    html.P("Run the supervised pipeline to generate comparison metrics.", className="text-muted"),
                ]
            ),
            className="chart-card",
        )

    comparison_display = comparison.copy()
    model_display_names = {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svm_rbf": "SVM RBF",
        "logistic_regression": "Logistic Regression",
    }
    comparison_display["model"] = comparison_display["model_name"].map(model_display_names)
    fig = px.bar(
        comparison_display,
        x="model",
        y=["accuracy", "macro_f1", "weighted_f1"],
        title="Model Performance Comparison",
        barmode="group",
        color="model",
        labels={"model": "Model", "value": "Score", "variable": "Metric"},
    )
    fig.update_layout(legend_title="Metric", plot_bgcolor="white")
    return dbc.Card(
        dbc.CardBody([dcc.Graph(figure=fig, className="chart-card")]),
        className="chart-card",
    )


def _render_confusion_card(confusion: pd.DataFrame) -> html.Div:
    if confusion.empty:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Confusion Matrix", className="page-title"),
                    html.P("No confusion matrix is available yet.", className="text-muted"),
                ]
            ),
            className="chart-card",
        )

    fig = px.imshow(
        confusion.values,
        x=list(confusion.columns),
        y=list(confusion.index),
        color_continuous_scale="Viridis",
        title="Confusion Matrix",
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(margin=dict(t=50, b=80))
    return dbc.Card(
        dbc.CardBody([dcc.Graph(figure=fig, className="chart-card")]),
        className="chart-card",
    )


def _render_importance_card(feature_importance: pd.DataFrame) -> html.Div:
    if feature_importance.empty:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Feature Importance", className="page-title"),
                    html.P("No feature importance output yet.", className="text-muted"),
                ]
            ),
            className="chart-card",
        )

    fi_filtered = feature_importance.loc[feature_importance["importance_mean"] > 0].copy()
    if fi_filtered.empty:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Feature Importance", className="page-title"),
                    html.P("No positive feature importance output yet.", className="text-muted"),
                ]
            ),
            className="chart-card",
        )

    fi_sorted = fi_filtered.sort_values("importance_mean", ascending=True).copy()
    fi_sorted["feature_label"] = fi_sorted["feature"].map(_feature_label_for_chart)
    fig = px.bar(
        fi_sorted,
        y="feature_label",
        x="importance_mean",
        orientation="h",
        title="Feature Importance",
        color="importance_mean",
        color_continuous_scale="Viridis",
        hover_data={"feature": True, "feature_label": False, "importance_std": ":.4f"},
        labels={"feature_label": "Question", "feature": "Full question", "importance_mean": "Importance"},
    )
    fig.update_layout(plot_bgcolor="white", showlegend=False, yaxis=dict(automargin=True))
    return dbc.Card(
        dbc.CardBody([dcc.Graph(figure=fig, className="chart-card")]),
        className="chart-card",
    )


def _feature_label_for_chart(feature_name: object) -> str:
    if feature_name is None:
        return "Unknown feature"
    text = str(feature_name)
    return SUPERVISED_SHORT_LABELS.get(text, text)


__all__ = ["render_supervised_tab"]

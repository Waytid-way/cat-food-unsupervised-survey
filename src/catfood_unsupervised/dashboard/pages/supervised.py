from __future__ import annotations

from pathlib import Path
import os

from dash import html, dcc, register_page
import dash_bootstrap_components as dbc
import plotly.express as px

from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_dashboard_bundle

SUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs")) / "supervised"


def _render():
    try:
        data = load_supervised_dashboard_bundle(SUPERVISED_OUTPUT_DIR)
    except Exception:
        return html.Div("Supervised data not available. Please run the supervised pipeline first.")

    comparison = data.comparison
    confusion = data.confusion_matrix
    feature_importance = data.feature_importance

    model_display_names = {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svm_rbf": "SVM RBF",
        "logistic_regression": "Logistic Regression",
    }
    model_colors = {
        "Random Forest": "#2E86AB",
        "Gradient Boosting": "#A23B72",
        "SVM RBF": "#F18F01",
        "Logistic Regression": "#3A7D44",
    }
    comparison_display = comparison.copy()
    comparison_display["model"] = comparison_display["model_name"].map(model_display_names)

    bar_fig = px.bar(
        comparison_display,
        x="model",
        y=["accuracy", "macro_f1", "weighted_f1"],
        title="Model Performance Comparison",
        barmode="group",
        color="model",
        color_discrete_map=model_colors,
        labels={"model": "Model", "value": "Score", "variable": "Metric"},
    )
    bar_fig.update_layout(legend_title="Metric", plot_bgcolor="white")

    confusion_fig = px.imshow(
        confusion.values,
        x=list(confusion.columns),
        y=list(confusion.index),
        color_continuous_scale="Viridis",
        title="Confusion Matrix",
        aspect="auto",
        text_auto=True,
    )
    confusion_fig.update_layout(margin=dict(t=50, b=80))

    fi_sorted = feature_importance.sort_values("importance_mean", ascending=True)
    fi_fig = px.bar(
        fi_sorted,
        y="feature",
        x="importance_mean",
        orientation="h",
        title="Feature Importance",
        color="importance_mean",
        color_continuous_scale="Viridis",
        labels={"feature": "Feature", "importance_mean": "Importance"},
    )
    fi_fig.update_layout(plot_bgcolor="white", showlegend=False)

    metrics = data.metrics
    accuracy = metrics.get("test_accuracy", 0)
    precision = metrics.get("weighted_precision", 0)
    recall = metrics.get("weighted_recall", 0)
    f1 = metrics.get("weighted_f1", 0)

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Accuracy", className="page-title"), html.H3(f"{accuracy:.1%}", style={"color": "#2E86AB"})]), className="ind-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Precision", className="page-title"), html.H3(f"{precision:.1%}", style={"color": "#F18F01"})]), className="ind-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Recall", className="page-title"), html.H3(f"{recall:.1%}", style={"color": "#A23B72"})]), className="ind-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("F1 Score", className="page-title"), html.H3(f"{f1:.1%}", style={"color": "#3A7D44"})]), className="ind-card"), width=3),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=bar_fig, className="chart-card")]), className="chart-card"), width=12),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=confusion_fig, className="chart-card")]), className="chart-card"), width=6),
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fi_fig, className="chart-card")]), className="chart-card"), width=6),
                ],
                className="mb-4",
            ),
        ],
        className="shell-content",
    )


register_page(__name__, path="/supervised", title="Supervised Learning", layout=_render)
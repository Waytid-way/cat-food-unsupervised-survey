from __future__ import annotations

from pathlib import Path
import os

from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

UNSUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))
SUPERVISED_OUTPUT_DIR = UNSUPERVISED_OUTPUT_DIR / "supervised"


def _get_kpi_value(metrics: dict, item: dict) -> str:
    try:
        node = metrics
        for key in item["metric_path"]:
            node = node[key]
        format_str = item["format"]
        if format_str.startswith(".1%") and isinstance(node, (int, float)):
            node = node * 100
            return f"{node:.1f}%"
        elif format_str.startswith(".3f"):
            return f"{node:.3f}"
        elif format_str.startswith(".1f"):
            return f"{node:.1f}"
        return str(node)
    except Exception:
        return "N/A"


def _render():
    from catfood_unsupervised.dashboard.data_loader import load_all_data

    try:
        data = load_all_data(UNSUPERVISED_OUTPUT_DIR)
        metrics = data.metrics
    except Exception:
        metrics = {}

    kpi_items = [
        {"id": "silhouette", "label": "Silhouette Score (k=2)", "metric_path": ["kmeans_evaluation", 0, "silhouette_score"], "format": ".3f", "color": "#2E86AB"},
        {"id": "davies_bouldin", "label": "Davies-Bouldin Index", "metric_path": ["kmeans_evaluation", 0, "davies_bouldin_score"], "format": ".3f", "color": "#A23B72"},
        {"id": "inertia", "label": "Inertia (k=2)", "metric_path": ["kmeans_evaluation", 0, "inertia"], "format": ".1f", "color": "#F18F01"},
        {"id": "variance_explained", "label": "Variance Explained (8 PCs)", "metric_path": ["pca", "total_explained_variance_ratio"], "format": ".1%", "color": "#C73E1D"},
        {"id": "anomaly_rate", "label": "Anomaly Rate", "metric_path": ["anomaly_detection", "anomaly_rate"], "format": ".1%", "color": "#3A7D44"},
    ]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.P(item["label"], className="kpi-label"),
                                html.P(_get_kpi_value(metrics, item), className="kpi-value", style={"color": item["color"]}),
                            ]),
                            className="kpi-card",
                        ),
                        width=3,
                    )
                    for item in kpi_items
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Clustering", className="page-title"), html.P("PCA, K-Means, and cluster evaluation", className="text-muted")]), className="ind-card"), width=6),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Correlation", className="page-title"), html.P("Option relationships and preference clusters", className="text-muted")]), className="ind-card"), width=6),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Persona", className="page-title"), html.P("Segment profile deep-dive", className="text-muted")]), className="ind-card"), width=6),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Supervised", className="page-title"), html.P("Prediction workflow and model validation", className="text-muted")]), className="ind-card"), width=6),
                ],
                className="mb-4 g-3",
            ),
        ],
        className="shell-content",
    )


register_page(__name__, path="/", title="Home", layout=_render)
from __future__ import annotations

import os
from pathlib import Path

from dash import dcc, html, dash_table as dt, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from catfood_unsupervised.dashboard.data_loader import load_all_data
from catfood_unsupervised.unsupervised.reporting import BUY_FACTOR_LABELS, PACKAGING_IMPORTANCE_LABELS

UNSUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs")) / "unsupervised"

_HEATMAP_LABELS = {
    **{f"vote_{index:02d}": f"คะแนนโหวต: Option {index}" for index in range(1, 11)},
    "packaging_effect_01": "บรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อ",
    **{
        f"buy_factor_{index:02d}": label
        for index, label in enumerate(BUY_FACTOR_LABELS, start=1)
    },
    **{
        f"packaging_importance_{index:02d}": label
        for index, label in enumerate(PACKAGING_IMPORTANCE_LABELS, start=1)
    },
}


def render():
    try:
        data = load_all_data(UNSUPERVISED_OUTPUT_DIR)
    except Exception:
        return html.Div("Data not available. Please run the pipeline first.")

    metrics = data.metrics
    clean_df = data.clean_df
    segment_profiles = data.segment_profiles
    final_cluster_k = int(metrics.get("final_cluster_k", clean_df["segment"].nunique()))

    ev_ratio = metrics["pca"]["explained_variance_ratio"]
    cumulative = metrics["pca"]["cumulative_explained_variance"]
    components = list(range(1, len(ev_ratio) + 1))

    scree_fig = px.bar(
        x=components,
        y=[v * 100 for v in ev_ratio],
        title="PCA Scree Plot - Variance Explained per Component",
        labels={"x": "Principal Component", "y": "Variance Explained (%)"},
        color=[v * 100 for v in ev_ratio],
        color_continuous_scale="Viridis",
    )
    for i, (comp, cum) in enumerate(zip(components, cumulative)):
        scree_fig.add_annotation(
            x=comp,
            y=(ev_ratio[i] * 100) + 0.5,
            text=f"{cum * 100:.1f}%",
            showarrow=False,
            font=dict(size=9),
        )

    pca_df = data.pca_scores.copy()
    pca_df["segment"] = clean_df["segment"].values
    segment_colors = ["#2E86AB", "#F18F01", "#A23B72", "#3A7D44", "#C73E1D"]
    color_map = {
        segment: color
        for segment, color in zip(sorted(pd.unique(clean_df["segment"])), segment_colors)
    }
    scatter_fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="segment",
        title=f"PC1 vs PC2 - K-Means Clusters (k={final_cluster_k})",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map=color_map,
    )

    corr = data.correlation_matrix
    corr_fig = px.imshow(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        color_continuous_scale="Viridis",
        title="Correlation Heatmap",
        aspect="auto",
    )
    corr_fig.update_layout(margin=dict(t=50, b=80))

    segment_size_df = segment_profiles[["segment_size"]].copy()
    segment_size_df.index.name = "segment"
    segment_size_df = segment_size_df.reset_index()
    segment_size_fig = px.bar(
        segment_size_df,
        x="segment",
        y="segment_size",
        text="segment_size",
        title="Segment Size Distribution",
        color="segment",
        color_discrete_sequence=["#2E86AB", "#F18F01", "#A23B72", "#3A7D44", "#C73E1D"],
    )
    segment_size_fig.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        xaxis_title="Segment",
        yaxis_title="Respondents",
    )

    heatmap_columns = [
        column
        for column in segment_profiles.columns
        if column.startswith(("vote_", "buy_factor_", "packaging_effect_", "packaging_importance_"))
    ]
    heatmap_df = segment_profiles[heatmap_columns].T
    heatmap_df.index = [_friendly_heatmap_label(column) for column in heatmap_df.index]
    heatmap_df.columns = [f"Segment {segment}" for segment in heatmap_df.columns]
    heatmap_fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Segment Profile Heatmap",
        labels={"x": "Segment", "y": "Feature", "color": "Score"},
    )
    heatmap_fig.update_layout(plot_bgcolor="white")

    ev = metrics.get("kmeans_evaluation", [])
    rows = []
    for row in ev:
        k = row.get("k", 0)
        sil = row.get("silhouette_score", 0)
        db = row.get("davies_bouldin_score", 0)
        inert = row.get("inertia", 0)
        rows.append(
            {
                "k": k,
                "Silhouette Score": round(sil, 3),
                "Davies-Bouldin Index": round(db, 3),
                "Inertia": round(inert, 1),
            }
        )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=scree_fig, className="chart-card")])), width=6),
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=scatter_fig, className="chart-card")])), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=segment_size_fig, className="chart-card")])), xs=12, lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=heatmap_fig, className="chart-card")])), xs=12, lg=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=corr_fig, className="chart-card")])), width=12),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("K-Means Evaluation Table", className="page-title"),
                                    dt.DataTable(
                                        columns=[{"name": c, "id": c} for c in ["k", "Silhouette Score", "Davies-Bouldin Index", "Inertia"]],
                                        data=rows,
                                        style_table={"width": "60%"},
                                        style_cell={"textAlign": "center"},
                                        style_header={"background": "#F8F9FA", "fontWeight": "bold"},
                                    ),
                                ]
                            ),
                            className="chart-card",
                        ),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
        ],
        className="shell-content",
    )


def _friendly_heatmap_label(column: str) -> str:
    return _HEATMAP_LABELS.get(column, column.replace("_", " "))


register_page(__name__, path="/unsupervised", title="Unsupervised Learning", layout=render)

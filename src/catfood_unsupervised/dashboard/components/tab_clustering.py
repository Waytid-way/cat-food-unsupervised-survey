from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from dash import dcc, dash_table as dt, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.data_loader import DashboardData


def _render_kmeans_table(metrics: dict) -> dt.DataTable:
    ev = metrics.get("kmeans_evaluation", [])
    rows = []
    best_silhouette = -1
    best_db = float("inf")
    best_inertia = float("inf")

    for row in ev:
        k = row.get("k", 0)
        sil = row.get("silhouette_score", 0)
        db = row.get("davies_bouldin_score", 0)
        inert = row.get("inertia", 0)
        rows.append({"k": k, "Silhouette Score": round(sil, 3), "Davies-Bouldin Index": round(db, 3), "Inertia": round(inert, 1)})
        if sil > best_silhouette:
            best_silhouette = sil
        if db < best_db:
            best_db = db
        if inert < best_inertia:
            best_inertia = inert

    for row in rows:
        sil = row["Silhouette Score"]
        row["Silhouette Score"] = str(sil) + (" ★" if sil == best_silhouette else "")
        db = row["Davies-Bouldin Index"]
        row["Davies-Bouldin Index"] = str(db) + (" ★" if db == best_db else "")
        inert = row["Inertia"]
        row["Inertia"] = str(inert) + (" ★" if inert == best_inertia else "")

    return dt.DataTable(
        columns=[{"name": c, "id": c} for c in ["k", "Silhouette Score", "Davies-Bouldin Index", "Inertia"]],
        data=rows,
        style_table={"width": "60%"},
        style_cell={"textAlign": "center"},
        style_header={"background": "#F8F9FA", "fontWeight": "bold"},
    )


def _make_dendrogram_fig(linkage_matrix):
    d = sch.dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
    fig = go.Figure()
    for i in range(len(d["icoord"])):
        fig.add_trace(
            go.Scatter(
                x=d["icoord"][i],
                y=d["dcoord"][i],
                mode="lines",
                showlegend=False,
                line=dict(color="#2E86AB", width=1.5),
            )
        )
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram (Ward)",
        showlegend=False,
        xaxis=dict(showticklabels=False, title="", showgrid=False),
        yaxis=dict(title="Distance", showgrid=True),
        plot_bgcolor="white",
        width=900,
        height=400,
    )
    return fig


def _get_nested(metrics: dict, path: list) -> Any:
    node = metrics
    for key in path:
        node = node[key]
    return node


def render_clustering_tab(data: DashboardData) -> html.Div:
    metrics = data.metrics
    clean_df = data.clean_df

    ev_ratio = metrics["pca"]["explained_variance_ratio"]
    cumulative = metrics["pca"]["cumulative_explained_variance"]
    components = list(range(1, len(ev_ratio) + 1))

    scree_fig = px.bar(
        x=components,
        y=[v * 100 for v in ev_ratio],
        title="PCA Scree Plot — Variance Explained per Component",
        labels={"x": "Principal Component", "y": "Variance Explained (%)"},
        color=[v * 100 for v in ev_ratio],
        color_continuous_scale="Viridis",
    )
    for i, (comp, cum) in enumerate(zip(components, cumulative)):
        scree_fig.add_annotation(
            x=comp, y=(ev_ratio[i] * 100) + 0.5,
            text=f"{cum * 100:.1f}%",
            showarrow=False,
            font=dict(size=9),
        )

    pca_df = data.pca_scores.copy()
    pca_df["segment"] = clean_df["segment"].values
    scatter_fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="segment",
        title="PC1 vs PC2 — K-Means Clusters (k=2)",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map={1: "#2E86AB", 2: "#F18F01"},
    )

    ks = [row["k"] for row in metrics["kmeans_evaluation"]]
    inertias = [row["inertia"] for row in metrics["kmeans_evaluation"]]
    silhouettes = [row["silhouette_score"] for row in metrics["kmeans_evaluation"]]
    db_scores = [row["davies_bouldin_score"] for row in metrics["kmeans_evaluation"]]

    elbow_fig = px.line(
        x=ks, y=inertias,
        title="Elbow Chart — Inertia vs k",
        labels={"x": "k", "y": "Inertia"},
        markers=True,
    )
    elbow_fig.update_traces(line_color="#2E86AB")

    silhouette_fig = px.line(
        x=ks, y=silhouettes,
        title="Silhouette Score vs k",
        labels={"x": "k", "y": "Silhouette Score"},
        markers=True,
    )
    silhouette_fig.update_traces(line_color="#A23B72")

    db_fig = px.line(
        x=ks, y=db_scores,
        title="Davies-Bouldin Index vs k",
        labels={"x": "k", "y": "Davies-Bouldin Index"},
        markers=True,
    )
    db_fig.update_traces(line_color="#F18F01")

    option_cols = [c for c in clean_df.columns if c.startswith("option_") and "_ips" in c]
    X_for_hier = clean_df[option_cols].values
    linkage_matrix = sch.linkage(X_for_hier, method="ward")

    dendrogram_fig = _make_dendrogram_fig(linkage_matrix)

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=scree_fig), width=6),
                    dbc.Col(dcc.Graph(figure=scatter_fig), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=elbow_fig), width=4),
                    dbc.Col(dcc.Graph(figure=silhouette_fig), width=4),
                    dbc.Col(dcc.Graph(figure=db_fig), width=4),
                ],
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(figure=dendrogram_fig), width=12),
            ),
            html.H5("K-Means Evaluation Table", className="section-header mt-4"),
            dbc.Row(dbc.Col(_render_kmeans_table(metrics), width=12)),
        ],
        id="tab_clustering",
    )

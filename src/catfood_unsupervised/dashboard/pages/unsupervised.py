from __future__ import annotations

from dash import dcc, html, dash_table as dt, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import scipy.cluster.hierarchy as sch
import plotly.graph_objects as go

from catfood_unsupervised.dashboard.data_loader import load_all_data

from pathlib import Path
import os

UNSUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))


def render():
    try:
        data = load_all_data(UNSUPERVISED_OUTPUT_DIR)
    except Exception:
        return html.Div("Data not available. Please run the pipeline first.")

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
        scree_fig.add_annotation(x=comp, y=(ev_ratio[i] * 100) + 0.5, text=f"{cum * 100:.1f}%", showarrow=False, font=dict(size=9))

    pca_df = data.pca_scores.copy()
    pca_df["segment"] = clean_df["segment"].values
    scatter_fig = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="segment",
        title="PC1 vs PC2 — K-Means Clusters (k=2)",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map={1: "#2E86AB", 2: "#F18F01"},
    )

    option_cols = [c for c in clean_df.columns if c.startswith("option_") and "_ips" in c]
    X_for_hier = clean_df[option_cols].values
    linkage_matrix = sch.linkage(X_for_hier, method="ward")

    def make_dendrogram_fig(linkage_matrix):
        d = sch.dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
        fig = go.Figure()
        for i in range(len(d["icoord"])):
            fig.add_trace(go.Scatter(x=d["icoord"][i], y=d["dcoord"][i], mode="lines", showlegend=False, line=dict(color="#2E86AB", width=1.5)))
        fig.update_layout(title="Hierarchical Clustering Dendrogram (Ward)", showlegend=False, xaxis=dict(showticklabels=False, title="", showgrid=False), yaxis=dict(title="Distance", showgrid=True), plot_bgcolor="white", height=400)
        return fig

    dendrogram_fig = make_dendrogram_fig(linkage_matrix)

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

    ev = metrics.get("kmeans_evaluation", [])
    rows = []
    for row in ev:
        k = row.get("k", 0)
        sil = row.get("silhouette_score", 0)
        db = row.get("davies_bouldin_score", 0)
        inert = row.get("inertia", 0)
        rows.append({"k": k, "Silhouette Score": round(sil, 3), "Davies-Bouldin Index": round(db, 3), "Inertia": round(inert, 1)})

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
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=dendrogram_fig, className="chart-card")])), width=12),
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
                            dbc.CardBody([
                                html.H5("K-Means Evaluation Table", className="page-title"),
                                dt.DataTable(
                                    columns=[{"name": c, "id": c} for c in ["k", "Silhouette Score", "Davies-Bouldin Index", "Inertia"]],
                                    data=rows,
                                    style_table={"width": "60%"},
                                    style_cell={"textAlign": "center"},
                                    style_header={"background": "#F8F9FA", "fontWeight": "bold"},
                                ),
                            ]),
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


register_page(__name__, path="/unsupervised", title="Unsupervised Learning", layout=render)
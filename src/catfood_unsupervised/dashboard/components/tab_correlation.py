from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


def render_correlation_tab(data: DashboardData) -> html.Div:
    corr = data.correlation_matrix

    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Spearman Correlation Heatmap — 10 Packaging Options",
        labels={"x": "Option", "y": "Option", "color": "r"},
    )
    fig.update_layout(
        width=900,
        height=700,
        xaxis_tickangle=45,
        font=dict(size=10),
    )

    cluster_text = [
        html.H5("Correlation Clusters", className="mt-3"),
        html.Ul([
            html.Li([html.Strong("Classic/Safe (Blue):"), " Option 01 × Option 02 — high r"]),
            html.Li([html.Strong("Mainstream (Orange):"), " Option 03 × 04 × 05 — cluster together"]),
            html.Li([html.Strong("Niche Aesthetic (Purple):"), " Option 08 × 09 × 10 — distinct preference"]),
        ]),
    ]

    return html.Div(
        [
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12),
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(cluster_text, width=8),
            ),
        ],
        id="tab_correlation",
    )
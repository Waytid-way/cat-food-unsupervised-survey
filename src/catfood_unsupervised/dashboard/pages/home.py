from __future__ import annotations

from pathlib import Path
import os

from dash import html, dcc, register_page
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))
UNSUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "unsupervised"
SUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "supervised"


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
        clean_df = data.clean_df
    except Exception:
        metrics = {}
        clean_df = pd.DataFrame()

    kpi_items = [
        {"id": "silhouette", "label": "Silhouette Score (k=2)", "metric_path": ["kmeans_evaluation", 0, "silhouette_score"], "format": ".3f", "color": "#2E86AB"},
        {"id": "davies_bouldin", "label": "Davies-Bouldin Index", "metric_path": ["kmeans_evaluation", 0, "davies_bouldin_score"], "format": ".3f", "color": "#A23B72"},
        {"id": "inertia", "label": "Inertia (k=2)", "metric_path": ["kmeans_evaluation", 0, "inertia"], "format": ".1f", "color": "#F18F01"},
        {"id": "variance_explained", "label": "Variance Explained (8 PCs)", "metric_path": ["pca", "total_explained_variance_ratio"], "format": ".1%", "color": "#C73E1D"},
        {"id": "anomaly_rate", "label": "Anomaly Rate", "metric_path": ["anomaly_detection", "anomaly_rate"], "format": ".1%", "color": "#3A7D44"},
    ]

    snapshot_section = _render_respondent_snapshot(clean_df)

    return html.Div(
        [
            snapshot_section,
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


def _render_respondent_snapshot(clean_df: pd.DataFrame) -> html.Div:
    charts = [
        _build_distribution_chart(
            clean_df,
            title="Gender Distribution",
            chart_kind="pie",
            category_column="เพศของคุณ",
            category_values=("ชาย", "หญิง", "อื่นๆ"),
            labels=("ชาย", "หญิง", "อื่นๆ"),
            columns=("gender_ชาย", "gender_หญิง", "gender_อื่นๆ"),
            fallback_labels=("ชาย", "หญิง", "อื่นๆ"),
            colors=("#2E86AB", "#F18F01", "#A23B72"),
        ),
        _build_distribution_chart(
            clean_df,
            title="Age Distribution",
            chart_kind="bar",
            category_column="อายุของคุณ",
            category_values=("20-29ปี", "30-39ปี", "40-49ปี", "50ปี ขึ้นไป"),
            labels=("20-29 ปี", "30-39 ปี", "40-49 ปี", "50 ปีขึ้นไป"),
            columns=("age_20-29ปี", "age_30-39ปี", "age_40-49ปี", "age_50ปี ขึ้นไป"),
            fallback_labels=("20-29 ปี", "30-39 ปี", "40-49 ปี", "50 ปีขึ้นไป"),
            colors=("#2E86AB", "#3A7D44", "#F18F01", "#A23B72"),
        ),
        _build_distribution_chart(
            clean_df,
            title="Marital Status Distribution",
            chart_kind="pie",
            category_column="สถานภาพสมรส",
            category_values=("โสด ไม่มีแฟน", "มีแฟนแต่ยังไม่แต่งงาน", "แต่งงานแล้ว", "หย่าร้าง/เป็นม่าย"),
            labels=("โสด", "มีแฟนแต่ยังไม่แต่งงาน", "แต่งงานแล้ว", "หย่าร้าง/เป็นม่าย"),
            columns=(
                "marital_โสด ไม่มีแฟน",
                "marital_มีแฟนแต่ยังไม่แต่งงาน",
                "marital_แต่งงานแล้ว",
                "marital_หย่าร้าง/เป็นม่าย",
            ),
            fallback_labels=("โสด", "มีแฟนแต่ยังไม่แต่งงาน", "แต่งงานแล้ว", "หย่าร้าง/เป็นม่าย"),
            colors=("#2E86AB", "#F18F01", "#3A7D44", "#A23B72"),
        ),
    ]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Respondent Snapshot", className="page-title"),
                                    html.P(
                                        "Quick view of the survey audience composition from the cleaned dataset.",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                            className="ind-card",
                        ),
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(dbc.CardBody([dcc.Graph(figure=figure, className="chart-card")]), className="chart-card"),
                        xs=12,
                        lg=4,
                    )
                    for figure in charts
                ],
                className="mb-4 g-3",
            ),
        ]
    )


def _build_distribution_chart(
    clean_df: pd.DataFrame,
    *,
    title: str,
    chart_kind: str,
    category_column: str | None = None,
    category_values: tuple[str, ...] = (),
    labels: tuple[str, ...] = (),
    columns: tuple[str, ...],
    fallback_labels: tuple[str, ...],
    colors: tuple[str, ...],
):
    if category_column and category_column in clean_df.columns:
        series = clean_df[category_column].dropna().astype(str).str.strip()
        chart_rows = []
        for raw_value, label in zip(category_values, labels, strict=False):
            count = int(series.eq(raw_value).sum())
            if count > 0:
                chart_rows.append({"Category": label, "Count": count})
        chart_df = pd.DataFrame(chart_rows)
    else:
        counts = []
        for column in columns:
            if column not in clean_df.columns:
                counts.append(0)
                continue
            series = pd.to_numeric(clean_df[column], errors="coerce").fillna(0)
            counts.append(float(series.sum()))
        chart_df = pd.DataFrame({"Category": fallback_labels, "Count": counts})
        chart_df = chart_df[chart_df["Count"] > 0]

    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        fig.add_annotation(text="No demographic data available", x=0.5, y=0.5, showarrow=False)
        return fig

    if chart_kind == "bar":
        fig = px.bar(
            chart_df,
            x="Category",
            y="Count",
            title=title,
            color="Category",
            color_discrete_sequence=[colors[0]],
            text="Count",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Count",
            xaxis_tickangle=-15,
        )
    else:
        fig = px.pie(
            chart_df,
            names="Category",
            values="Count",
            title=title,
            hole=0.55,
            color_discrete_sequence=colors,
        )
        fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
    )
    return fig


register_page(__name__, path="/", title="Home", layout=_render)

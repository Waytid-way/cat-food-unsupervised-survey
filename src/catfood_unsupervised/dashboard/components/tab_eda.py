from __future__ import annotations

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.data_loader import DashboardData


def _build_top3_breakdown_figure(clean_df: pd.DataFrame) -> go.Figure:
    options = [f"Opt {i+1:02d}" for i in range(10)]

    fig = go.Figure()
    for rank_idx, (col_idx, label, color) in enumerate([
        (79, "Rank 1 (3 pts)", "#2E86AB"),
        (80, "Rank 2 (2 pts)", "#A23B72"),
        (81, "Rank 3 (1 pt)", "#F18F01"),
    ]):
        col = clean_df.columns[col_idx]
        vc = clean_df[col].astype(str).str.strip().value_counts()
        values = [vc.get(str(i), 0) for i in range(1, 11)]
        fig.add_trace(go.Bar(
            name=label,
            x=options,
            y=values,
            marker_color=color,
        ))

    fig.update_layout(
        title="Top-3 Ranking Breakdown by Option",
        barmode="group",
        height=350,
        plot_bgcolor="white",
        showlegend=True,
    )
    return fig


def _build_option_rating_heatmap_figure(clean_df: pd.DataFrame) -> go.Figure:
    opt_means = clean_df[[c for c in clean_df.columns if c.startswith("option_") and "_attribute_" in c]].mean()

    matrix = []
    for opt in range(1, 11):
        row = [opt_means.get(f"option_{opt:02d}_attribute_{attr:02d}", 0) for attr in range(1, 6)]
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
        y=[f"Opt {i:02d}" for i in range(1, 11)],
        colorscale="RdYlGn",
        zmin=1, zmax=5,
        colorbar_title="Rating",
    ))
    fig.update_layout(
        title="Option Rating Distribution (Mean Agreement 1-5)",
        height=400,
        plot_bgcolor="white",
    )
    return fig


def render_eda_tab(data: DashboardData) -> html.Div:
    clean_df = data.clean_df

    age_col = clean_df.columns[73]
    gender_col = clean_df.columns[74]
    marital_col = clean_df.columns[75]

    age_counts = clean_df[age_col].astype(str).str.strip().value_counts()
    gender_counts = clean_df[gender_col].astype(str).str.strip().value_counts()
    marital_counts = clean_df[marital_col].astype(str).str.strip().value_counts()

    age_labels = {"20-29฿": "20-29", "30-39฿": "30-39", "40-49฿": "40-49", "50+ ฿": "50+"}
    age_fig = px.bar(
        x=[age_labels.get(k, k) for k in age_counts.index],
        y=list(age_counts.values),
        title="อายุ",
        labels={"x": "ช่วงอายุ", "y": "จำนวน"},
        color=list(age_counts.values),
        color_discrete_sequence=["#2E86AB"],
    )
    gender_fig = px.bar(
        x=list(gender_counts.index),
        y=list(gender_counts.values),
        title="เพศ",
        labels={"x": "เพศ", "y": "จำนวน"},
        color=list(gender_counts.index),
        color_discrete_sequence=["#2E86AB", "#A23B72", "#F18F01"],
    )
    marital_fig = px.bar(
        x=list(marital_counts.index),
        y=list(marital_counts.values),
        title="สถานภาพสมรส",
        labels={"x": "สถานภาพ", "y": "จำนวน"},
        color=list(marital_counts.values),
        color_discrete_sequence=["#F18F01"],
    )

    buy_factor_labels = [
        "ใช้วัตถุดิบจากธรรมชาติ",
        "ใช้วัตถุดิบนำเข้า",
        "รสชาติกลมกล่อม",
        "ผลิตภัณฑ์จากต่างประเทศ",
        "แบรนด์มีชื่อเสียง",
    ]
    buy_factor_figs = []
    for i, label in enumerate(buy_factor_labels):
        col = clean_df.columns[5 + i]
        vc = clean_df[col].astype(str).str.strip().value_counts()
        fig = px.bar(
            x=list(vc.index),
            y=list(vc.values),
            title=label,
            labels={"x": "", "y": "จำนวน"},
            color=list(vc.values),
            color_discrete_sequence=["#2E86AB"],
        )
        buy_factor_figs.append(dcc.Graph(figure=fig))

    vote_cols = [clean_df.columns[i] for i in range(79, 89)]
    vote_sums = clean_df[vote_cols].sum().sort_values(ascending=False)
    vote_fig = px.bar(
        x=[f"Opt {int(c.split('_')[1]):02d}" for c in vote_sums.index],
        y=vote_sums.values,
        title="Weighted Vote (Top-3)",
        labels={"x": "ตัวเลือก", "y": "คะแนนถ่วงน้ำหนัก"},
        color=vote_sums.values,
        color_continuous_scale="Reds",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=gender_fig), width=4),
                    dbc.Col(dcc.Graph(figure=age_fig), width=4),
                    dbc.Col(dcc.Graph(figure=marital_fig), width=4),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [dbc.Col(fig, width=4) for fig in buy_factor_figs],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=vote_fig), width=6),
                ],
            ),
            html.H5("Top-3 Ranking Breakdown", className="section-header mt-4"),
            dbc.Row(dbc.Col(dcc.Graph(figure=_build_top3_breakdown_figure(clean_df)), width=12)),
            html.H5("Option Rating Distribution", className="section-header mt-4"),
            dbc.Row(dbc.Col(dcc.Graph(figure=_build_option_rating_heatmap_figure(clean_df)), width=12)),
        ],
        id="tab_eda",
    )

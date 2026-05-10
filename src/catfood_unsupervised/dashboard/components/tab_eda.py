from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


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
        ],
        id="tab_eda",
    )
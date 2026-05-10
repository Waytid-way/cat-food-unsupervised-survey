from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from catfood_unsupervised.dashboard.components.shared import segment_color_map
from catfood_unsupervised.dashboard.data_loader import DashboardData


SEGMENT_STRATEGIES = {
    1: {
        "title": "กลุ่มพรีเมียมเน้นคุณภาพและดีไซน์",
        "description": "ให้ความสำคัญสูงกับรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์บนแพ็กเกจ",
        "color": "#2E86AB",
        "strategy": "สื่อสารแบบ premium ให้ชัดเจน เน้นรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์ที่บอกประโยชน์ของสินค้า",
    },
    2: {
        "title": "กลุ่มตลาดหลักที่ชั่งน้ำหนักหลายปัจจัย",
        "description": "ตอบสนองต่อ messaging กว้าง เน้นความคุ้มค่าและความน่าเชื่อถือ",
        "color": "#F18F01",
        "strategy": "ใช้ข้อความที่อ่านง่ายและมี social proof เน้นความคุ้มค่าและข้อมูลที่เข้าใจง่าย",
    },
}


def render_persona_tab(data: DashboardData) -> html.Div:
    clean_df = data.clean_df
    metrics = data.metrics

    anomaly_flag = clean_df["anomaly_flag"] if "anomaly_flag" in clean_df.columns else None
    segment = clean_df["segment"]

    scatter_df = data.pca_scores.copy()
    scatter_df["segment"] = segment.values
    if anomaly_flag is not None:
        scatter_df["anomaly"] = anomaly_flag.values
        scatter_df["anomaly_label"] = scatter_df["anomaly"].map({1: "Anomaly", 0: "Normal"})
    else:
        scatter_df["anomaly"] = 0
        scatter_df["anomaly_label"] = "Normal"

    scatter_fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color="segment",
        symbol="anomaly",
        title="Isolation Forest Anomalies Highlighted on PCA Space",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map={"1": "#2E86AB", "2": "#F18F01"},
        symbol_map={1: "circle", 0: "x"},
    )

    anomaly_count = metrics["anomaly_detection"]["anomaly_count"]
    anomaly_rate = metrics["anomaly_detection"]["anomaly_rate"] * 100

    persona_cards = []
    seg_colors = segment_color_map()

    for segment_id in [1, 2]:
        seg_df = clean_df[clean_df["segment"] == segment_id]
        seg_size = len(seg_df)
        seg_pct = seg_size / len(clean_df) * 100
        color = seg_colors.get(segment_id, "#2E86AB")

        gender_col = clean_df.columns[74]
        age_col = clean_df.columns[73]
        marital_col = clean_df.columns[75]

        gender_vc = seg_df[gender_col].astype(str).str.strip().value_counts()
        age_vc = seg_df[age_col].astype(str).str.strip().value_counts()
        marital_vc = seg_df[marital_col].astype(str).str.strip().value_counts()

        gender_fig = go.Figure(data=[go.Pie(
            labels=list(gender_vc.index),
            values=list(gender_vc.values),
            marker_colors=["#2E86AB", "#A23B72", "#F18F01"][:len(gender_vc)],
        )])
        gender_fig.update_layout(title="Gender", height=200, margin=dict(t=30, b=30))

        age_fig = go.Figure(data=[go.Pie(
            labels=[str(x) for x in age_vc.index],
            values=list(age_vc.values),
        )])
        age_fig.update_layout(title="Age", height=200, margin=dict(t=30, b=30))

        marital_fig = go.Figure(data=[go.Pie(
            labels=list(marital_vc.index),
            values=list(marital_vc.values),
        )])
        marital_fig.update_layout(title="Marital", height=200, margin=dict(t=30, b=30))

        buy_labels = ["Buy Factor 1", "Buy Factor 2", "Buy Factor 3", "Buy Factor 4", "Buy Factor 5"]
        buy_means = [seg_df[clean_df.columns[5+i]].astype(float).mean() for i in range(5)]

        buy_fig = go.Figure(data=[go.Bar(
            x=buy_labels,
            y=buy_means,
            marker_color=color,
        )])
        buy_fig.update_layout(title="Buy Factor Preferences (Mean)", height=220, plot_bgcolor="white")

        strategy_info = SEGMENT_STRATEGIES.get(segment_id, {})

        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"Segment {segment_id}: {strategy_info.get('title', '')}", style={"color": color}),
                html.P(strategy_info.get("description", "")),
                html.Hr(),
                html.P(f"Size: {seg_size} ({seg_pct:.1f}%)", className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=gender_fig), width=4),
                    dbc.Col(dcc.Graph(figure=age_fig), width=4),
                    dbc.Col(dcc.Graph(figure=marital_fig), width=4),
                ]),
                dbc.Row(dbc.Col(dcc.Graph(figure=buy_fig), width=12)),
                html.Hr(),
                html.Strong("Strategy: "),
                html.P(strategy_info.get("strategy", ""), className="text-muted"),
            ]),
            class_name=f"shadow-sm mb-3 segment-{segment_id}-accent",
            style={"border-left": f"4px solid {color}"},
        )
        persona_cards.append(dbc.Col(card, width=6))

    anomaly_banner = dbc.Alert(
        f"Detected {anomaly_count} anomalies ({anomaly_rate:.1f}% of respondents) — see highlighted points in scatter plot.",
        color="warning",
        className="mb-3",
    )

    return html.Div(
        [
            dbc.Row(dbc.Col(anomaly_banner, width=12), className="mb-3"),
            dbc.Row(dbc.Col(dcc.Graph(figure=scatter_fig), width=12), className="mb-4"),
            dbc.Row(persona_cards),
        ],
        id="tab_persona",
    )

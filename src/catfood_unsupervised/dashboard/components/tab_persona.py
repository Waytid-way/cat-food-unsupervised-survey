from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

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
    for segment_id, strategy_info in SEGMENT_STRATEGIES.items():
        seg_df = clean_df[clean_df["segment"] == segment_id]
        seg_size = len(seg_df)
        seg_pct = seg_size / len(clean_df) * 100
        top_gender = seg_df.iloc[:, 74].astype(str).str.strip().value_counts().index[0]
        top_age = seg_df.iloc[:, 73].astype(str).str.strip().value_counts().index[0]
        anomaly_rate_seg = (
            seg_df["anomaly_flag"].mean() * 100 if "anomaly_flag" in seg_df.columns else 0
        )
        card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        f"Segment {segment_id}: {strategy_info['title']}",
                        style={"color": strategy_info["color"]},
                    ),
                    html.P(strategy_info["description"]),
                    html.Hr(),
                    html.P(f"Size: {seg_size} ({seg_pct:.1f}%)", className="mb-1"),
                    html.P(f"Top Gender: {top_gender}", className="mb-1"),
                    html.P(f"Top Age: {top_age}", className="mb-1"),
                    html.P(f"Anomaly Rate: {anomaly_rate_seg:.1f}%", className="mb-1"),
                    html.Hr(),
                    html.Strong("Strategy: "),
                    html.P(strategy_info["strategy"], className="text-muted"),
                ]
            ),
            className="shadow-sm mb-3",
            style={"border-left": f"4px solid {strategy_info['color']}"},
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

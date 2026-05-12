from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import pandas as pd
from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
import plotly.express as px

from catfood_unsupervised.dashboard.data_loader import load_all_data
from catfood_unsupervised.unsupervised.reporting import BUY_FACTOR_LABELS, PACKAGING_IMPORTANCE_LABELS

UNSUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs")) / "unsupervised"

SEGMENT_COLORS = ["#2E86AB", "#F18F01", "#3A7D44", "#A23B72"]
BUY_FACTOR_LABEL_MAP = {
    f"buy_factor_{index:02d}": label for index, label in enumerate(BUY_FACTOR_LABELS, start=1)
}
PACKAGING_IMPORTANCE_LABEL_MAP = {
    f"packaging_importance_{index:02d}": label
    for index, label in enumerate(PACKAGING_IMPORTANCE_LABELS, start=1)
}
GENDER_LABEL_MAP = {
    "gender_ชาย": "ชาย",
    "gender_หญิง": "หญิง",
    "gender_อื่นๆ": "อื่นๆ",
}
AGE_LABEL_MAP = {
    "age_20-29ปี": "20-29 ปี",
    "age_30-39ปี": "30-39 ปี",
    "age_40-49ปี": "40-49 ปี",
    "age_50ปี ขึ้นไป": "50 ปีขึ้นไป",
}
MARITAL_LABEL_MAP = {
    "marital_โสด ไม่มีแฟน": "โสด",
    "marital_มีแฟนแต่ยังไม่แต่งงาน": "มีแฟน",
    "marital_แต่งงานแล้ว": "แต่งงานแล้ว",
    "marital_หย่าร้าง/เป็นม่าย": "หย่าร้าง/เป็นม่าย",
}
def _render():
    try:
        data = load_all_data(UNSUPERVISED_OUTPUT_DIR)
    except Exception:
        return html.Div(
            dbc.Alert(
                "ไม่พบข้อมูลเซ็กเมนต์ กรุณารัน pipeline ก่อนใช้งาน",
                color="info",
                className="business-insight-empty-state",
            ),
            className="shell-content",
        )

    if data.segment_profiles.empty:
        return html.Div(
            dbc.Alert(
                "ไม่พบข้อมูลเซ็กเมนต์ กรุณารัน pipeline ก่อนใช้งาน",
                color="info",
                className="business-insight-empty-state",
            ),
            className="shell-content",
        )

    view = _build_business_insight_view(data)

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Business Insight | มุมมองธุรกิจ", className="page-title"),
                        html.H3("Executive Readout | สรุปภาพรวม", className="mb-2"),
                        html.P(view.headline, className="dashboard-muted-text mb-2"),
                        html.P(view.lede, className="dashboard-muted-text mb-0"),
                    ]
                ),
                className="ind-card mb-4",
            ),
            dbc.Row(
                [_build_kpi_card(label, value) for label, value in view.kpis],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _build_segment_cards(view.segment_snapshots),
                        xl=8,
                        width=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Recommended actions | แนวทางแนะนำ", className="page-title"),
                                    html.P("What to do next | แนวทางถัดไป", className="dashboard-muted-text"),
                                    html.Ul(
                                        [html.Li(item) for item in view.recommendations],
                                        className="business-insight-list",
                                    ),
                                ]
                            ),
                            className="ind-card",
                        ),
                        xl=4,
                        width=12,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Segment profile heatmap | ภาพรวมโปรไฟล์เซ็กเมนต์", className="page-title"),
                        html.P("Buy factors x packaging cues | ปัจจัยซื้อและสัญญาณบนบรรจุภัณฑ์", className="dashboard-muted-text"),
                        dcc.Graph(
                            figure=view.profile_heatmap_fig,
                            className="chart-card",
                            config={"displayModeBar": False},
                        ),
                    ]
                ),
                className="chart-card mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Span("Source: outputs/unsupervised | แหล่งข้อมูล", className="business-insight-footer__meta"),
                                dcc.Link("Open deep-dive view | เปิดหน้าวิเคราะห์เชิงลึก", href="/unsupervised", className="business-insight-footer__link"),
                            ],
                            className="business-insight-footer__row",
                        ),
                        html.P(view.footer_text, className="business-insight-footer__text"),
                    ]
                ),
                className="ind-card mt-4",
            ),
        ],
        className="shell-content",
    )


def _build_business_insight_view(data) -> SimpleNamespace:
    metrics = data.metrics
    segment_profiles = data.segment_profiles.copy()
    segment_sizes = _extract_segment_sizes(metrics, segment_profiles)
    total_responses = int(
        metrics.get("row_counts", {}).get("completed_top3") or sum(segment_sizes.values()) or segment_profiles["segment_size"].sum()
    )
    ordered_segments = sorted(segment_sizes.items(), key=lambda item: item[1], reverse=True)
    snapshots = []

    for segment_id, size in ordered_segments:
        row = _get_segment_row(segment_profiles, segment_id)
        top_vote = _top_labeled_value(row, "vote_", default_prefix="Option")
        top_buy_factor = _top_labeled_value(row, "buy_factor_", friendly_map=BUY_FACTOR_LABEL_MAP)
        top_packaging_importance = _top_labeled_value(
            row,
            "packaging_importance_",
            friendly_map=PACKAGING_IMPORTANCE_LABEL_MAP,
        )
        summary = _build_segment_summary(row, top_buy_factor, top_packaging_importance)
        snapshots.append(
            SimpleNamespace(
                segment_id=segment_id,
                size=size,
                share=size / total_responses if total_responses else 0,
                top_vote=top_vote,
                top_buy_factor=top_buy_factor,
                top_packaging_importance=top_packaging_importance,
                summary=summary,
            )
        )

    largest = snapshots[0]
    secondary = snapshots[1] if len(snapshots) > 1 else snapshots[0]
    dominant_buy_factor = largest.top_buy_factor.lower()
    dominant_packaging = largest.top_packaging_importance.lower()
    anomaly_rate = float(metrics.get("anomaly_detection", {}).get("anomaly_rate", 0) or 0)

    headline = f"Segment {largest.segment_id} is the core audience, representing {largest.share:.0%} of total respondents."
    lede = (
        f"The strongest buying signal is {dominant_buy_factor}, while {dominant_packaging} remains a meaningful cue. "
        f"Keep the story broad, direct, and ready for stakeholder review, with a light eye on the Thai market."
    )
    kpis = [
        ("Total responses", f"{total_responses:,}"),
        ("Largest segment", f"Segment {largest.segment_id} · {largest.share:.0%}"),
        ("Final clusters", str(int(metrics.get("final_cluster_k", len(snapshots))))),
        ("Anomaly rate", f"{anomaly_rate:.1%}"),
    ]
    recommendations = [
        f"Make Segment {largest.segment_id} the primary target because it covers {largest.share:.0%} of total respondents.",
        f"Keep {largest.top_buy_factor.lower()} as the lead message and hold {largest.top_packaging_importance.lower()} as a clear packaging cue for the Thai market.",
        f"Use Segment {secondary.segment_id} as the secondary lane and monitor the anomaly rate at {anomaly_rate:.1%} closely.",
    ]
    footer_text = "This snapshot is designed for a concise executive readout. For deeper detail, open the deep-dive page to review cluster behavior and model metrics."
    return SimpleNamespace(
        headline=headline,
        lede=lede,
        kpis=kpis,
        segment_snapshots=snapshots,
        profile_heatmap_fig=_build_profile_heatmap_figure(segment_profiles),
        recommendations=recommendations,
        footer_text=footer_text,
    )


def _build_profile_heatmap_figure(segment_profiles: pd.DataFrame):
    heatmap_columns = [
        column
        for column in segment_profiles.columns
        if column.startswith(("buy_factor_", "packaging_importance_"))
    ]
    heatmap_df = segment_profiles[heatmap_columns].T
    heatmap_df.index = [_friendly_heatmap_label(column) for column in heatmap_df.index]
    heatmap_df.columns = [f"Segment {segment}" for segment in heatmap_df.columns]
    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Segment profile heatmap",
        labels={"x": "Segment", "y": "Signal", "color": "Score"},
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def _build_segment_cards(snapshots: list[SimpleNamespace]) -> dbc.Row:
    cards = []
    for index, snapshot in enumerate(snapshots):
        color = SEGMENT_COLORS[index % len(SEGMENT_COLORS)]
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.H4(f"Segment {snapshot.segment_id} | เซ็กเมนต์ {snapshot.segment_id}", className="business-insight-segment__title"),
                                    html.Span(f"Share {snapshot.share:.0%} | สัดส่วน {snapshot.share:.0%}", className="business-insight-segment__tag"),
                                ],
                                className="business-insight-segment__head",
                            ),
                            html.Div(
                                [
                                    _build_segment_fact("Responses | จำนวนผู้ตอบ", f"{snapshot.size:,}"),
                                    _build_segment_fact("Top choice | ตัวเลือกที่เด่น", snapshot.top_vote),
                                    _build_segment_fact("Buying signal | ปัจจัยซื้อเด่น", snapshot.top_buy_factor),
                                    _build_segment_fact("Packaging cue | สัญญาณบรรจุภัณฑ์", snapshot.top_packaging_importance),
                                ],
                                className="business-insight-segment__facts",
                            ),
                            html.P(snapshot.summary, className="business-insight-segment__summary"),
                        ]
                    ),
                    className="ind-card",
                    style={"border-left": f"5px solid {color}"},
                ),
                xs=12,
                md=6,
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Segment snapshots | สรุปแต่ละเซ็กเมนต์", className="page-title"),
                html.P("Audience signals by group | สัญญาณของผู้ตอบแต่ละกลุ่ม", className="dashboard-muted-text"),
                dbc.Row(cards, className="g-3 business-insight-segment-grid"),
            ]
        ),
        className="ind-card",
    )


def _build_kpi_card(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(label, className="kpi-label"),
                    html.P(value, className="kpi-value"),
                ]
            ),
            className="kpi-card",
        ),
        xs=12,
        sm=6,
        lg=3,
    )


def _build_segment_fact(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="business-insight-segment__fact-label"),
            html.Div(value, className="business-insight-segment__fact-value"),
        ],
        className="business-insight-segment__fact",
    )


def _friendly_heatmap_label(column: str) -> str:
    if column in BUY_FACTOR_LABEL_MAP:
        return BUY_FACTOR_LABEL_MAP[column]
    if column in PACKAGING_IMPORTANCE_LABEL_MAP:
        return PACKAGING_IMPORTANCE_LABEL_MAP[column]
    return column.replace("_", " ")


def _extract_segment_sizes(metrics: dict, segment_profiles: pd.DataFrame) -> dict[int, int]:
    raw_sizes = metrics.get("segment_sizes") or {}
    if raw_sizes:
        sizes = {int(segment_id): int(size) for segment_id, size in raw_sizes.items()}
    else:
        sizes = {}
        for segment_id in segment_profiles.index:
            row = _get_segment_row(segment_profiles, int(segment_id))
            size_value = pd.to_numeric(row.get("segment_size", 0), errors="coerce")
            sizes[int(segment_id)] = int(size_value) if pd.notna(size_value) else 0
    return dict(sorted(sizes.items(), key=lambda item: item[0]))


def _get_segment_row(segment_profiles: pd.DataFrame, segment_id: int) -> pd.Series:
    for key in (segment_id, str(segment_id)):
        if key in segment_profiles.index:
            return segment_profiles.loc[key]
    return segment_profiles.iloc[0]


def _top_labeled_value(
    row: pd.Series,
    prefix: str,
    *,
    friendly_map: dict[str, str] | None = None,
    default_prefix: str | None = None,
) -> str:
    columns = [column for column in row.index if column.startswith(prefix)]
    if not columns:
        return "Unknown"
    values = pd.to_numeric(row[columns], errors="coerce").fillna(0)
    if values.empty:
        return "Unknown"
    top_column = values.idxmax()
    if friendly_map and top_column in friendly_map:
        return friendly_map[top_column]
    suffix = top_column.removeprefix(prefix).replace("_", " ").strip()
    if default_prefix:
        return f"{default_prefix} {suffix}".strip()
    return suffix or top_column


def _build_segment_summary(row: pd.Series, top_buy_factor: str, top_packaging_importance: str) -> str:
    gender = _top_labeled_value(row, "gender_", friendly_map=GENDER_LABEL_MAP)
    age = _top_labeled_value(row, "age_", friendly_map=AGE_LABEL_MAP)
    marital = _top_labeled_value(row, "marital_", friendly_map=MARITAL_LABEL_MAP)
    demographic_bits = [bit for bit in (gender, age, marital) if bit != "Unknown"]
    demographic_text = " · ".join(demographic_bits) if demographic_bits else "Balanced respondent mix"
    return (
        f"{demographic_text}. "
        f"This group prioritizes {top_buy_factor.lower()} and still sees {top_packaging_importance.lower()} as important."
    )


register_page(__name__, path="/business", title="มุมมองธุรกิจ", layout=_render)

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def render_summary_stats(metrics: dict) -> dbc.Row:
    raw = metrics.get("row_counts", {}).get("raw_loaded", 0)
    completed = metrics.get("row_counts", {}).get("completed_top3", 0)
    rate = (completed / raw * 100) if raw > 0 else 0

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Raw Responses", className="text-muted mb-1"),
                        html.H4(str(raw), className="mb-0"),
                    ]),
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Completed (Top-3)", className="text-muted mb-1"),
                        html.H4(str(completed), className="mb-0"),
                    ]),
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Completion Rate", className="text-muted mb-1"),
                        html.H4(f"{rate:.1f}%", className="mb-0 text-success"),
                    ]),
                ),
                width=3,
            ),
        ],
        className="mb-3",
    )


def segment_color_map() -> dict[int, str]:
    return {
        1: "#2E86AB",
        2: "#F18F01",
    }

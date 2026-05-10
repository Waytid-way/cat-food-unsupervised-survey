from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from catfood_unsupervised.dashboard.config import KPI_CARDS


def _get_nested(metrics: dict, path: list) -> Any:
    node = metrics
    for key in path:
        node = node[key]
    return node


def render_kpi_banner(metrics: dict) -> html.Div:
    cards = []
    for card_def in KPI_CARDS:
        value = _get_nested(metrics, card_def["metric_path"])
        formatted = format(value, card_def["format"])
        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4(
                            formatted,
                            className="card-title text-center",
                            style={"color": card_def["color"], "font-weight": "bold"},
                        ),
                        html.P(card_def["label"], className="card-text text-center text-muted"),
                    ]
                ),
                className="text-center shadow-sm",
                style={"background": "#FFFFFF", "border-radius": "12px"},
            ),
            width=2,
        )
        cards.append(card)
    return html.Div(
        dbc.Row(cards, className="mb-4"),
        style={"background": "#F8F9FA", "padding": "20px", "border-radius": "8px"},
    )
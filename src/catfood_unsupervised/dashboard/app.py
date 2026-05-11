from __future__ import annotations

import os
from pathlib import Path

import dash
from dash import html
from flask import send_from_directory

from catfood_unsupervised.dashboard.components.kpi_banner import render_kpi_banner
from catfood_unsupervised.dashboard.components.shell import render_dashboard_shell
from catfood_unsupervised.dashboard.components.tab_business_insight import (
    render_business_insight_tab,
)
from catfood_unsupervised.dashboard.components.tab_clustering import render_clustering_tab
from catfood_unsupervised.dashboard.components.tab_correlation import render_correlation_tab
from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab
from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab
from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab
from catfood_unsupervised.dashboard.config import SUPERVISED_OUTPUT_DIR, TAB_ITEMS
from catfood_unsupervised.dashboard.data_loader import load_all_data
from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.supervised_callbacks import register_supervised_callbacks
from catfood_unsupervised.dashboard.supervised_data_loader import (
    load_supervised_dashboard_bundle,
)

OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))
INITIAL_TAB = TAB_ITEMS[0]["value"] if TAB_ITEMS else "tab_eda"

dash_app = dash.Dash(
    __name__,
    title="Cat Food ML Dashboard",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "/assets/custom.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


@dash_app.server.route("/assets/<path:path>")
def serve_static(path):
    styles_dir = Path(__file__).parent / "styles"
    return send_from_directory(styles_dir, path)


try:
    dashboard_data = load_all_data(OUTPUT_DIR)
    KPI_METRICS = dashboard_data.metrics
except Exception:
    dashboard_data = None
    KPI_METRICS = {}

try:
    supervised_dashboard_data = load_supervised_dashboard_bundle(SUPERVISED_OUTPUT_DIR)
except Exception:
    supervised_dashboard_data = None


@dash_app.callback(
    dash.Output("tab_content", "children"),
    dash.Input("unsupervised-tabs", "value"),
)
def render_tab_content(selected_tab: str):
    if selected_tab == "tab_eda":
        if dashboard_data is None:
            return html.Div("Data not available")
        return render_eda_tab(dashboard_data)
    elif selected_tab == "tab_correlation":
        if dashboard_data is None:
            return html.Div("Data not available")
        return render_correlation_tab(dashboard_data)
    elif selected_tab == "tab_clustering":
        if dashboard_data is None:
            return html.Div("Data not available")
        return render_clustering_tab(dashboard_data)
    elif selected_tab == "tab_persona":
        if dashboard_data is None:
            return html.Div("Data not available")
        return render_persona_tab(dashboard_data)
    elif selected_tab == "tab_supervised":
        if supervised_dashboard_data is None:
            return html.Div("Supervised data not available")
        return render_supervised_tab(supervised_dashboard_data)
    elif selected_tab == "tab_business_insight":
        if supervised_dashboard_data is None:
            return html.Div("Supervised data not available")
        return render_business_insight_tab(supervised_dashboard_data)
    return html.Div("Select a tab")


dash_app.layout = render_dashboard_shell(
    active_tab=INITIAL_TAB,
    content=html.Div(
        [
            render_kpi_banner(KPI_METRICS),
            html.Div(id="tab_content", children=render_tab_content(INITIAL_TAB)),
        ],
        className="shell-tab-body",
    ),
)


if supervised_dashboard_data is not None:
    register_supervised_callbacks(dash_app, supervised_dashboard_data)

from __future__ import annotations

import os
from pathlib import Path

import dash
from dash import html, page_container
import dash_bootstrap_components as dbc

from catfood_unsupervised.dashboard.supervised_callbacks import register_supervised_callbacks
from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_runtime_bundle

OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))
UNSUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "unsupervised"
SUPERVISED_OUTPUT_DIR = OUTPUT_DIR / "supervised"

dash_app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "/assets/style.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

dash_app.title = "CatFood ML Dashboard"

dashboard_pages = [
    {"label": "Home", "icon": "ph ph-house", "href": "/"},
    {"label": "Unsupervised", "icon": "ph ph-chart-scatter", "href": "/unsupervised"},
    {"label": "Supervised", "icon": "ph ph-robot", "href": "/supervised"},
    {"label": "Business Insight", "icon": "ph ph-briefcase", "href": "/business"},
]

sidebar_nav = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink(
                [html.I(className=item["icon"], **{"aria-hidden": "true"}), html.Span(item["label"], className="shell-nav__text")],
                href=item["href"],
                active="exact",
                className="nav-item",
            )
        )
        for item in dashboard_pages
    ],
    vertical=True,
    pills=True,
    className="sidebar-nav",
)


def render_shell():
    return html.Div(
        [
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("AI", className="shell-logo"),
                            html.Div(
                                [html.H1("CatFood ML", className="shell-brand"), html.P("Survey intelligence workspace", className="shell-brand__lede")],
                                className="shell-brand-col",
                            ),
                        ],
                        className="shell-brand-row",
                    ),
                    html.Div("Navigation", className="shell-nav-heading"),
                    sidebar_nav,
                ],
                className="shell-sidebar",
            ),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div([html.P("Cat Food Packaging Survey Analysis (n=148)", className="shell-eyebrow"), html.H2("Welcome back, Team!", className="shell-title")], className="shell-heading"),
                            html.Div(
                                [
                                    html.Button(html.I(className="ph ph-calendar-blank", **{"aria-hidden": "true"}), className="shell-icon-button", type="button", n_clicks=0),
                                ],
                                className="shell-actions",
                            ),
                        ],
                        className="shell-header",
                    ),
                    html.Div(page_container, className="shell-content"),
                ],
                className="shell-main",
            ),
        ],
        className="dash-shell shell-root",
    )


dash_app.layout = render_shell()

_supervised_bundle = load_supervised_runtime_bundle(SUPERVISED_OUTPUT_DIR)
register_supervised_callbacks(dash_app, _supervised_bundle)

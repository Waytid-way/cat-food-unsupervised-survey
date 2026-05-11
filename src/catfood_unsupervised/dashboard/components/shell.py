from __future__ import annotations

from typing import Any

from dash import dcc, html

from catfood_unsupervised.dashboard.config import TAB_ITEMS


def _render_tab_label(item: dict[str, str]) -> html.Span:
    icon_class = item.get("icon", "ph ph-circle")
    return html.Span(
        [
            html.I(className=icon_class, **{"aria-hidden": "true"}),
            html.Span(item["label"], className="shell-tab__text"),
        ],
        className="shell-tab__label",
    )


def render_dashboard_shell(active_tab: str, content: Any) -> html.Div:
    return html.Div(
        [
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("AI", className="shell-logo"),
                            html.Div(
                                [
                                    html.H1("CatFood ML", className="shell-brand"),
                                    html.P("Survey intelligence workspace", className="shell-brand__lede"),
                                ]
                            ),
                        ],
                        className="shell-brand-row",
                    ),
                    html.Div("Home Dashboard", className="shell-nav-heading"),
                    dcc.Tabs(
                        id="unsupervised-tabs",
                        value=active_tab,
                        className="shell-tabs",
                        parent_className="shell-tabs__parent",
                        content_style={"display": "none"},
                        children=[
                            dcc.Tab(
                                label=_render_tab_label(item),
                                value=item["value"],
                                className="shell-tab",
                                selected_className="shell-tab--selected",
                            )
                            for item in TAB_ITEMS
                        ],
                    ),
                    html.Div(
                        html.Button(
                            [
                                html.I(className="ph ph-sign-out", **{"aria-hidden": "true"}),
                                html.Span("Log out"),
                            ],
                            className="shell-logout",
                            type="button",
                        ),
                        className="shell-footer",
                    ),
                ],
                className="shell-sidebar",
            ),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.P("Cat Food Packaging Survey Analysis (n=148)", className="shell-eyebrow"),
                                    html.H2("Welcome back, Team!", className="shell-title"),
                                ],
                                className="shell-heading",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        html.I(className="ph ph-calendar-blank", **{"aria-hidden": "true"}),
                                        className="shell-icon-button",
                                        type="button",
                                    ),
                                    html.Button(
                                        [
                                            html.I(className="ph ph-plus", **{"aria-hidden": "true"}),
                                            html.Span("Add new widget"),
                                        ],
                                        className="shell-primary-button",
                                        type="button",
                                    ),
                                ],
                                className="shell-actions",
                            ),
                        ],
                        className="shell-header",
                    ),
                    html.Div(content, className="shell-content"),
                ],
                className="shell-main",
            ),
        ],
        className="shell-root",
    )

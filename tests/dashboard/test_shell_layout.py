from __future__ import annotations

from dash import html

from catfood_unsupervised.dashboard.config import TAB_ITEMS


def test_dashboard_shell_contains_sidebar_header_and_content_slot():
    from catfood_unsupervised.dashboard.components.shell import render_dashboard_shell

    shell = render_dashboard_shell(active_tab="tab_eda", content=html.Div("body"))
    rendered = " ".join(_collect_text(shell))

    assert "CatFood ML" in rendered
    assert "Home Dashboard" in rendered
    assert "Business Insight" in rendered
    assert "body" in rendered


def test_tab_items_provide_shell_navigation_metadata():
    assert [item["value"] for item in TAB_ITEMS] == [
        "tab_eda",
        "tab_correlation",
        "tab_clustering",
        "tab_persona",
        "tab_supervised",
        "tab_business_insight",
    ]
    assert all("label" in item for item in TAB_ITEMS)
    assert all("icon" in item for item in TAB_ITEMS)
    assert all("section" in item for item in TAB_ITEMS)


def test_dashboard_app_uses_shell_layout():
    from catfood_unsupervised.dashboard.app import dash_app

    rendered = " ".join(_collect_text(dash_app.layout))

    assert "CatFood ML" in rendered
    assert "Welcome back, Team!" in rendered


def _collect_text(component) -> list[str]:
    texts: list[str] = []
    if isinstance(component, str):
        return [component]

    label = getattr(component, "label", None)
    if label is not None:
        texts.extend(_collect_text(label))

    children = getattr(component, "children", None)
    if children is None:
        return texts
    if isinstance(children, (list, tuple)):
        for child in children:
            texts.extend(_collect_text(child))
    else:
        texts.extend(_collect_text(children))
    return texts

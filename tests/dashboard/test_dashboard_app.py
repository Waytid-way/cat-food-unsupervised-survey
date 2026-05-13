from __future__ import annotations

from dash import html


def test_dashboard_shell_contains_sidebar_header_and_content():
    from catfood_unsupervised.dashboard.app import dash_app

    shell = dash_app.layout
    rendered = " ".join(_collect_text(shell))

    assert "CatFood ML" in rendered
    assert "Navigation" in rendered
    assert "Home" in rendered
    assert "Unsupervised" in rendered
    assert "Supervised" in rendered
    assert "Business Insight" in rendered
    assert "Welcome back, Team!" in rendered
    assert "Add new widget" not in rendered
    assert "Log out" not in rendered


def test_dashboard_app_uses_multi_page():
    from catfood_unsupervised.dashboard.app import dash_app

    assert dash_app.use_pages is True
    assert dash_app.title == "CatFood ML Dashboard"


def test_dashboard_shell_navigation_links():
    from catfood_unsupervised.dashboard.app import dash_app

    nav_links = _find_navlinks(dash_app.layout)
    hrefs = [link.href for link in nav_links if hasattr(link, "href") and link.href]

    assert "/" in hrefs, f"Home link not found. Found hrefs: {hrefs}"
    assert "/unsupervised" in hrefs, f"Unsupervised link not found. Found hrefs: {hrefs}"
    assert "/supervised" in hrefs, f"Supervised link not found. Found hrefs: {hrefs}"
    assert "/business" in hrefs, f"Business Insight link not found. Found hrefs: {hrefs}"


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


def _find_navlinks(component, results=None):
    if results is None:
        results = []
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _find_navlinks(child, results)
        elif children:
            _find_navlinks(children, results)
    if hasattr(component, "__class__"):
        cls_name = component.__class__.__name__
        if "NavLink" in cls_name:
            results.append(component)
    return results

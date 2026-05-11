from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_all_page_modules_import_without_error():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    assert dash_app is not None
    assert dash_app.title == "CatFood ML Dashboard"


def test_shell_layout_has_sidebar_navigation():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    shell = dash_app.layout

    nav_links = _find_navlinks(shell)
    hrefs = [link.href for link in nav_links if hasattr(link, "href") and link.href]

    assert "/" in hrefs
    assert "/unsupervised" in hrefs
    assert "/supervised" in hrefs
    assert "/business" in hrefs


def test_shell_header_renders():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    shell = dash_app.layout

    texts = _collect_text(shell)
    rendered = " ".join(texts)

    assert "CatFood ML" in rendered
    assert "Welcome back, Team!" in rendered
    assert "Cat Food Packaging Survey Analysis" in rendered


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
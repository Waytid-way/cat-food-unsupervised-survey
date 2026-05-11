from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_supervised_page_module_imports():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    assert dash_app is not None
    assert dash_app.title == "CatFood ML Dashboard"


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
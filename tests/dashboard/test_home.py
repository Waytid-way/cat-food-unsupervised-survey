from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_all_page_modules_import_without_error():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    assert dash_app is not None
    assert dash_app.title == "CatFood ML Dashboard"


def test_home_page_uses_unsupervised_output_dir(monkeypatch):
    from catfood_unsupervised.dashboard.pages import home as home_page
    import catfood_unsupervised.dashboard.data_loader as dashboard_data_loader

    captured: dict[str, Path] = {}

    def fake_load_all_data(output_dir: Path):
        captured["output_dir"] = Path(output_dir)
        return SimpleNamespace(
            metrics={
                "kmeans_evaluation": [
                    {
                        "silhouette_score": 0.321,
                        "davies_bouldin_score": 1.234,
                        "inertia": 12.3,
                    }
                ],
                "pca": {"total_explained_variance_ratio": 0.81},
                "anomaly_detection": {"anomaly_rate": 0.1},
            }
        )

    monkeypatch.setattr(dashboard_data_loader, "load_all_data", fake_load_all_data)

    rendered = home_page._render()

    assert captured["output_dir"] == Path("outputs") / "unsupervised"
    assert rendered is not None


def test_home_page_renders_respondent_snapshot_graphs(monkeypatch):
    from catfood_unsupervised.dashboard.pages import home as home_page
    import catfood_unsupervised.dashboard.data_loader as dashboard_data_loader

    clean_df = pd.DataFrame(
        {
            "เพศของคุณ": ["ชาย", "หญิง", "หญิง", "ชาย"],
            "อายุของคุณ": ["20-29ปี", "20-29ปี", "30-39ปี", "30-39ปี"],
            "สถานภาพสมรส": ["โสด ไม่มีแฟน", "โสด ไม่มีแฟน", "แต่งงานแล้ว", "แต่งงานแล้ว"],
        }
    )

    def fake_load_all_data(output_dir: Path):
        return SimpleNamespace(metrics={}, clean_df=clean_df)

    monkeypatch.setattr(dashboard_data_loader, "load_all_data", fake_load_all_data)

    rendered = home_page._render()
    graphs = _find_graphs(rendered)
    title_to_graph = {getattr(graph.figure.layout.title, "text", ""): graph for graph in graphs}
    rendered_text = " ".join(_collect_text(rendered))

    assert "Respondent Snapshot" in rendered_text
    assert title_to_graph["Gender Distribution"].figure.data[0].type == "pie"
    assert title_to_graph["Age Distribution"].figure.data[0].type == "bar"
    assert title_to_graph["Marital Status Distribution"].figure.data[0].type == "pie"


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


def _find_graphs(component, results=None):
    if results is None:
        results = []
    if hasattr(component, "children"):
        children = component.children
        if isinstance(children, (list, tuple)):
            for child in children:
                _find_graphs(child, results)
        elif children is not None:
            _find_graphs(children, results)
    if component.__class__.__name__ == "Graph":
        results.append(component)
    return results

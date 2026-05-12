from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_insight_page_module_imports():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    assert dash_app is not None
    assert dash_app.title == "CatFood ML Dashboard"


def test_insight_page_renders_compact_boardroom_snapshot(monkeypatch):
    from catfood_unsupervised.dashboard.pages import insight as insight_page

    clean_df = pd.DataFrame(
        {
            "segment": [1, 1, 2, 2],
            "เพศของคุณ": ["ชาย", "หญิง", "หญิง", "หญิง"],
            "อายุของคุณ": ["20-29ปี", "20-29ปี", "30-39ปี", "30-39ปี"],
            "สถานภาพสมรส": ["โสด ไม่มีแฟน", "โสด ไม่มีแฟน", "แต่งงานแล้ว", "แต่งงานแล้ว"],
        }
    )
    segment_profiles = pd.DataFrame(
        {
            "segment_size": [36, 112],
            "vote_01": [1.0, 0.7],
            "vote_02": [1.1, 0.8],
            "vote_03": [2.0, 1.6],
            "buy_factor_01": [4.3, 4.2],
            "buy_factor_02": [3.6, 3.1],
            "buy_factor_03": [4.6, 4.4],
            "buy_factor_04": [3.5, 3.0],
            "buy_factor_05": [4.5, 3.9],
            "packaging_importance_01": [4.1, 3.7],
            "packaging_importance_02": [4.3, 3.5],
            "packaging_importance_03": [4.4, 4.1],
            "packaging_importance_04": [4.2, 4.0],
        },
        index=[1, 2],
    )
    metrics = {
        "row_counts": {"completed_top3": 148},
        "final_cluster_k": 2,
        "anomaly_detection": {"anomaly_rate": 0.1756},
        "segment_sizes": {"1": 36, "2": 112},
    }

    monkeypatch.setattr(
        insight_page,
        "load_all_data",
        lambda output_dir: SimpleNamespace(
            metrics=metrics,
            clean_df=clean_df,
            correlation_matrix=pd.DataFrame([[1.0]], columns=["x"], index=["x"]),
            segment_profiles=segment_profiles,
            pca_scores=pd.DataFrame({"PC1": [1.0], "PC2": [0.5]}),
        ),
    )

    rendered = insight_page._render()
    texts = " ".join(_collect_text(rendered))
    graphs = _collect_graphs(rendered)

    assert "Business Insight" in texts
    assert "Executive Readout" in texts
    assert "Segment snapshots" in texts
    assert "Audience signals by group" in texts
    assert "Segment profile heatmap" in texts
    assert "Buy factors x packaging cues" in texts
    assert "Segment 2 is the core audience" in texts
    assert "primary target" in texts
    assert "secondary lane" in texts
    assert "Recommended actions" in texts
    assert "What to do next" in texts
    assert "Persona Cards" not in texts
    assert "Segment mix" not in texts
    assert len(graphs) == 1
    assert not any(getattr(graph.figure.data[0], "type", "") == "pie" for graph in graphs)
    assert any(getattr(graph.figure.data[0], "type", "") == "heatmap" for graph in graphs)


def test_insight_page_shows_empty_state_when_outputs_missing(monkeypatch):
    from catfood_unsupervised.dashboard.pages import insight as insight_page

    monkeypatch.setattr(
        insight_page,
        "load_all_data",
        lambda output_dir: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    rendered = insight_page._render()
    texts = " ".join(_collect_text(rendered))

    assert "ไม่พบข้อมูลเซ็กเมนต์" in texts


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


def _collect_graphs(component, results=None):
    if results is None:
        results = []
    if component.__class__.__name__ == "Graph":
        results.append(component)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_graphs(child, results)
    elif children is not None:
        _collect_graphs(children, results)
    return results

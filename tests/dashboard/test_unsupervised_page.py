from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_unsupervised_page_module_imports():
    import catfood_unsupervised.dashboard.app as app_module

    dash_app = app_module.dash_app
    assert dash_app is not None
    assert dash_app.title == "CatFood ML Dashboard"


def test_unsupervised_page_uses_unsupervised_output_dir_and_segment_count(monkeypatch):
    from catfood_unsupervised.dashboard.pages import unsupervised as unsupervised_page

    captured: dict[str, Path] = {}

    def fake_load_all_data(output_dir: Path):
        captured["output_dir"] = Path(output_dir)
        clean_df = pd.DataFrame(
            {
                "segment": [1, 2, 3],
                "option_01_ips": [0.1, 0.2, 0.3],
                "option_02_ips": [0.4, 0.5, 0.6],
            }
        )
        segment_profiles = pd.DataFrame(
            {
                "segment_size": [2, 1],
                "vote_01": [1.0, 2.0],
                "vote_02": [2.0, 1.0],
                "vote_03": [1.5, 0.5],
                "vote_04": [0.5, 1.5],
                "vote_05": [1.0, 1.0],
                "vote_06": [0.0, 0.5],
                "vote_07": [0.0, 0.0],
                "vote_08": [0.0, 0.0],
                "vote_09": [0.0, 0.0],
                "vote_10": [0.0, 0.0],
                "buy_factor_01": [1.0, 2.0],
                "buy_factor_02": [2.0, 1.0],
                "buy_factor_03": [1.5, 0.5],
                "buy_factor_04": [0.5, 1.5],
                "buy_factor_05": [1.0, 1.0],
                "packaging_effect_01": [1.0, 2.0],
                "packaging_importance_01": [1.0, 2.0],
                "packaging_importance_02": [1.5, 0.5],
                "packaging_importance_03": [0.5, 1.5],
                "packaging_importance_04": [1.0, 1.0],
                "packaging_importance_05": [0.0, 0.5],
                "packaging_importance_06": [0.0, 0.0],
                "packaging_importance_07": [0.0, 0.0],
                "packaging_importance_08": [0.0, 0.0],
            },
            index=[1, 2],
        )
        return SimpleNamespace(
            metrics={
                "pca": {
                    "explained_variance_ratio": [0.5, 0.3],
                    "cumulative_explained_variance": [0.5, 0.8],
                },
                "kmeans_evaluation": [
                    {"k": 2, "silhouette_score": 0.21, "davies_bouldin_score": 1.2, "inertia": 11.0},
                    {"k": 3, "silhouette_score": 0.31, "davies_bouldin_score": 0.9, "inertia": 9.0},
                ],
                "final_cluster_k": 3,
                "anomaly_detection": {"anomaly_rate": 0.1},
            },
            clean_df=clean_df,
            pca_scores=pd.DataFrame(
                {"PC1": [1.0, 2.0, 3.0], "PC2": [0.5, 0.6, 0.7]}
            ),
            correlation_matrix=pd.DataFrame([[1.0]], columns=["option_01_ips"], index=["option_01_ips"]),
            segment_profiles=segment_profiles,
        )

    monkeypatch.setattr(unsupervised_page, "load_all_data", fake_load_all_data)

    rendered = unsupervised_page.render()
    graphs = _find_graphs(rendered)
    titles = [getattr(graph.figure.layout.title, "text", "") for graph in graphs]
    heatmap_graph = next(graph for graph in graphs if getattr(graph.figure.layout.title, "text", "") == "Segment Profile Heatmap")
    heatmap_labels = [str(label) for label in heatmap_graph.figure.data[0].y]

    assert captured["output_dir"] == Path("outputs") / "unsupervised"
    assert any("k=3" in title for title in titles)
    assert "Segment Size Distribution" in titles
    assert "Segment Profile Heatmap" in titles
    assert "Hierarchical Clustering Dendrogram (Ward)" not in titles
    assert all(
        not label.startswith(("vote_", "buy_factor_", "packaging_effect_", "packaging_importance_"))
        for label in heatmap_labels
    )


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

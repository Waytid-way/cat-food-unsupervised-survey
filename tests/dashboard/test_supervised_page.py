from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_supervised_page_renders_predict_form_and_upload(monkeypatch):
    import catfood_unsupervised.dashboard.app  # noqa: F401
    from catfood_unsupervised.dashboard.pages import supervised as supervised_page
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2"]
        for column in FEATURE_COLUMNS
    }
    bundle = SimpleNamespace(
        metrics={
            "best_model_name": "random_forest",
            "best_model_accuracy": 0.88,
            "best_model_macro_f1": 0.79,
            "best_model_weighted_f1": 0.86,
            "classification_report": {
                "weighted avg": {"precision": 0.9, "recall": 0.88, "f1-score": 0.86}
            },
        },
        comparison=pd.DataFrame(
            [
                {
                    "model_name": "random_forest",
                    "accuracy": 0.88,
                    "macro_f1": 0.79,
                    "weighted_f1": 0.86,
                }
            ]
        ),
        confusion_matrix=pd.DataFrame([[1, 0], [0, 1]], index=["actual_1", "actual_2"], columns=["pred_1", "pred_2"]),
        feature_importance=pd.DataFrame(
            [{"feature": "f1", "importance_mean": 0.4, "importance_std": 0.05}]
        ),
        predictions=pd.DataFrame(),
        feature_options=feature_options,
        model_path=Path("outputs/supervised/best_model.pkl"),
        history_db_path=Path("outputs/supervised/prediction_history.sqlite3"),
    )

    monkeypatch.setattr(supervised_page, "load_supervised_runtime_bundle", lambda _path: bundle)

    rendered = supervised_page._render()
    ids = _collect_ids(rendered)
    texts = " ".join(_collect_text(rendered))

    assert "supervised-model-upload" in ids
    assert "supervised-predict-button" in ids
    assert "supervised-result-panel" in ids
    assert "supervised-history-panel" in ids
    assert "Upload" in texts
    assert "Predict" in texts


def test_supervised_feature_importance_graph_uses_short_labels():
    from catfood_unsupervised.dashboard.components.supervised_form import _SHORT_LABELS
    from catfood_unsupervised.dashboard.components.tab_supervised import _render_importance_card
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    card = _render_importance_card(
        pd.DataFrame(
            [
                {"feature": FEATURE_COLUMNS[0], "importance_mean": 0.4, "importance_std": 0.05},
                {"feature": FEATURE_COLUMNS[16], "importance_mean": 0.2, "importance_std": 0.02},
            ]
        )
    )

    graph = _find_graph(card)
    y_values = list(graph.figure.data[0].y)

    assert _SHORT_LABELS[FEATURE_COLUMNS[0]] in y_values
    assert _SHORT_LABELS[FEATURE_COLUMNS[16]] in y_values
    assert FEATURE_COLUMNS[0] not in y_values
    assert graph.figure.layout.yaxis.title.text == "Question"


def test_supervised_feature_importance_graph_filters_non_positive_rows():
    from catfood_unsupervised.dashboard.components.supervised_form import _SHORT_LABELS
    from catfood_unsupervised.dashboard.components.tab_supervised import _render_importance_card
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    card = _render_importance_card(
        pd.DataFrame(
            [
                {"feature": FEATURE_COLUMNS[0], "importance_mean": 0.4, "importance_std": 0.05},
                {"feature": FEATURE_COLUMNS[1], "importance_mean": 0.0, "importance_std": 0.01},
                {"feature": FEATURE_COLUMNS[2], "importance_mean": -0.1, "importance_std": 0.02},
            ]
        )
    )

    graph = _find_graph(card)
    y_values = list(graph.figure.data[0].y)
    x_values = list(graph.figure.data[0].x)

    assert y_values == [_SHORT_LABELS[FEATURE_COLUMNS[0]]]
    assert x_values == [0.4]


def test_supervised_feature_importance_card_falls_back_when_no_positive_rows():
    from catfood_unsupervised.dashboard.components.tab_supervised import _render_importance_card
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    card = _render_importance_card(
        pd.DataFrame(
            [
                {"feature": FEATURE_COLUMNS[0], "importance_mean": 0.0, "importance_std": 0.05},
                {"feature": FEATURE_COLUMNS[1], "importance_mean": -0.1, "importance_std": 0.01},
            ]
        )
    )

    texts = " ".join(_collect_text(card))
    assert "No positive feature importance output yet." in texts
    assert _find_graph(card) is None


def _find_graph(component):
    graph = getattr(component, "children", None)
    if hasattr(component, "figure"):
        return component
    if isinstance(graph, (list, tuple)):
        for child in graph:
            found = _find_graph(child)
            if found is not None:
                return found
    elif graph is not None:
        found = _find_graph(graph)
        if found is not None:
            return found
    return None


def _collect_ids(component) -> list[str]:
    ids: list[str] = []
    if hasattr(component, "id") and component.id:
        ids.append(str(component.id))

    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.extend(_collect_ids(child))
    elif children is not None:
        ids.extend(_collect_ids(children))
    return ids


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

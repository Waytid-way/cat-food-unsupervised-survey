from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_build_prediction_summary_formats_friendly_class_and_confidence():
    from catfood_unsupervised.dashboard.components.supervised_results import (
        build_prediction_summary,
    )

    prediction_frame = pd.DataFrame(
        [
            {
                "predicted_segment": 2,
                "prob_class_1": 0.128,
                "prob_class_2": 0.872,
            }
        ]
    )

    summary = build_prediction_summary(
        prediction_frame.iloc[0],
        {"class_labels": [1, 2]},
    )

    assert summary.display_label == "Segment 2 - Main-market audience"
    assert summary.confidence == pytest.approx(0.872)
    assert summary.raw_class == "2"
    assert summary.top_probabilities[0][0] == "Segment 2 - Main-market audience"
    assert summary.top_probabilities[0][1] == pytest.approx(0.872)
    assert summary.top_probabilities[1][0] == "Segment 1 - Premium audience"
    assert summary.top_probabilities[1][1] == pytest.approx(0.128)


def test_render_prediction_result_panel_shows_readable_summary_and_raw_class():
    from catfood_unsupervised.dashboard.components.supervised_results import (
        render_prediction_result_panel,
    )

    prediction_frame = pd.DataFrame(
        [
            {
                "predicted_segment": 2,
                "prob_class_1": 0.128,
                "prob_class_2": 0.872,
            }
        ]
    )

    panel = render_prediction_result_panel(
        prediction_frame,
        {"best_model_name": "random_forest", "class_labels": [1, 2]},
    )

    texts = " ".join(_collect_text(panel))

    assert "Segment 2 - Main-market audience" in texts
    assert "Confidence" in texts
    assert "87.2%" in texts
    assert "Top probabilities" in texts
    assert "Segment 1 - Premium audience: 12.8%" in texts
    assert "Raw class: 2" in texts


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

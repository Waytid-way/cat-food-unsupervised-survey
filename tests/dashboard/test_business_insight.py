from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.dashboard.components.tab_business_insight import (
    render_business_insight_tab,
)
from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_dashboard_bundle
from catfood_unsupervised.supervised.history_store import (
    append_prediction_history,
    fetch_recent_prediction_history,
    initialize_prediction_history_db,
)
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


def test_business_insight_reads_prediction_history_and_summarizes_usage(
    supervised_fixture_path: Path,
    tmp_path: Path,
):
    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"
    run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )
    bundle = load_supervised_dashboard_bundle(output_dir)

    history_db = tmp_path / "history.sqlite3"
    initialize_prediction_history_db(history_db)
    append_prediction_history(
        history_db,
        source="dashboard",
        model_name="random_forest",
        predicted_segment=1,
        probability={"prob_class_1": 0.82, "prob_class_2": 0.18},
        raw_input={"age": "20-29", "gender": "หญิง"},
    )
    append_prediction_history(
        history_db,
        source="dashboard",
        model_name="random_forest",
        predicted_segment=2,
        probability={"prob_class_1": 0.21, "prob_class_2": 0.79},
        raw_input={"age": "30-39", "gender": "ชาย"},
    )

    history = fetch_recent_prediction_history(history_db, limit=10)
    tab = render_business_insight_tab(bundle, history)
    rendered_text = " ".join(_collect_text(tab))

    assert "Prediction history" in rendered_text
    assert "Count by segment" in rendered_text
    assert "Total predictions" in rendered_text
    assert "Segment 1" in rendered_text
    assert "Segment 2" in rendered_text
    assert "random_forest" in rendered_text


def _collect_text(component) -> list[str]:
    texts: list[str] = []
    if isinstance(component, str):
        return [component]

    children = getattr(component, "children", None)
    if children is None:
        return texts
    if isinstance(children, (list, tuple)):
        for child in children:
            texts.extend(_collect_text(child))
    else:
        texts.extend(_collect_text(children))
    return texts

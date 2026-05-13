from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
from dash import dcc
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_supervised_form_groups_questions_and_keeps_all_fields_as_dropdowns():
    from catfood_unsupervised.dashboard.components.supervised_form import render_supervised_form
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2", f"{column}-opt-3"]
        for column in FEATURE_COLUMNS
    }

    form = render_supervised_form(
        feature_options,
        model_path_text="outputs/supervised/best_model.pkl",
        history_path_text="outputs/supervised/prediction_history.sqlite3",
    )

    sliders = _collect_components(form, dcc.Slider)
    dropdowns = _collect_components(form, dcc.Dropdown)
    texts = " ".join(_collect_text(form))

    assert len(sliders) == 0
    assert len(dropdowns) == 18
    assert "คุณสมบัติอาหารและรสชาติ" in texts
    assert "บรรจุภัณฑ์และการสื่อสาร" in texts
    assert "ข้อมูลผู้ตอบ" in texts


def test_supervised_field_specs_sort_answers_in_logical_order():
    from catfood_unsupervised.dashboard.components.supervised_form import build_supervised_field_specs
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2", f"{column}-opt-3"]
        for column in FEATURE_COLUMNS
    }
    feature_options[FEATURE_COLUMNS[0]] = ["มากที่สุด", "ปานกลาง", "มาก", "น้อย"]
    feature_options[FEATURE_COLUMNS[5]] = ["มีผล", "ไม่มีผล"]
    feature_options[FEATURE_COLUMNS[6]] = [
        "ภาพการ์ตูน หรือลายเส้น",
        "ได้ทั้งสองแบบ หากถูกใจ",
        "ภาพแมวจริง หรือแมวสมจริง (AI)",
    ]
    feature_options[FEATURE_COLUMNS[15]] = ["50ปี ขึ้นไป", "20-29ปี", "40-49ปี", "30-39ปี"]
    feature_options[FEATURE_COLUMNS[16]] = ["อื่นๆ", "หญิง", "ชาย"]
    feature_options[FEATURE_COLUMNS[17]] = [
        "หย่าร้าง/เป็นม่าย",
        "แต่งงานแล้ว",
        "โสด ไม่มีแฟน",
        "มีแฟนแต่ยังไม่แต่งงาน",
    ]

    specs = build_supervised_field_specs(feature_options)
    spec_by_column = {spec.column: spec for spec in specs}

    assert spec_by_column[FEATURE_COLUMNS[0]].options == ("น้อย", "ปานกลาง", "มาก", "มากที่สุด")
    assert spec_by_column[FEATURE_COLUMNS[5]].options == ("ไม่มีผล", "มีผล")
    assert spec_by_column[FEATURE_COLUMNS[6]].options == (
        "ภาพการ์ตูน หรือลายเส้น",
        "ได้ทั้งสองแบบ หากถูกใจ",
        "ภาพแมวจริง หรือแมวสมจริง (AI)",
    )
    assert spec_by_column[FEATURE_COLUMNS[15]].options == ("20-29ปี", "30-39ปี", "40-49ปี", "50ปี ขึ้นไป")
    assert spec_by_column[FEATURE_COLUMNS[16]].options == ("ชาย", "หญิง", "อื่นๆ")
    assert spec_by_column[FEATURE_COLUMNS[17]].options == (
        "โสด ไม่มีแฟน",
        "มีแฟนแต่ยังไม่แต่งงาน",
        "แต่งงานแล้ว",
        "หย่าร้าง/เป็นม่าย",
    )


def test_supervised_form_does_not_show_missing_neutral_choice_note():
    from catfood_unsupervised.dashboard.components.supervised_form import render_supervised_form
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2", f"{column}-opt-3"]
        for column in FEATURE_COLUMNS
    }

    form = render_supervised_form(
        feature_options,
        model_path_text="outputs/supervised/best_model.pkl",
        history_path_text="outputs/supervised/prediction_history.sqlite3",
    )

    texts = " ".join(_collect_text(form))
    assert "เน€เธเธขเน" not in texts
    assert "เนเธกเนเนเธชเธ”เธเน€เธเธดเนเธกเนเธซเนเน€เธญเธ" not in texts


def test_supervised_scoring_maps_dropdown_and_slider_inputs_back_to_original_values(supervised_bundle):
    from catfood_unsupervised.dashboard.components.supervised_form import build_supervised_field_specs
    from catfood_unsupervised.dashboard.supervised_callbacks import score_and_store_supervised_row

    specs = build_supervised_field_specs(supervised_bundle.feature_options)
    values = [spec.options[0] for spec in specs]

    outcome = score_and_store_supervised_row(values, supervised_bundle)

    for spec in specs:
        assert outcome.input_payload[spec.column] == spec.options[0]


@pytest.fixture()
def supervised_bundle(tmp_path: Path):
    from catfood_unsupervised.dashboard.supervised_data_loader import SupervisedDashboardBundle
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2", f"{column}-opt-3"]
        for column in FEATURE_COLUMNS
    }
    model_path = tmp_path / "best_model.pkl"
    history_db_path = tmp_path / "prediction_history.sqlite3"

    model = _fit_constant_model(constant=1)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    return SupervisedDashboardBundle(
        metrics={
            "best_model_name": "dummy_model",
            "best_model_accuracy": 0.88,
            "best_model_macro_f1": 0.79,
            "best_model_weighted_f1": 0.86,
            "classification_report": {
                "weighted avg": {"precision": 0.9, "recall": 0.88, "f1-score": 0.86}
            },
        },
        comparison=pd.DataFrame(),
        confusion_matrix=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        predictions=pd.DataFrame(),
        feature_options=feature_options,
        model_path=model_path,
        history_db_path=history_db_path,
    )


def _collect_components(component, component_type):
    matches = []
    if isinstance(component, component_type):
        matches.append(component)

    children = getattr(component, "children", None)
    if children is None:
        return matches
    if isinstance(children, (list, tuple)):
        for child in children:
            matches.extend(_collect_components(child, component_type))
    else:
        matches.extend(_collect_components(children, component_type))
    return matches


def _collect_text(component):
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


def _fit_constant_model(constant: int) -> DummyClassifier:
    from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS

    X = pd.DataFrame(
        [[f"row0-{column}" for column in FEATURE_COLUMNS], [f"row1-{column}" for column in FEATURE_COLUMNS]],
        columns=list(FEATURE_COLUMNS),
    )
    y = pd.Series([1, 2])
    model = DummyClassifier(strategy="constant", constant=constant)
    model.fit(X, y)
    return model

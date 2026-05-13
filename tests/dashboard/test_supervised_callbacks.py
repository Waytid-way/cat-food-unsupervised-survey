from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from dash import Dash, html
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from catfood_unsupervised.dashboard.supervised_callbacks import (
    register_supervised_callbacks,
    score_and_store_supervised_row,
    store_uploaded_supervised_model,
)
from catfood_unsupervised.dashboard.supervised_data_loader import SupervisedDashboardBundle
from catfood_unsupervised.supervised.history_store import fetch_recent_prediction_history
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


@pytest.fixture()
def supervised_bundle(tmp_path: Path) -> SupervisedDashboardBundle:
    feature_options = {
        column: [f"{column}-opt-1", f"{column}-opt-2"]
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


def test_score_and_store_supervised_row_predicts_and_records_history(supervised_bundle: SupervisedDashboardBundle):
    values = [options[0] for options in supervised_bundle.feature_options.values()]

    outcome = score_and_store_supervised_row(values, supervised_bundle)

    assert outcome.prediction_frame["predicted_segment"].iloc[0] == 1
    assert any(column.startswith("prob_class_") for column in outcome.prediction_frame.columns)
    history = fetch_recent_prediction_history(supervised_bundle.history_db_path, limit=10)
    assert len(history) == 1
    assert history.iloc[0]["predicted_segment"] == 1


def test_store_uploaded_supervised_model_replaces_model_file(supervised_bundle: SupervisedDashboardBundle):
    replacement_model = _fit_constant_model(constant=2)
    replacement_bytes = pickle.dumps(replacement_model)
    contents = "data:application/octet-stream;base64," + base64.b64encode(replacement_bytes).decode("ascii")

    message = store_uploaded_supervised_model(
        contents=contents,
        filename="retrained_model.pkl",
        model_path=supervised_bundle.model_path,
    )

    assert "retrained_model.pkl" in message
    outcome = score_and_store_supervised_row(
        [options[0] for options in supervised_bundle.feature_options.values()],
        supervised_bundle,
    )
    assert outcome.prediction_frame["predicted_segment"].iloc[0] == 2


def test_register_supervised_callbacks_wires_predict_and_upload(supervised_bundle: SupervisedDashboardBundle):
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            html.Div(id="supervised-result-panel"),
            html.Div(id="supervised-history-panel"),
            html.Div(id="supervised-error-panel"),
            html.Div(id="supervised-model-status"),
            html.Button("Predict", id="supervised-predict-button"),
            html.Div(id="supervised-model-upload"),
        ]
    )

    register_supervised_callbacks(app, supervised_bundle)

    callback_keys = " ".join(app.callback_map.keys())
    assert "supervised-result-panel" in callback_keys
    assert "supervised-history-panel" in callback_keys
    assert "supervised-error-panel" in callback_keys
    assert "supervised-model-status" in callback_keys


def _fit_constant_model(constant: int) -> DummyClassifier:
    X = pd.DataFrame(
        [[f"row0-{column}" for column in FEATURE_COLUMNS], [f"row1-{column}" for column in FEATURE_COLUMNS]],
        columns=list(FEATURE_COLUMNS),
    )
    y = pd.Series([1, 2])
    model = DummyClassifier(strategy="constant", constant=constant)
    model.fit(X, y)
    return model

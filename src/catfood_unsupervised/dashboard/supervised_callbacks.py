from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from dash import Input, Output, State

from catfood_unsupervised.dashboard.components.supervised_form import (
    build_supervised_field_specs,
)
from catfood_unsupervised.dashboard.components.supervised_results import (
    render_prediction_result_panel,
    render_recent_history_panel,
)
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
)
from catfood_unsupervised.supervised.history_store import (
    append_prediction_history,
    fetch_recent_prediction_history,
)
from catfood_unsupervised.supervised.scoring import predict_supervised_segment


@dataclass(frozen=True)
class SupervisedPredictionOutcome:
    prediction_frame: pd.DataFrame
    history_frame: pd.DataFrame
    input_payload: dict[str, Any]
    probability_map: dict[str, float]


def score_and_store_supervised_row(
    values: Sequence[Any],
    bundle: SupervisedDashboardBundle,
) -> SupervisedPredictionOutcome:
    field_specs = build_supervised_field_specs(bundle.feature_options)
    if len(values) != len(field_specs):
        raise ValueError("The supervised form is missing one or more values.")

    input_payload: dict[str, Any] = {}
    missing_labels: list[str] = []
    for spec, value in zip(field_specs, values, strict=True):
        if value is None or str(value).strip() == "":
            missing_labels.append(spec.label)
        else:
            input_payload[spec.column] = value

    if missing_labels:
        raise ValueError(
            "Please complete all supervised inputs before predicting: "
            + ", ".join(missing_labels)
        )

    prediction_frame = predict_supervised_segment(bundle.model_path, input_payload)
    first_row = prediction_frame.iloc[0]
    probability_map = {
        column: float(first_row[column])
        for column in first_row.index
        if str(column).startswith("prob_class_")
    }
    predicted_segment = int(first_row["predicted_segment"])

    append_prediction_history(
        bundle.history_db_path,
        source="dashboard",
        model_name=str(bundle.metrics.get("best_model_name", bundle.model_path.stem)),
        predicted_segment=predicted_segment,
        probability_map=probability_map,
        input_payload=input_payload,
    )
    history_frame = fetch_recent_prediction_history(bundle.history_db_path, limit=8)

    return SupervisedPredictionOutcome(
        prediction_frame=prediction_frame,
        history_frame=history_frame,
        input_payload=input_payload,
        probability_map=probability_map,
    )


def register_supervised_callbacks(app, bundle: SupervisedDashboardBundle) -> None:
    field_specs = build_supervised_field_specs(bundle.feature_options)
    states = [State(spec.field_id, "value") for spec in field_specs]

    @app.callback(
        Output("supervised-result-panel", "children"),
        Output("supervised-history-panel", "children"),
        Output("supervised-error-panel", "children"),
        Input("supervised-predict-button", "n_clicks"),
        *states,
        prevent_initial_call=True,
    )
    def _handle_prediction(n_clicks: int | None, *values: Any):
        if not n_clicks:
            return (
                render_prediction_result_panel(None, bundle.metrics),
                render_recent_history_panel(None),
                "",
            )

        try:
            outcome = score_and_store_supervised_row(values, bundle)
        except Exception as exc:  # pragma: no cover - exercised via callback test
            return (
                render_prediction_result_panel(None, bundle.metrics),
                render_recent_history_panel(None),
                _error_alert(str(exc)),
            )

        return (
            render_prediction_result_panel(outcome.prediction_frame, bundle.metrics),
            render_recent_history_panel(outcome.history_frame),
            "",
        )


def _error_alert(message: str):
    from catfood_unsupervised.dashboard.bootstrap import dbc

    return dbc.Alert(message, color="danger", className="supervised-error")

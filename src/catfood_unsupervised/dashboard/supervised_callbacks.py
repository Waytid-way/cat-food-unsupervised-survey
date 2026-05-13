from __future__ import annotations

import base64
import binascii
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from dash import Input, Output, State

from catfood_unsupervised.dashboard.bootstrap import dbc
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
        resolved_value = _resolve_supervised_input_value(spec, value)
        if resolved_value is None or str(resolved_value).strip() == "":
            missing_labels.append(spec.label)
        else:
            input_payload[spec.column] = resolved_value

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


def store_uploaded_supervised_model(
    *,
    contents: str,
    filename: str | None,
    model_path: str | Path,
) -> str:
    if not contents:
        raise ValueError("No model contents were uploaded.")

    raw_bytes = _decode_upload_contents(contents)
    model = pickle.loads(raw_bytes)
    if not hasattr(model, "predict"):
        raise TypeError("Uploaded model must expose predict().")

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)

    saved_name = filename or path.name
    return f"Saved uploaded model {saved_name} to {path.resolve()}"


def register_supervised_callbacks(app, bundle: SupervisedDashboardBundle) -> None:
    field_specs = build_supervised_field_specs(bundle.feature_options)
    states = [State(spec.field_id, "value") for spec in field_specs]

    @app.callback(
        Output("supervised-model-status", "children"),
        Input("supervised-model-upload", "contents"),
        State("supervised-model-upload", "filename"),
        prevent_initial_call=True,
    )
    def _handle_model_upload(contents: str | None, filename: str | None):
        if not contents:
            return ""
        try:
            message = store_uploaded_supervised_model(
                contents=contents,
                filename=filename,
                model_path=bundle.model_path,
            )
        except Exception as exc:  # pragma: no cover - exercised by callback tests
            return _status_alert(str(exc), color="danger")
        return _status_alert(message, color="success")

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
                _status_alert(str(exc), color="danger"),
            )

        return (
            render_prediction_result_panel(outcome.prediction_frame, bundle.metrics),
            render_recent_history_panel(outcome.history_frame),
            "",
        )


def _decode_upload_contents(contents: str) -> bytes:
    try:
        _header, encoded = contents.split(",", 1)
    except ValueError as exc:
        raise ValueError("Uploaded model contents are not a valid data URI.") from exc

    try:
        return base64.b64decode(encoded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Uploaded model could not be decoded.") from exc


def _status_alert(message: str, *, color: str) -> dbc.Alert:
    return dbc.Alert(message, color=color, className="supervised-status")


def _resolve_supervised_input_value(spec, value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    options = tuple(spec.options)
    if not options:
        return value

    slider_index = _coerce_slider_index(value)
    if slider_index is not None:
        if 0 <= slider_index < len(options):
            return options[slider_index]
        raise ValueError(f"Selection value for {spec.label} is out of range.")

    if str(value) in options:
        return str(value)

    return value


def _coerce_slider_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric_value = float(text)
        except ValueError:
            return None
        if numeric_value.is_integer():
            return int(numeric_value)
    return None


__all__ = [
    "SupervisedPredictionOutcome",
    "register_supervised_callbacks",
    "score_and_store_supervised_row",
    "store_uploaded_supervised_model",
]

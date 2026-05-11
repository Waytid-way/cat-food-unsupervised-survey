from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


def load_supervised_model(model_path: str | Path):
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded supervised artifact does not expose predict().")
    return model


def prepare_supervised_input_frame(
    input_data: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, Mapping):
        frame = pd.DataFrame([dict(input_data)])
    else:
        frame = pd.DataFrame(list(input_data))

    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required supervised input columns: {missing_columns}"
        )

    ordered = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    return ordered.fillna("__missing__").astype(str)


def predict_supervised_segment(
    model_path: str | Path,
    input_data: pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    model = load_supervised_model(model_path)
    feature_frame = prepare_supervised_input_frame(input_data)

    predictions = pd.DataFrame(feature_frame.copy())
    predicted_labels = model.predict(feature_frame)
    predictions["predicted_segment"] = predicted_labels

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        for class_index, class_label in enumerate(model.classes_):
            predictions[f"prob_class_{class_label}"] = probabilities[:, class_index]

    return predictions


__all__ = [
    "load_supervised_model",
    "prepare_supervised_input_frame",
    "predict_supervised_segment",
]

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from catfood_unsupervised.supervised.config import MODEL_ORDER, N_PERMUTATION_REPEATS
from catfood_unsupervised.supervised.features import (
    build_supervised_preprocessor,
    make_supervised_pipeline,
)


def build_model_suite(random_state: int) -> Mapping[str, Pipeline]:
    def _make_logistic(feature_columns):
        return make_supervised_pipeline(
            build_supervised_preprocessor(feature_columns),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
        )

    def _make_random_forest(feature_columns):
        return make_supervised_pipeline(
            build_supervised_preprocessor(feature_columns),
            RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                class_weight="balanced",
            ),
        )

    def _make_gradient_boosting(feature_columns):
        return make_supervised_pipeline(
            build_supervised_preprocessor(feature_columns),
            GradientBoostingClassifier(random_state=random_state),
        )

    def _make_svm(feature_columns):
        return make_supervised_pipeline(
            build_supervised_preprocessor(feature_columns),
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        )

    return {
        "logistic_regression": _make_logistic,
        "random_forest": _make_random_forest,
        "gradient_boosting": _make_gradient_boosting,
        "svm_rbf": _make_svm,
    }


def evaluate_model_suite(
    model_builders: Mapping[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, pd.DataFrame] = {}

    for model_name in MODEL_ORDER:
        builder = model_builders[model_name]
        model = builder(X_train.columns)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models[model_name] = model
        prediction_frame = pd.DataFrame(
            {
                "row_id": y_test.index,
                "y_true": y_test.to_numpy(),
                "y_pred": y_pred,
            }
        )
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            for class_index, class_label in enumerate(model.classes_):
                prediction_frame[f"prob_class_{class_label}"] = proba[:, class_index]
        predictions[model_name] = prediction_frame
        rows.append(
            {
                "model_name": model_name,
                "model_rank": MODEL_ORDER.index(model_name),
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "weighted_f1", "accuracy", "model_rank"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return comparison, fitted_models, predictions


def extract_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=random_state,
        scoring="accuracy",
    )
    feature_names = list(X_test.columns)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values(
        by=["importance_mean", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

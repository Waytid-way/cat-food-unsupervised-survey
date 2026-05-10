from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame
from catfood_unsupervised.supervised.models import (
    build_model_suite,
    evaluate_model_suite,
    extract_feature_importance,
)


def test_evaluate_model_suite_returns_four_models_and_ranked_metrics(
    supervised_fixture_path,
):
    df = load_supervised_dataset(supervised_fixture_path)
    X, y = build_supervised_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7
    )

    comparison, fitted_models, predictions = evaluate_model_suite(
        build_model_suite(random_state=7),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert comparison["model_name"].tolist() == [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm_rbf",
    ]
    assert set(fitted_models) == set(comparison["model_name"])
    assert set(predictions) == set(comparison["model_name"])


def test_extract_feature_importance_returns_sorted_feature_scores(supervised_fixture_path):
    df = load_supervised_dataset(supervised_fixture_path)
    X, y = build_supervised_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7
    )

    comparison, fitted_models, _ = evaluate_model_suite(
        build_model_suite(random_state=7),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    best_model_name = comparison.iloc[0]["model_name"]
    importance = extract_feature_importance(
        fitted_models[best_model_name],
        X_test,
        y_test,
        random_state=7,
    )

    assert list(importance.columns) == ["feature", "importance_mean", "importance_std"]
    assert importance.iloc[0]["importance_mean"] >= importance.iloc[-1]["importance_mean"]

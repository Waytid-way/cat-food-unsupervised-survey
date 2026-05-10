import numpy as np
import pandas as pd

from catfood_unsupervised.models import (
    evaluate_kmeans_range,
    run_hierarchical_validation,
    run_isolation_forest,
    run_pca,
)


def test_run_pca_returns_requested_component_count():
    X = np.random.RandomState(42).randn(20, 10)

    model, scores, summary = run_pca(X, n_components=5, random_state=42)

    assert model.n_components == 5
    assert scores.shape == (20, 5)
    assert len(summary["explained_variance_ratio"]) == 5
    assert len(summary["cumulative_explained_variance"]) == 5


def test_evaluate_kmeans_range_returns_metrics_for_each_k():
    X = np.random.RandomState(7).randn(18, 4)

    summary = evaluate_kmeans_range(X, ks=[2, 3, 4], random_state=7)

    assert isinstance(summary, pd.DataFrame)
    assert summary["k"].tolist() == [2, 3, 4]
    assert {"inertia", "silhouette_score", "davies_bouldin_score"} <= set(summary.columns)


def test_evaluate_kmeans_range_returns_nan_metrics_when_labels_are_not_distinct():
    X = np.ones((6, 3))

    summary = evaluate_kmeans_range(X, ks=[2], random_state=7)

    assert summary["k"].tolist() == [2]
    assert pd.isna(summary.loc[0, "silhouette_score"])
    assert pd.isna(summary.loc[0, "davies_bouldin_score"])
    assert pd.notna(summary.loc[0, "inertia"])


def test_evaluate_kmeans_range_accepts_numpy_integer_cluster_counts():
    X = np.random.RandomState(11).randn(10, 3)

    summary = evaluate_kmeans_range(X, ks=[np.int64(2)], random_state=11)

    assert summary["k"].tolist() == [2]


def test_run_hierarchical_validation_returns_cophenetic_value():
    X = np.random.RandomState(0).randn(12, 4)

    linkage_matrix, summary = run_hierarchical_validation(X)

    assert linkage_matrix.shape == (11, 4)
    assert 0.0 <= summary["cophenetic_correlation"] <= 1.0


def test_run_isolation_forest_returns_binary_flags():
    X = np.random.RandomState(0).randn(25, 6)

    result = run_isolation_forest(X, contamination=0.12, random_state=42)

    assert isinstance(result, pd.DataFrame)
    assert set(result["anomaly_flag"]) <= {0, 1}
    assert len(result) == 25
    assert "anomaly_score" in result.columns

from __future__ import annotations

from numbers import Integral
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score


def run_pca(
    X: Any, n_components: int, random_state: int | None = None
) -> tuple[PCA, pd.DataFrame, dict[str, Any]]:
    array, index = _coerce_feature_matrix(X, function_name="run_pca")
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")

    max_components = min(array.shape)
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum allowed value of {max_components}."
        )

    model = PCA(n_components=n_components, random_state=random_state)
    transformed = model.fit_transform(array)
    component_columns = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(transformed, index=index, columns=component_columns)

    explained_variance_ratio = model.explained_variance_ratio_.tolist()
    cumulative_explained_variance = np.cumsum(model.explained_variance_ratio_).tolist()
    summary = {
        "n_components": int(n_components),
        "explained_variance": model.explained_variance_.tolist(),
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": cumulative_explained_variance,
        "singular_values": model.singular_values_.tolist(),
        "total_explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
    }
    return model, scores, summary


def evaluate_kmeans_range(
    X: Any, ks: Sequence[int], random_state: int | None = None
) -> pd.DataFrame:
    array, _ = _coerce_feature_matrix(X, function_name="evaluate_kmeans_range")
    k_values = list(ks)
    summary_columns = [
        "k",
        "inertia",
        "silhouette_score",
        "davies_bouldin_score",
    ]
    if not k_values:
        return pd.DataFrame(columns=summary_columns)

    rows: list[dict[str, float | int]] = []
    for k in k_values:
        normalized_k = _validate_cluster_count(k, sample_count=array.shape[0])

        model = KMeans(n_clusters=normalized_k, n_init=30, random_state=random_state)
        labels = model.fit_predict(array)
        silhouette, davies_bouldin = _summarize_cluster_metrics(array, labels)
        rows.append(
            {
                "k": normalized_k,
                "inertia": float(model.inertia_),
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
            }
        )

    return pd.DataFrame(rows, columns=summary_columns)


def run_hierarchical_validation(
    X: Any,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    array, _ = _coerce_feature_matrix(X, function_name="run_hierarchical_validation")
    if array.shape[0] < 2:
        raise ValueError("run_hierarchical_validation requires at least 2 samples.")

    pairwise_distances = pdist(array, metric="euclidean")
    linkage_matrix = linkage(array, method="ward")
    cophenetic_correlation, _ = cophenet(linkage_matrix, pairwise_distances)
    summary = {
        "linkage_method": "ward",
        "distance_metric": "euclidean",
        "cophenetic_correlation": float(cophenetic_correlation),
        "n_observations": int(array.shape[0]),
    }
    return linkage_matrix, summary


def run_isolation_forest(
    X: Any, contamination: float | str = "auto", random_state: int | None = None
) -> pd.DataFrame:
    array, index = _coerce_feature_matrix(X, function_name="run_isolation_forest")
    if contamination != "auto" and not (0 < float(contamination) <= 0.5):
        raise ValueError("contamination must be 'auto' or a float in the interval (0, 0.5].")

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
    )
    predictions = model.fit_predict(array)
    anomaly_scores = -model.score_samples(array)
    return pd.DataFrame(
        {
            "anomaly_flag": (predictions == -1).astype(int),
            "anomaly_score": anomaly_scores.astype(float),
        },
        index=index,
    )


def _coerce_feature_matrix(
    X: Any, *, function_name: str
) -> tuple[np.ndarray, pd.Index | pd.RangeIndex]:
    if isinstance(X, pd.DataFrame):
        index = X.index
        array = X.to_numpy(dtype=float, copy=False)
    else:
        array = np.asarray(X, dtype=float)
        index = pd.RangeIndex(array.shape[0]) if array.ndim == 2 else pd.RangeIndex(0)

    if array.ndim != 2:
        raise ValueError(f"{function_name} expects a 2D array-like input.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{function_name} requires at least one row and one column.")
    if not np.isfinite(array).all():
        raise ValueError(f"{function_name} requires all values to be finite.")

    return array, index


def _summarize_cluster_metrics(
    array: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    distinct_label_count = len(np.unique(labels))
    if not 2 <= distinct_label_count < len(labels):
        return float("nan"), float("nan")

    return (
        float(silhouette_score(array, labels)),
        float(davies_bouldin_score(array, labels)),
    )


def _validate_cluster_count(k: int, sample_count: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError("Each k value must be an integer.")
    normalized_k = int(k)
    if normalized_k < 2:
        raise ValueError("Each k value must be at least 2.")
    if normalized_k >= sample_count:
        raise ValueError(
            f"Each k value must be smaller than the sample count; received k={normalized_k} for {sample_count} samples."
        )
    return normalized_k

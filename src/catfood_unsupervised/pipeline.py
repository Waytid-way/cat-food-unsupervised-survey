from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from catfood_unsupervised.config import OUTPUT_DIR, RAW_DATA_PATH
from catfood_unsupervised.data_loading import (
    filter_completed_responses,
    load_raw_export,
)
from catfood_unsupervised.features import build_vote_features, ipsatize_rows
from catfood_unsupervised.models import (
    evaluate_kmeans_range,
    run_hierarchical_validation,
    run_isolation_forest,
    run_pca,
)
from sklearn.model_selection import train_test_split
from catfood_unsupervised.preprocessing import (
    fit_buy_factor_imputer,
    map_series_values,
    transform_buy_factors,
)


TOP3_COLUMN_INDEX = 72
AGE_COLUMN_INDEX = 73
GENDER_COLUMN_INDEX = 74
MARITAL_COLUMN_INDEX = 75
BUY_FACTOR_COLUMN_INDICES = range(5, 10)
PACKAGING_EFFECT_COLUMN_INDEX = 10
PACKAGING_IMPORTANCE_COLUMN_INDICES = range(12, 20)
OPTION_RATING_COLUMN_INDICES = range(22, 72)
OPTION_COUNT = 10
SEGMENT_COLUMN = "segment"
TOP3_PATTERN = re.compile(r"Option\s*(\d+)", flags=re.IGNORECASE)

IMPORTANCE_MAPPING = {
    "น้อยที่สุด": 1,
    "น้อย": 2,
    "ปานกลาง": 3,
    "มาก": 4,
    "มากที่สุด": 5,
}
AGREEMENT_MAPPING = {
    "ไม่เห็นด้วยเลย": 1,
    "ไม่เห็นด้วย": 2,
    "เฉยๆ": 3,
    "เห็นด้วย": 4,
    "เห็นด้วยที่สุด": 5,
}
PACKAGING_EFFECT_MAPPING = {
    "ไม่มีผล": 0,
    "มีผล": 1,
}


def run_pipeline(
    data_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    random_state: int = 42,
    k_values: Sequence[int] | None = None,
    pca_variance_threshold: float = 0.8,
    max_pca_components: int = 8,
    isolation_contamination: float | str = "auto",
) -> dict[str, Any]:
    raw_df = load_raw_export(data_path)
    top3_column = raw_df.columns[TOP3_COLUMN_INDEX]
    completed_df = filter_completed_responses(raw_df, top3_column=top3_column)

    if len(completed_df) < 3:
        raise ValueError("run_pipeline requires at least 3 completed top-3 responses.")

    feature_bundle = _build_feature_bundle(
        completed_df, top3_column=top3_column, random_state=random_state
    )
    pca_bundle = _run_pca_pipeline(
        feature_bundle["ipsatized_option_ratings"],
        random_state=random_state,
        variance_threshold=pca_variance_threshold,
        max_pca_components=max_pca_components,
    )

    resolved_k_values = _resolve_k_values(len(completed_df), k_values)
    kmeans_evaluation = evaluate_kmeans_range(
        pca_bundle["scores"],
        ks=resolved_k_values,
        random_state=random_state,
    )
    final_cluster_k = _select_final_k(kmeans_evaluation)
    segment_series = _fit_final_segments(
        pca_bundle["scores"],
        n_clusters=final_cluster_k,
        random_state=random_state,
    )

    _, hierarchical_summary = run_hierarchical_validation(pca_bundle["scores"])
    anomaly_results = run_isolation_forest(
        feature_bundle["anomaly_features"],
        contamination=isolation_contamination,
        random_state=random_state,
    )

    enriched_df = _build_enriched_dataframe(
        completed_df=completed_df,
        ranking_df=feature_bundle["rankings"],
        vote_features=feature_bundle["vote_features"],
        pca_scores=pca_bundle["scores"],
        segment_series=segment_series,
        anomaly_results=anomaly_results,
    )
    segment_profiles = _build_segment_profiles(
        anomaly_features=feature_bundle["anomaly_features"],
        segment_series=segment_series,
    )
    correlation_matrix = feature_bundle["correlation_features"].corr(numeric_only=True)

    output_paths = _build_output_paths(output_dir)
    metrics = _build_metrics(
        data_path=data_path,
        output_paths=output_paths,
        raw_df=raw_df,
        completed_df=completed_df,
        top3_column=top3_column,
        pca_bundle=pca_bundle,
        kmeans_evaluation=kmeans_evaluation,
        final_cluster_k=final_cluster_k,
        hierarchical_summary=hierarchical_summary,
        segment_series=segment_series,
        anomaly_results=anomaly_results,
        isolation_contamination=isolation_contamination,
        feature_bundle=feature_bundle,
    )
    _write_outputs(
        output_paths=output_paths,
        enriched_df=enriched_df,
        metrics=metrics,
        correlation_matrix=correlation_matrix,
        segment_profiles=segment_profiles,
    )
    metrics["output_files"] = {name: str(path) for name, path in output_paths.items()}

    return {
        "metrics": metrics,
        "dataframe": enriched_df,
        "segment_profiles": segment_profiles,
        "correlation_matrix": correlation_matrix,
    }


def _build_feature_bundle(
    completed_df: pd.DataFrame, *, top3_column: str, random_state: int
) -> dict[str, pd.DataFrame]:
    rankings = _extract_top3_rankings(completed_df[top3_column])
    vote_features = build_vote_features(
        rankings,
        rank_columns=rankings.columns.tolist(),
        option_count=OPTION_COUNT,
    )

    buy_factors = _encode_block(
        completed_df,
        BUY_FACTOR_COLUMN_INDICES,
        IMPORTANCE_MAPPING,
        prefix="buy_factor",
    )
    imputation_reference_df, _ = train_test_split(
        buy_factors,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    buy_factor_artifact = fit_buy_factor_imputer(
        imputation_reference_df,
        buy_factors.columns.tolist(),
        n_neighbors=5,
    )
    imputed_buy_factors = transform_buy_factors(
        buy_factors,
        buy_factors.columns.tolist(),
        buy_factor_artifact,
    )
    imputation_reference_rows = int(len(imputation_reference_df))
    packaging_effect = _encode_block(
        completed_df,
        [PACKAGING_EFFECT_COLUMN_INDEX],
        PACKAGING_EFFECT_MAPPING,
        prefix="packaging_effect",
    )
    packaging_importance = _encode_block(
        completed_df,
        PACKAGING_IMPORTANCE_COLUMN_INDICES,
        IMPORTANCE_MAPPING,
        prefix="packaging_importance",
    )
    option_ratings = _encode_option_ratings(completed_df)
    ipsatized_option_ratings = pd.DataFrame(
        ipsatize_rows(option_ratings.to_numpy()),
        index=completed_df.index,
        columns=[f"{column}_ips" for column in option_ratings.columns],
    ).fillna(0.0)
    demographics = _encode_demographics(completed_df)

    anomaly_features = pd.concat(
        [
            vote_features.astype(float),
            imputed_buy_factors.astype(float),
            packaging_effect.astype(float),
            packaging_importance.astype(float),
            ipsatized_option_ratings.astype(float),
            demographics.astype(float),
        ],
        axis=1,
    )

    return {
        "rankings": rankings,
        "vote_features": vote_features,
        "buy_factors": imputed_buy_factors,
        "packaging_effect": packaging_effect,
        "packaging_importance": packaging_importance,
        "option_ratings": option_ratings,
        "ipsatized_option_ratings": ipsatized_option_ratings,
        "demographics": demographics,
        "anomaly_features": anomaly_features,
        "correlation_features": anomaly_features,
        "imputation_reference_rows": imputation_reference_rows,
    }


def _extract_top3_rankings(top3_series: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for row_number, value in enumerate(top3_series.tolist()):
        options = TOP3_PATTERN.findall("" if pd.isna(value) else str(value))
        if len(options) < 3:
            raise ValueError(
                f"Expected at least 3 option references in top-3 response row {row_number}."
            )
        rows.append(
            {
                "rank1": int(options[0]),
                "rank2": int(options[1]),
                "rank3": int(options[2]),
            }
        )
    return pd.DataFrame(rows, index=top3_series.index)


def _encode_block(
    df: pd.DataFrame,
    column_indices: Sequence[int],
    mapping: dict[str, int],
    *,
    prefix: str,
) -> pd.DataFrame:
    encoded_columns: dict[str, pd.Series] = {}
    for position, column_index in enumerate(column_indices, start=1):
        column_name = df.columns[column_index]
        encoded = map_series_values(df[column_name], mapping, default=np.nan)
        encoded_columns[f"{prefix}_{position:02d}"] = pd.to_numeric(
            encoded, errors="coerce"
        )
    return pd.DataFrame(encoded_columns, index=df.index)


def _encode_option_ratings(df: pd.DataFrame) -> pd.DataFrame:
    encoded_columns: dict[str, pd.Series] = {}
    for offset, column_index in enumerate(OPTION_RATING_COLUMN_INDICES):
        option_index, attribute_index = divmod(offset, 5)
        column_name = df.columns[column_index]
        encoded = map_series_values(df[column_name], AGREEMENT_MAPPING, default=np.nan)
        encoded_columns[
            f"option_{option_index + 1:02d}_attribute_{attribute_index + 1:02d}"
        ] = pd.to_numeric(encoded, errors="coerce")
    return pd.DataFrame(encoded_columns, index=df.index)


def _encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    demographic_frame = pd.DataFrame(
        {
            "age": df.iloc[:, AGE_COLUMN_INDEX].astype(str).str.strip(),
            "gender": df.iloc[:, GENDER_COLUMN_INDEX].astype(str).str.strip(),
            "marital": df.iloc[:, MARITAL_COLUMN_INDEX].astype(str).str.strip(),
        },
        index=df.index,
    )
    return pd.get_dummies(
        demographic_frame,
        columns=["age", "gender", "marital"],
        prefix=["age", "gender", "marital"],
        dtype=int,
    )


def _run_pca_pipeline(
    ipsatized_option_ratings: pd.DataFrame,
    *,
    random_state: int,
    variance_threshold: float,
    max_pca_components: int,
) -> dict[str, Any]:
    standardized = pd.DataFrame(
        StandardScaler().fit_transform(ipsatized_option_ratings),
        index=ipsatized_option_ratings.index,
        columns=ipsatized_option_ratings.columns,
    )
    max_components = min(
        max_pca_components,
        standardized.shape[0] - 1,
        standardized.shape[1],
    )
    if max_components < 2:
        raise ValueError("PCA requires at least 2 usable components for clustering.")

    _, _, exploratory_summary = run_pca(
        standardized, n_components=max_components, random_state=random_state
    )
    selected_components = _select_pca_component_count(
        exploratory_summary["cumulative_explained_variance"],
        variance_threshold=variance_threshold,
        max_components=max_components,
    )
    model, scores, selected_summary = run_pca(
        standardized, n_components=selected_components, random_state=random_state
    )

    return {
        "model": model,
        "scores": scores,
        "standardized_input": standardized,
        "summary": {
            **selected_summary,
            "input_feature_count": int(ipsatized_option_ratings.shape[1]),
            "ipsatized": True,
            "standardized": True,
            "variance_threshold": float(variance_threshold),
            "max_evaluated_components": int(max_components),
            "selected_n_components": int(selected_components),
            "exploratory_cumulative_explained_variance": exploratory_summary[
                "cumulative_explained_variance"
            ],
        },
    }


def _select_pca_component_count(
    cumulative_explained_variance: Sequence[float],
    *,
    variance_threshold: float,
    max_components: int,
) -> int:
    for component_index, cumulative_value in enumerate(
        cumulative_explained_variance, start=1
    ):
        if cumulative_value >= variance_threshold:
            return max(2, component_index)
    return max(2, max_components)


def _resolve_k_values(
    sample_count: int, k_values: Sequence[int] | None
) -> list[int]:
    resolved = (
        list(range(2, min(7, sample_count - 1) + 1))
        if k_values is None
        else [int(k) for k in k_values]
    )
    if not resolved:
        raise ValueError("No valid k values are available for the current sample size.")
    return resolved


def _select_final_k(kmeans_evaluation: pd.DataFrame) -> int:
    scored = kmeans_evaluation.copy()
    scored["silhouette_rank"] = scored["silhouette_score"].fillna(-np.inf)
    scored["davies_rank"] = scored["davies_bouldin_score"].fillna(np.inf)
    ranked = scored.sort_values(
        by=["silhouette_rank", "davies_rank", "inertia", "k"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return int(ranked.iloc[0]["k"])


def _fit_final_segments(
    pca_scores: pd.DataFrame, *, n_clusters: int, random_state: int
) -> pd.Series:
    model = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
    labels = model.fit_predict(pca_scores)
    return pd.Series(labels + 1, index=pca_scores.index, name=SEGMENT_COLUMN)


def _build_enriched_dataframe(
    *,
    completed_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    vote_features: pd.DataFrame,
    pca_scores: pd.DataFrame,
    segment_series: pd.Series,
    anomaly_results: pd.DataFrame,
) -> pd.DataFrame:
    enriched = completed_df.reset_index(drop=True).copy()
    rankings = ranking_df.reset_index(drop=True).rename(
        columns={
            "rank1": "top3_rank_1",
            "rank2": "top3_rank_2",
            "rank3": "top3_rank_3",
        }
    )
    return pd.concat(
        [
            enriched,
            rankings,
            vote_features.reset_index(drop=True),
            pca_scores.reset_index(drop=True),
            segment_series.reset_index(drop=True),
            anomaly_results.reset_index(drop=True),
        ],
        axis=1,
    )


def _build_segment_profiles(
    *, anomaly_features: pd.DataFrame, segment_series: pd.Series
) -> pd.DataFrame:
    profiled = anomaly_features.copy()
    profiled[SEGMENT_COLUMN] = segment_series
    summary = profiled.groupby(SEGMENT_COLUMN).mean(numeric_only=True)
    summary.insert(0, "segment_size", segment_series.value_counts().sort_index())
    return summary


def _build_metrics(
    *,
    data_path: str | Path,
    output_paths: dict[str, Path],
    raw_df: pd.DataFrame,
    completed_df: pd.DataFrame,
    top3_column: str,
    pca_bundle: dict[str, Any],
    kmeans_evaluation: pd.DataFrame,
    final_cluster_k: int,
    hierarchical_summary: dict[str, Any],
    segment_series: pd.Series,
    anomaly_results: pd.DataFrame,
    isolation_contamination: float | str,
    feature_bundle: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        "data_path": str(Path(data_path).resolve()),
        "row_counts": {
            "raw_loaded": int(len(raw_df)),
            "completed_top3": int(len(completed_df)),
        },
        "selected_top3_column": top3_column,
        "pca": pca_bundle["summary"],
        "kmeans_evaluation": kmeans_evaluation.to_dict(orient="records"),
        "final_cluster_k": int(final_cluster_k),
        "final_cluster_key": SEGMENT_COLUMN,
        "hierarchical_validation": hierarchical_summary,
        "anomaly_detection": {
            "contamination": isolation_contamination,
            "anomaly_count": int(anomaly_results["anomaly_flag"].sum()),
            "anomaly_rate": float(anomaly_results["anomaly_flag"].mean()),
        },
        "segment_sizes": {
            str(int(segment)): int(size)
            for segment, size in segment_series.value_counts().sort_index().items()
        },
        "output_files": {name: str(path) for name, path in output_paths.items()},
        "imputation": {
            "reference_rows": int(feature_bundle["imputation_reference_rows"]),
            "target_rows": int(len(completed_df)),
            "reference_fraction": float(
                feature_bundle["imputation_reference_rows"] / len(completed_df)
            ),
            "n_neighbors": 5,
        },
    }
    return _json_safe(metrics)


def _write_outputs(
    *,
    output_paths: dict[str, Path],
    enriched_df: pd.DataFrame,
    metrics: dict[str, Any],
    correlation_matrix: pd.DataFrame,
    segment_profiles: pd.DataFrame,
) -> None:
    enriched_df.to_csv(output_paths["clean_dataset_with_segments"], index=False)
    correlation_matrix.to_csv(output_paths["correlation_matrix"])
    segment_profiles.to_csv(output_paths["segment_profiles"])
    output_paths["metrics_summary"].write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return {
        "clean_dataset_with_segments": output_path / "clean_dataset_with_segments.csv",
        "metrics_summary": output_path / "metrics_summary.json",
        "correlation_matrix": output_path / "correlation_matrix.csv",
        "segment_profiles": output_path / "segment_profiles.csv",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value

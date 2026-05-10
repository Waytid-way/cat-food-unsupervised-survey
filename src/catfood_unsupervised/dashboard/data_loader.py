from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DashboardData:
    metrics: dict
    clean_df: pd.DataFrame
    correlation_matrix: pd.DataFrame
    segment_profiles: pd.DataFrame
    pca_scores: pd.DataFrame


def load_metrics_summary(output_dir: Path) -> dict:
    return json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))


def load_all_data(output_dir: Path) -> DashboardData:
    metrics = load_metrics_summary(output_dir)
    try:
        clean_df = pd.read_csv(output_dir / "clean_dataset_with_segments.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"clean_dataset_with_segments.csv not found in {output_dir}")
    try:
        correlation_matrix = pd.read_csv(output_dir / "correlation_matrix.csv", index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"correlation_matrix.csv not found in {output_dir}")
    try:
        segment_profiles = pd.read_csv(output_dir / "segment_profiles.csv", index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"segment_profiles.csv not found in {output_dir}")
    pca_scores = _build_pca_scores(clean_df, metrics)
    return DashboardData(
        metrics=metrics,
        clean_df=clean_df,
        correlation_matrix=correlation_matrix,
        segment_profiles=segment_profiles,
        pca_scores=pca_scores,
    )


def _build_pca_scores(clean_df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    n_components = metrics["pca"]["n_components"]
    columns = [f"PC{i+1}" for i in range(n_components)]
    return clean_df[columns].copy()
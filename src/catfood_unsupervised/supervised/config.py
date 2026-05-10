from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.supervised.schema import (
    ANOMALY_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "clean_dataset_with_segments.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "supervised"

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_PERMUTATION_REPEATS = 10

__all__ = [
    "ANOMALY_COLUMN",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_DIR",
    "FEATURE_COLUMNS",
    "LEAKAGE_COLUMNS",
    "MODEL_ORDER",
    "N_PERMUTATION_REPEATS",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "TEST_SIZE",
]

from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.shared.paths import (
    PROJECT_ROOT,
    SUPERVISED_OUTPUT_DIR,
    SUPERVISED_REPORT_DIR,
)

from catfood_unsupervised.supervised.schema import (
    ANOMALY_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
)


DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "clean_dataset_with_segments.csv"
DEFAULT_OUTPUT_DIR = SUPERVISED_OUTPUT_DIR
DEFAULT_REPORT_DIR = SUPERVISED_REPORT_DIR

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_PERMUTATION_REPEATS = 10
BEST_MODEL_FILENAME = "best_model.pkl"
SCORING_ENTRYPOINT = "catfood_unsupervised.supervised.scoring:predict_supervised_segment"

__all__ = [
    "ANOMALY_COLUMN",
    "BEST_MODEL_FILENAME",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_DIR",
    "FEATURE_COLUMNS",
    "LEAKAGE_COLUMNS",
    "MODEL_ORDER",
    "N_PERMUTATION_REPEATS",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "SCORING_ENTRYPOINT",
    "SUPERVISED_OUTPUT_DIR",
    "SUPERVISED_REPORT_DIR",
    "TARGET_COLUMN",
    "TEST_SIZE",
]

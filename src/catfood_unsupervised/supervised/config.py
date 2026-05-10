from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "clean_dataset_with_segments.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "supervised"

TARGET_COLUMN = "segment"
ANOMALY_COLUMN = "anomaly_flag"

# The cleaned dataset keeps the original survey columns and appends derived
# columns at the end. These indices match the current exported CSV contract.
FEATURE_COLUMN_INDICES = (
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    73,
    74,
    75,
)

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_PERMUTATION_REPEATS = 10

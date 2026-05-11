from __future__ import annotations

from catfood_unsupervised.supervised.config import (
    ANOMALY_COLUMN,
    BEST_MODEL_FILENAME,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_ORDER,
    N_PERMUTATION_REPEATS,
    PROJECT_ROOT,
    RANDOM_STATE,
    SCORING_ENTRYPOINT,
    TARGET_COLUMN,
    TEST_SIZE,
)
from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame
from catfood_unsupervised.supervised.history_store import (
    append_prediction_history,
    fetch_recent_prediction_history,
    initialize_prediction_history_db,
)
from catfood_unsupervised.supervised.models import build_model_suite
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.reporting import (
    run_supervised_workflow,
    write_reports_from_output_dir,
)
from catfood_unsupervised.supervised.scoring import (
    load_supervised_model,
    predict_supervised_segment,
    prepare_supervised_input_frame,
)

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
    "TARGET_COLUMN",
    "TEST_SIZE",
    "append_prediction_history",
    "build_model_suite",
    "build_supervised_feature_frame",
    "fetch_recent_prediction_history",
    "initialize_prediction_history_db",
    "load_supervised_dataset",
    "load_supervised_model",
    "predict_supervised_segment",
    "prepare_supervised_input_frame",
    "run_supervised_pipeline",
    "run_supervised_workflow",
    "write_reports_from_output_dir",
]

from __future__ import annotations

from repo_bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.compat.supervised import (
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
    append_prediction_history,
    build_model_suite,
    build_supervised_feature_frame,
    fetch_recent_prediction_history,
    initialize_prediction_history_db,
    load_supervised_dataset,
    load_supervised_model,
    predict_supervised_segment,
    prepare_supervised_input_frame,
    run_supervised_pipeline,
    run_supervised_workflow,
    write_reports_from_output_dir,
)
from catfood_unsupervised.supervised.features import (
    build_supervised_preprocessor,
    get_supervised_feature_columns,
    make_supervised_pipeline,
)
from catfood_unsupervised.supervised.models import (
    evaluate_model_suite,
    extract_feature_importance,
)
from catfood_unsupervised.supervised.reporting import (
    REPORT_FILENAMES,
    SupervisedReportContext,
    load_report_context,
    render_supervised_model_report,
    render_supervised_owner_memo,
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
    "REPORT_FILENAMES",
    "SCORING_ENTRYPOINT",
    "TARGET_COLUMN",
    "TEST_SIZE",
    "SupervisedReportContext",
    "append_prediction_history",
    "build_model_suite",
    "build_supervised_feature_frame",
    "build_supervised_preprocessor",
    "evaluate_model_suite",
    "extract_feature_importance",
    "fetch_recent_prediction_history",
    "get_supervised_feature_columns",
    "initialize_prediction_history_db",
    "load_report_context",
    "load_supervised_dataset",
    "load_supervised_model",
    "make_supervised_pipeline",
    "predict_supervised_segment",
    "prepare_supervised_input_frame",
    "render_supervised_model_report",
    "render_supervised_owner_memo",
    "run_supervised_pipeline",
    "run_supervised_workflow",
    "write_reports_from_output_dir",
]

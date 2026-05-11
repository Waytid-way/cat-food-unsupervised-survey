from __future__ import annotations

from catfood_unsupervised.dashboard.app import (
    INITIAL_TAB,
    KPI_METRICS,
    dash_app,
    dashboard_data,
    render_tab_content,
    supervised_dashboard_data,
)
from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.components.kpi_banner import render_kpi_banner
from catfood_unsupervised.dashboard.components.shell import render_dashboard_shell
from catfood_unsupervised.dashboard.components.supervised_form import (
    SupervisedFieldSpec,
    build_supervised_field_specs,
    field_input_id,
    render_supervised_form,
)
from catfood_unsupervised.dashboard.components.supervised_results import (
    build_confidence_figure,
    render_prediction_result_panel,
    render_recent_history_panel,
)
from catfood_unsupervised.dashboard.components.tab_business_insight import (
    render_business_insight_tab,
)
from catfood_unsupervised.dashboard.components.tab_clustering import (
    render_clustering_tab,
)
from catfood_unsupervised.dashboard.components.tab_correlation import (
    render_correlation_tab,
)
from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab
from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab
from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab
from catfood_unsupervised.dashboard.config import (
    KPI_CARDS,
    SUPERVISED_HISTORY_DB_FILENAME,
    SUPERVISED_HISTORY_DB_PATH,
    SUPERVISED_MODEL_PATH,
    SUPERVISED_OUTPUT_DIR,
    TAB_ITEMS,
)
from catfood_unsupervised.dashboard.data_loader import DashboardData, load_all_data, load_metrics_summary
from catfood_unsupervised.dashboard.supervised_callbacks import (
    SupervisedPredictionOutcome,
    register_supervised_callbacks,
    score_and_store_supervised_row,
)
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
    load_supervised_dashboard_bundle,
    load_supervised_feature_options,
)

__all__ = [
    "DashboardData",
    "INITIAL_TAB",
    "KPI_CARDS",
    "KPI_METRICS",
    "SUPERVISED_HISTORY_DB_FILENAME",
    "SUPERVISED_HISTORY_DB_PATH",
    "SUPERVISED_MODEL_PATH",
    "SUPERVISED_OUTPUT_DIR",
    "SupervisedDashboardBundle",
    "SupervisedFieldSpec",
    "SupervisedPredictionOutcome",
    "TAB_ITEMS",
    "build_confidence_figure",
    "build_supervised_field_specs",
    "dash_app",
    "dashboard_data",
    "dbc",
    "field_input_id",
    "load_all_data",
    "load_metrics_summary",
    "load_supervised_dashboard_bundle",
    "load_supervised_feature_options",
    "register_supervised_callbacks",
    "render_business_insight_tab",
    "render_clustering_tab",
    "render_correlation_tab",
    "render_dashboard_shell",
    "render_eda_tab",
    "render_kpi_banner",
    "render_persona_tab",
    "render_prediction_result_panel",
    "render_recent_history_panel",
    "render_supervised_form",
    "render_supervised_tab",
    "render_tab_content",
    "score_and_store_supervised_row",
    "supervised_dashboard_data",
]

from __future__ import annotations

from catfood_unsupervised.dashboard.app import dash_app
from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.dashboard.config import (
    KPI_CARDS,
    SUPERVISED_HISTORY_DB_FILENAME,
    SUPERVISED_HISTORY_DB_PATH,
    SUPERVISED_MODEL_PATH,
    SUPERVISED_OUTPUT_DIR,
    TAB_ITEMS,
)
from catfood_unsupervised.dashboard.data_loader import DashboardData, load_all_data, load_metrics_summary
from catfood_unsupervised.dashboard.supervised_data_loader import (
    SupervisedDashboardBundle,
    load_supervised_dashboard_bundle,
    load_supervised_feature_options,
)

__all__ = [
    "DashboardData",
    "KPI_CARDS",
    "SupervisedDashboardBundle",
    "TAB_ITEMS",
    "dash_app",
    "dbc",
    "load_all_data",
    "load_metrics_summary",
    "load_supervised_dashboard_bundle",
    "load_supervised_feature_options",
    "SUPERVISED_HISTORY_DB_FILENAME",
    "SUPERVISED_HISTORY_DB_PATH",
    "SUPERVISED_MODEL_PATH",
    "SUPERVISED_OUTPUT_DIR",
]
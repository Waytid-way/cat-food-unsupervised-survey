from __future__ import annotations

from importlib import import_module


def test_canonical_import_surface_is_still_available():
    pipeline = import_module("catfood_unsupervised.pipeline")
    supervised_pipeline = import_module("catfood_unsupervised.supervised.pipeline")
    supervised_scoring = import_module("catfood_unsupervised.supervised.scoring")
    dashboard_app = import_module("catfood_unsupervised.dashboard.app")

    assert callable(pipeline.run_pipeline)
    assert callable(supervised_pipeline.run_supervised_pipeline)
    assert callable(supervised_scoring.predict_supervised_segment)
    assert hasattr(dashboard_app, "dash_app")


def test_compatibility_modules_re_export_current_public_apis():
    compat_unsupervised = import_module("catfood_unsupervised.compat.unsupervised")
    compat_supervised = import_module("catfood_unsupervised.compat.supervised")
    compat_dashboard = import_module("catfood_unsupervised.compat.dashboard")
    compat_bootstrap = import_module("catfood_unsupervised.compat.bootstrap")

    assert callable(compat_unsupervised.run_pipeline)
    assert callable(compat_unsupervised.run_unsupervised_workflow)
    assert callable(compat_supervised.run_supervised_pipeline)
    assert callable(compat_supervised.predict_supervised_segment)
    assert hasattr(compat_supervised, "FEATURE_COLUMNS")
    assert hasattr(compat_dashboard, "dash_app")
    assert callable(compat_dashboard.render_tab_content)
    assert callable(compat_dashboard.render_dashboard_shell)
    assert hasattr(compat_bootstrap, "dbc")

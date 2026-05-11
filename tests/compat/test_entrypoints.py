from __future__ import annotations

from importlib import import_module

from catfood_unsupervised.supervised.config import SCORING_ENTRYPOINT


def test_scoring_entrypoint_resolves_to_callable():
    module_path, function_name = SCORING_ENTRYPOINT.split(":")
    module = import_module(module_path)

    assert callable(getattr(module, function_name))


def test_dashboard_module_entrypoint_imports_canonical_and_compat_paths():
    canonical_dashboard_main = import_module("catfood_unsupervised.dashboard.__main__")
    canonical_dashboard_app = import_module("catfood_unsupervised.dashboard.app")
    compat_dashboard = import_module("catfood_unsupervised.compat.dashboard")

    assert canonical_dashboard_main.app.title == "CatFood ML Dashboard"
    assert canonical_dashboard_app.dash_app.title == "CatFood ML Dashboard"
    assert compat_dashboard.dash_app.title == "CatFood ML Dashboard"

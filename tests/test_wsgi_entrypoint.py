from __future__ import annotations

from importlib import import_module


def test_wsgi_exposes_dash_server_callable():
    wsgi = import_module("wsgi")
    dash_app = import_module("catfood_unsupervised.dashboard.app").dash_app

    assert wsgi.server is dash_app.server
    assert wsgi.app is wsgi.server

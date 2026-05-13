from __future__ import annotations

from repo_bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.dashboard.app import dash_app

server = dash_app.server
app = server


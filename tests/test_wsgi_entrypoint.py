from __future__ import annotations

from importlib import import_module
import os
import subprocess
import sys
from pathlib import Path


def test_wsgi_exposes_dash_server_callable():
    wsgi = import_module("wsgi")
    dash_app = import_module("catfood_unsupervised.dashboard.app").dash_app

    assert wsgi.server is dash_app.server
    assert wsgi.app is wsgi.server


def test_wsgi_bootstraps_src_from_repo_root():
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import wsgi; import catfood_unsupervised.dashboard.app as app; "
            "assert wsgi.server is app.dash_app.server; print('ok')",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "ok"

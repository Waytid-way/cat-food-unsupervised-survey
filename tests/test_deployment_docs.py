from __future__ import annotations

from pathlib import Path


def test_deployment_docs_document_build_and_runtime_commands():
    repo_root = Path(__file__).resolve().parents[1]

    readme_path = repo_root / "README.md"
    procfile_path = repo_root / "Procfile"

    assert readme_path.exists(), "README.md is missing"
    assert procfile_path.exists(), "Procfile is missing"

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "python scripts/build_deployment_artifacts.py" in readme_text
    assert "gunicorn wsgi:server --bind 0.0.0.0:$PORT" in readme_text
    assert "python -m catfood_unsupervised.dashboard" in readme_text

    assert procfile_path.read_text(encoding="utf-8") == (
        "web: gunicorn wsgi:server --bind 0.0.0.0:$PORT"
    )

from __future__ import annotations

from pathlib import Path


def test_deployment_docs_document_build_and_runtime_commands():
    repo_root = Path(__file__).resolve().parents[1]

    readme_path = repo_root / "README.md"
    procfile_path = repo_root / "Procfile"
    render_path = repo_root / "render.yaml"

    assert readme_path.exists(), "README.md is missing"
    assert procfile_path.exists(), "Procfile is missing"
    assert render_path.exists(), "render.yaml is missing"

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "python scripts/build_deployment_artifacts.py" in readme_text
    assert "gunicorn wsgi:server --bind 0.0.0.0:$PORT" in readme_text
    assert "python -m catfood_unsupervised.dashboard" in readme_text

    assert procfile_path.read_text(encoding="utf-8") == (
        "web: gunicorn wsgi:server --bind 0.0.0.0:$PORT"
    )

    render_text = render_path.read_text(encoding="utf-8")
    assert "name: catfood-unsupervised-survey" in render_text
    assert "branch: master" in render_text
    assert "autoDeployTrigger: commit" in render_text
    assert "python scripts/build_deployment_artifacts.py" in render_text
    assert "gunicorn wsgi:server --bind 0.0.0.0:$PORT" in render_text
    assert "PYTHON_VERSION" in render_text

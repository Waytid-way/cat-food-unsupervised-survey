# Repository Compatibility Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compatibility layer that preserves every current import path and CLI entrypoint while preparing the repo for the `GOAL.md` folder layout, so the later file move can happen without breaking the pipelines or dashboard.

**Architecture:** Keep `src/catfood_unsupervised` as the canonical implementation for this phase. Add a small `compat` package that re-exports the current unsupervised, supervised, and dashboard APIs under a migration-friendly surface, then switch scripts and module entrypoints to use that surface. Lock the current behavior with tests before changing the import plumbing so we can prove the compatibility layer did not alter outputs or startup behavior.

**Tech Stack:** Python, pytest, importlib, pathlib, sys.

---

## File Structure

**Create:**
- `src/catfood_unsupervised/compat/__init__.py`
- `src/catfood_unsupervised/compat/bootstrap.py`
- `src/catfood_unsupervised/compat/unsupervised.py`
- `src/catfood_unsupervised/compat/supervised.py`
- `src/catfood_unsupervised/compat/dashboard.py`
- `tests/compat/test_import_surface.py`
- `tests/compat/test_entrypoints.py`
- `README.md`

**Modify:**
- `scripts/run_pipeline.py`
- `scripts/run_supervised_pipeline.py`
- `scripts/predict_supervised_segment.py`
- `src/catfood_unsupervised/dashboard/__main__.py`
- `docs/supervised-team-handoff.md`

---

### Task 1: Freeze the current public surface with regression tests

**Files:**
- Create: `tests/compat/test_import_surface.py`
- Create: `tests/compat/test_entrypoints.py`

- [ ] **Step 1: Write the failing import-surface test**

```python
from importlib import import_module
import pytest


def test_canonical_modules_import_cleanly():
    pipeline = import_module("catfood_unsupervised.pipeline")
    supervised_pipeline = import_module("catfood_unsupervised.supervised.pipeline")
    dashboard_app = import_module("catfood_unsupervised.dashboard.app")

    assert callable(pipeline.run_pipeline)
    assert callable(supervised_pipeline.run_supervised_pipeline)
    assert hasattr(dashboard_app, "dash_app")


def test_compatibility_surface_not_present_yet():
    with pytest.raises(ModuleNotFoundError):
        import_module("catfood_unsupervised.compat.unsupervised")
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/compat/test_import_surface.py -q`

Expected: FAIL because the `catfood_unsupervised.compat` package does not exist yet.

- [ ] **Step 3: Write the current-state CLI smoke test**

```python
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_unsupervised_cli_help():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "Run the cat food unsupervised learning pipeline" in result.stdout


def test_supervised_cli_help():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_supervised_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

        assert result.returncode == 0
        assert "Run the cat food supervised learning pipeline" in result.stdout
```

- [ ] **Step 4: Run the focused test and confirm the current launchers still work**

Run: `python -m pytest tests/compat/test_entrypoints.py -q`

Expected: PASS, establishing a baseline before any compatibility refactor.

- [ ] **Step 5: Commit the regression tests**

Create a commit after the import and CLI surface are frozen so the compatibility work can be reviewed against a stable baseline.

---

### Task 2: Add the compatibility package and forward the current APIs

**Files:**
- Create: `src/catfood_unsupervised/compat/__init__.py`
- Create: `src/catfood_unsupervised/compat/bootstrap.py`
- Create: `src/catfood_unsupervised/compat/unsupervised.py`
- Create: `src/catfood_unsupervised/compat/supervised.py`
- Create: `src/catfood_unsupervised/compat/dashboard.py`
- Modify: `src/catfood_unsupervised/__init__.py`

- [ ] **Step 1: Write the failing compatibility-module tests**

```python
from importlib import import_module


def test_compat_unsupervised_module_forwards_public_pipeline_api():
    compat_module = import_module("catfood_unsupervised.compat.unsupervised")

    assert callable(compat_module.run_pipeline)
    assert callable(compat_module.run_unsupervised_workflow)


def test_compat_supervised_module_forwards_public_pipeline_api():
    compat_module = import_module("catfood_unsupervised.compat.supervised")

    assert callable(compat_module.run_supervised_pipeline)
    assert hasattr(compat_module, "FEATURE_COLUMNS")


def test_compat_dashboard_module_forwards_public_dashboard_api():
    compat_module = import_module("catfood_unsupervised.compat.dashboard")

    assert hasattr(compat_module, "dash_app")
    assert callable(compat_module.render_tab_content)
    assert callable(compat_module.render_dashboard_shell)
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/compat/test_import_surface.py -q`

Expected: FAIL until the compatibility package exists.

- [ ] **Step 3: Implement the compatibility package**

Create a small `compat` package that re-exports the existing implementation modules without moving the implementation yet.

```python
# src/catfood_unsupervised/compat/bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir
```

```python
# src/catfood_unsupervised/compat/unsupervised.py
from catfood_unsupervised.config import *  # noqa: F401,F403
from catfood_unsupervised.data_loading import *  # noqa: F401,F403
from catfood_unsupervised.features import *  # noqa: F401,F403
from catfood_unsupervised.models import *  # noqa: F401,F403
from catfood_unsupervised.pipeline import *  # noqa: F401,F403
from catfood_unsupervised.preprocessing import *  # noqa: F401,F403
from catfood_unsupervised.reporting import *  # noqa: F401,F403
```

```python
# src/catfood_unsupervised/compat/supervised.py
from catfood_unsupervised.supervised.config import *  # noqa: F401,F403
from catfood_unsupervised.supervised.data_loading import *  # noqa: F401,F403
from catfood_unsupervised.supervised.features import *  # noqa: F401,F403
from catfood_unsupervised.supervised.history_store import *  # noqa: F401,F403
from catfood_unsupervised.supervised.models import *  # noqa: F401,F403
from catfood_unsupervised.supervised.pipeline import *  # noqa: F401,F403
from catfood_unsupervised.supervised.reporting import *  # noqa: F401,F403
from catfood_unsupervised.supervised.scoring import *  # noqa: F401,F403
from catfood_unsupervised.supervised.schema import *  # noqa: F401,F403
```

```python
# src/catfood_unsupervised/compat/dashboard.py
from catfood_unsupervised.dashboard.app import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.bootstrap import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.config import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.data_loader import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.supervised_callbacks import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.supervised_data_loader import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.kpi_banner import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.shell import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.supervised_form import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.supervised_results import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_business_insight import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_clustering import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_correlation import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_eda import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_persona import *  # noqa: F401,F403
from catfood_unsupervised.dashboard.components.tab_supervised import *  # noqa: F401,F403
```

Keep `src/catfood_unsupervised/__init__.py` as a lightweight façade that exposes the package name and, if needed, documents the compatibility boundary rather than moving implementation code in this phase.

- [ ] **Step 4: Re-run the compatibility tests**

Run:
`python -m pytest tests/compat/test_import_surface.py -q`

Expected: PASS once the compatibility package exists.

- [ ] **Step 5: Keep the module surface explicit**

Avoid inventing new APIs in the compatibility package. Only forward the public functions, constants, and entrypoints that already exist in the canonical modules so the next migration step stays boring.

---

### Task 3: Switch the launchers and docs to the compatibility surface

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/run_supervised_pipeline.py`
- Modify: `scripts/predict_supervised_segment.py`
- Modify: `src/catfood_unsupervised/dashboard/__main__.py`
- Modify: `README.md`
- Modify: `docs/supervised-team-handoff.md`

- [ ] **Step 1: Write the failing launcher test**

```python
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_module_imports_from_compat_surface():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from catfood_unsupervised.compat.dashboard import dash_app; print(dash_app.title)",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "Cat Food ML Dashboard" in result.stdout
```

- [ ] **Step 2: Run the launcher test and confirm it fails**

Run: `python -m pytest tests/compat/test_entrypoints.py -q`

Expected: FAIL until the scripts and dashboard module point at the compatibility layer.

- [ ] **Step 3: Update the launchers to import through `catfood_unsupervised.compat`**

Use the shared bootstrap helper for direct script execution, then import from the compat modules instead of the canonical implementation modules.

```python
from catfood_unsupervised.compat.bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.compat.unsupervised import OUTPUT_DIR, RAW_DATA_PATH, REPORT_DIR, run_unsupervised_workflow
```

```python
from catfood_unsupervised.compat.bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.compat.supervised import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR, RANDOM_STATE, run_supervised_pipeline
```

```python
from catfood_unsupervised.compat.bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.compat.dashboard import dash_app as app
```

Update `README.md` and `docs/supervised-team-handoff.md` so they explain that `src/catfood_unsupervised` is still the canonical implementation for now, while `catfood_unsupervised.compat.*` is the migration-safe surface the next restructure step will use.

- [ ] **Step 4: Re-run the compatibility tests and a focused repo slice**

Run:
`python -m pytest tests/compat tests/dashboard/test_shell_layout.py tests/dashboard/test_tab_supervised.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the compatibility layer**

After the tests pass, commit the compatibility package, launcher updates, and docs together so the migration boundary is easy to review.

---

## Self-Review Checklist

**1. Spec coverage:**
- Current import surface frozen → Task 1
- Compatibility package and alias modules → Task 2
- Launcher / dashboard bootstrap and docs → Task 3

**2. Placeholder scan:** No TBD/TODO placeholders. Every step has concrete files, code, and verification commands.

**3. Type consistency:** The compatibility package forwards existing canonical names only. No new public API is introduced until the next migration phase.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-repo-compatibility-layer.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

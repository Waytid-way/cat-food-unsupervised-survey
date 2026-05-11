# Dashboard Shell Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current plain Dash wrapper with a polished application shell that matches the provided HTML layout, while preserving the existing tab order and routing for the unsupervised, supervised, and insight pages.

**Architecture:** Keep the existing Dash callbacks and tab content as the source of truth, but wrap them in a dedicated shell component that owns the sidebar, page header, and scrollable content region. Centralize colors, spacing, and card treatments in the dashboard styles so all existing tabs inherit the same visual language without rewriting their data flow.

**Tech Stack:** Python, Dash, dash-bootstrap-components fallback wrapper, Plotly, CSS.

---

## File Structure

**Create:**
- `src/catfood_unsupervised/dashboard/components/shell.py`
- `tests/dashboard/test_shell_layout.py`

**Modify:**
- `src/catfood_unsupervised/dashboard/app.py`
- `src/catfood_unsupervised/dashboard/config.py`
- `src/catfood_unsupervised/dashboard/styles/custom.css`
- `src/catfood_unsupervised/dashboard/components/tab_eda.py`
- `src/catfood_unsupervised/dashboard/components/tab_correlation.py`
- `src/catfood_unsupervised/dashboard/components/tab_clustering.py`
- `src/catfood_unsupervised/dashboard/components/tab_persona.py`
- `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`
- `src/catfood_unsupervised/dashboard/components/kpi_banner.py`
- `docs/supervised-team-handoff.md`

---

### Task 1: Extract the app shell into a dedicated component

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/shell.py`
- Modify: `src/catfood_unsupervised/dashboard/app.py`
- Test: `tests/dashboard/test_shell_layout.py`

- [ ] **Step 1: Write the failing shell-layout test**

```python
from catfood_unsupervised.dashboard.components.shell import render_dashboard_shell


def test_dashboard_shell_contains_sidebar_header_and_content_slot():
    shell = render_dashboard_shell(active_tab="tab_eda", content=html.Div("body"))
    rendered = " ".join(_collect_text(shell))

    assert "CatFood ML" in rendered
    assert "Home Dashboard" in rendered
    assert "Business Insight" in rendered
    assert "body" in rendered
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: FAIL because the shell component does not exist yet.

- [ ] **Step 3: Implement the shell component**

Create a component that owns:
- the left sidebar with the existing tab order
- the page header with the survey title and action buttons
- a scrollable content region that accepts the active tab content

Use the existing `TAB_ITEMS` config as the single source of truth for nav labels and tab ids so the shell and callbacks do not drift apart.

```python
from __future__ import annotations

from dash import dcc, html

from catfood_unsupervised.dashboard.config import TAB_ITEMS


def render_dashboard_shell(active_tab: str, content):
    nav_buttons = []
    for item in TAB_ITEMS:
        is_active = item["value"] == active_tab
        nav_buttons.append(
            html.Button(
                [
                    html.I(className=item["icon"]),
                    html.Span(item["label"]),
                ],
                id=f"nav-{item['value'].removeprefix('tab_')}",
                className=("nav-item active" if is_active else "nav-item text-textMuted"),
                n_clicks=0,
                **{"data-tab": item["value"].removeprefix("tab_")},
            )
        )

    return html.Div(
        [
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("AI", className="shell-logo"),
                            html.H1("CatFood ML", className="shell-brand"),
                        ],
                        className="shell-brand-row",
                    ),
                    html.Nav(nav_buttons, className="shell-nav"),
                    html.Div(
                        html.Button(
                            [html.I(className="ph ph-sign-out"), html.Span("Log out")],
                            className="shell-logout",
                        ),
                        className="shell-footer",
                    ),
                ],
                className="shell-sidebar",
            ),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.H2("Welcome back, Team!"),
                                    html.P("Cat Food Packaging Survey Analysis (n=148)"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Button(html.I(className="ph ph-calendar-blank")),
                                    html.Button(
                                        [html.I(className="ph ph-plus"), html.Span("Add new widget")],
                                        className="shell-primary-button",
                                    ),
                                ],
                                className="shell-actions",
                            ),
                        ],
                        className="shell-header",
                    ),
                    html.Div(content, className="shell-content"),
                ],
                className="shell-main",
            ),
        ],
        className="shell-root",
    )
```

- [ ] **Step 4: Re-run the test and confirm it passes**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: PASS.

- [ ] **Step 5: Point `app.py` at the shell**

Replace the current top-level `html.H1` + `dbc.Container` wrapper with the shell component, and keep the tab selection callback focused only on returning tab content.

---

### Task 2: Make tab metadata drive the navigation

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/config.py`
- Modify: `src/catfood_unsupervised/dashboard/app.py`
- Modify: `src/catfood_unsupervised/dashboard/components/shell.py`
- Test: `tests/dashboard/test_shell_layout.py`

- [ ] **Step 1: Write a metadata test for the tab registry**

```python
from catfood_unsupervised.dashboard.config import TAB_ITEMS


def test_tab_items_provide_shell_navigation_metadata():
    assert [item["value"] for item in TAB_ITEMS] == [
        "tab_eda",
        "tab_correlation",
        "tab_clustering",
        "tab_persona",
        "tab_supervised",
        "tab_business_insight",
    ]
    assert all("label" in item for item in TAB_ITEMS)
    assert all("icon" in item for item in TAB_ITEMS)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: FAIL until the registry includes the shell-facing metadata.

- [ ] **Step 3: Extend `TAB_ITEMS` with shell metadata**

Update each tab entry to include:
- `icon`
- `description`
- `section`

Keep the existing `value` identifiers untouched so the app callback and any existing tabs continue to work.

```python
TAB_ITEMS = [
    {"label": "1. EDA & Stats", "value": "tab_eda", "icon": "ph ph-squares-four", "description": "Survey overview", "section": "home"},
    {"label": "2. Correlation", "value": "tab_correlation", "icon": "ph ph-chart-scatter", "description": "Feature relationships", "section": "home"},
    {"label": "3. Clustering", "value": "tab_clustering", "icon": "ph ph-chart-line-up", "description": "Unsupervised learning", "section": "analysis"},
    {"label": "4. Persona", "value": "tab_persona", "icon": "ph ph-users", "description": "Segment profile", "section": "analysis"},
    {"label": "5. Supervised", "value": "tab_supervised", "icon": "ph ph-robot", "description": "Prediction workflow", "section": "modeling"},
    {"label": "6. Business Insight", "value": "tab_business_insight", "icon": "ph ph-briefcase", "description": "Decision summary", "section": "insight"},
]
```

- [ ] **Step 4: Re-run the metadata test**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: PASS.

- [ ] **Step 5: Wire the shell to the existing callback**

Have `app.py` choose the active tab from the current `dcc.Tabs` value and pass the matching rendered tab body into `render_dashboard_shell()`. Keep the callback output unchanged so the existing content factories remain reusable.

---

### Task 3: Restyle the dashboard to match the supplied HTML

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/styles/custom.css`
- Modify: `src/catfood_unsupervised/dashboard/components/kpi_banner.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_eda.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_correlation.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_clustering.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_persona.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`

- [ ] **Step 1: Add visual regression assertions for the new shell classes**

```python
from catfood_unsupervised.dashboard.components.shell import render_dashboard_shell


def test_shell_uses_expected_class_names():
    shell = render_dashboard_shell(active_tab="tab_eda", content=html.Div())
    class_names = " ".join(sorted(_collect_class_names(shell)))

    assert "shell-root" in class_names
    assert "shell-sidebar" in class_names
    assert "shell-main" in class_names
    assert "shell-content" in class_names
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: FAIL until the CSS-oriented class structure exists.

- [ ] **Step 3: Update `custom.css` to match the reference styling**

Add styles for:
- the fixed sidebar
- pill-shaped active nav items
- soft-shadow KPI cards
- a warm off-white background
- a compact, scrollable content surface
- responsive stacking for narrow screens

Reuse the same accent palette already present in the dashboard so the look changes without making the color system drift.

- [ ] **Step 4: Standardize tab-level spacing and card classes**

Update the existing tab renderers so they emit shell-friendly class names such as:
- `dashboard-card`
- `dashboard-section-title`
- `dashboard-grid`
- `dashboard-muted-text`

Do not rewrite the charts or business logic. The intent is to make the old tab content sit naturally inside the new shell.

- [ ] **Step 5: Re-run the shell test and the tab import smoke tests**

Run:
`python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: PASS.

Run:
`python -c "from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab; from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab; print('OK')"`

Expected: `OK`

---

### Task 4: Refresh the dashboard entrypoint and handoff notes

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/app.py`
- Modify: `docs/supervised-team-handoff.md`

- [ ] **Step 1: Write the failing app bootstrap test**

```python
from catfood_unsupervised.dashboard.app import dash_app


def test_dashboard_app_uses_shell_layout():
    layout_text = " ".join(_collect_text(dash_app.layout))
    assert "CatFood ML" in layout_text
    assert "Welcome back, Team!" in layout_text
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: FAIL until the top-level layout is switched from the old wrapper to the shell.

- [ ] **Step 3: Update the app layout to be shell-first**

Keep the callback that renders tab content, but let the top-level `dash_app.layout` become the shell. If the page needs an initial active tab, default to `tab_eda` so first load feels complete instead of blank.

```python
dash_app.layout = render_dashboard_shell(
    active_tab="tab_eda",
    content=html.Div(id="tab_content"),
)
```

- [ ] **Step 4: Update the handoff doc to match the new experience**

Document:
- the main dashboard entrypoint
- the tab order as it appears in the shell
- the purpose of each tab from a business-user perspective
- where to look first when validating the UI locally

- [ ] **Step 5: Re-run the app bootstrap test**

Run: `python -m pytest tests/dashboard/test_shell_layout.py -q`

Expected: PASS.

---

## Self-Review Checklist

**1. Spec coverage:**
- Shell layout replacement → Task 1
- Nav metadata and tab order preservation → Task 2
- CSS and card restyling → Task 3
- App entrypoint and handoff docs → Task 4

**2. Placeholder scan:** No TBD/TODO placeholders. All steps include concrete code or exact verification commands.

**3. Type consistency:** `tab_*` identifiers remain unchanged, and the shell uses `TAB_ITEMS` as the single navigation registry so the callback and sidebar stay aligned.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-dashboard-shell-adaptation.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

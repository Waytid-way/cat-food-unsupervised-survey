# Business Insight Compact Boardroom Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the `/business` page into a compact executive summary that surfaces the segment story, the key business signals, and the recommended next actions at a glance.

**Architecture:** Keep the current Dash page and data loader, but replace the current persona-card + wide chart layout with a compact hero, KPI row, two-column boardroom snapshot, and a short recommendation panel. Derive all copy and figures from the existing `outputs/unsupervised` artifacts so there is no pipeline change.

**Tech Stack:** Python, Dash, Dash Bootstrap Components, Plotly, pandas, pytest.

---

### Task 1: Add failing tests for the new executive summary layout

**Files:**
- Modify: `tests/dashboard/test_insight_page.py`

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

import pandas as pd

from catfood_unsupervised.dashboard.pages import insight as insight_page


def test_business_insight_renders_compact_snapshot(monkeypatch):
    clean_df = pd.DataFrame(
        {
            "segment": [1, 1, 2, 2],
            "เพศของคุณ": ["ชาย", "หญิง", "หญิง", "หญิง"],
            "อายุของคุณ": ["20-29ปี", "20-29ปี", "30-39ปี", "30-39ปี"],
            "สถานภาพสมรส": ["โสด ไม่มีแฟน", "โสด ไม่มีแฟน", "แต่งงานแล้ว", "แต่งงานแล้ว"],
        }
    )
    segment_profiles = pd.DataFrame(
        {
            "segment_size": [36, 112],
            "vote_01": [1.0, 0.7],
            "vote_02": [1.1, 0.8],
            "vote_03": [2.0, 1.6],
            "buy_factor_01": [4.3, 4.2],
            "buy_factor_02": [3.6, 3.1],
            "buy_factor_03": [4.6, 4.4],
            "buy_factor_04": [3.5, 3.0],
            "buy_factor_05": [4.5, 3.9],
            "packaging_importance_01": [4.1, 3.7],
            "packaging_importance_02": [4.3, 3.5],
            "packaging_importance_03": [4.4, 4.1],
        },
        index=[1, 2],
    )
    metrics = {
        "row_counts": {"completed_top3": 148},
        "final_cluster_k": 2,
        "anomaly_detection": {"anomaly_rate": 0.1756},
        "segment_sizes": {"1": 36, "2": 112},
    }

    monkeypatch.setattr(
        insight_page,
        "load_all_data",
        lambda output_dir: SimpleNamespace(
            metrics=metrics,
            clean_df=clean_df,
            correlation_matrix=pd.DataFrame([[1.0]], columns=["x"], index=["x"]),
            segment_profiles=segment_profiles,
            pca_scores=pd.DataFrame({"PC1": [1.0], "PC2": [0.5]}),
        ),
    )

    rendered = insight_page._render()
    texts = " ".join(_collect_text(rendered))
    graphs = _collect_graphs(rendered)

    assert "Compact Boardroom Snapshot" in texts
    assert "Persona Cards" not in texts
    assert "What to do next" in texts
    assert any(getattr(graph.figure.layout.title, "text", "") == "Segment Mix" for graph in graphs)
    assert any(getattr(graph.figure.data[0], "type", "") == "pie" for graph in graphs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/dashboard/test_insight_page.py -q
```

Expected: fail because the current page still renders the old persona-card layout and grouped bar chart.

- [ ] **Step 3: Keep the helper functions in the test file**

```python
def _collect_graphs(component, results=None):
    if results is None:
        results = []
    graph_type = type(insight_page.dcc.Graph(figure={})).__name__
    if component.__class__.__name__ == graph_type:
        results.append(component)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _collect_graphs(child, results)
    elif children is not None:
        _collect_graphs(children, results)
    return results
```

### Task 2: Rebuild the page layout and business summary logic

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/pages/insight.py`

- [ ] **Step 1: Add view-model helpers**

Implement helpers that derive a boardroom-friendly view model from `DashboardData`:

```python
def _build_business_summary(data) -> dict[str, object]:
    # derive:
    # - headline
    # - supporting sentence
    # - KPI values
    # - segment cards
    # - recommendation bullets
    # - donut chart figure
    return {...}
```

The helper should use:
- `metrics["row_counts"]["completed_top3"]`
- `metrics["final_cluster_k"]`
- `metrics["anomaly_detection"]["anomaly_rate"]`
- `metrics["segment_sizes"]`
- `segment_profiles`
- `BUY_FACTOR_LABELS`
- `PACKAGING_IMPORTANCE_LABELS`

- [ ] **Step 2: Replace the current chart block**

Swap the existing grouped bar chart for a compact `Segment Mix` donut chart:

```python
segment_mix_fig = px.pie(
    segment_mix_df,
    names="Segment",
    values="Count",
    hole=0.58,
    title="Segment Mix",
    color="Segment",
    color_discrete_sequence=["#2E86AB", "#F18F01", "#3A7D44", "#A23B72"],
)
segment_mix_fig.update_traces(textinfo="percent+label")
```

- [ ] **Step 3: Render the compact executive layout**

Compose the page into:
- hero summary strip
- KPI row
- two-column body
- footer note with a link back to `/unsupervised`

The left column should show one compact card per segment with:
- segment ID
- share/count
- top voted option
- top buy factor
- top packaging importance
- one-sentence summary

The right column should show:
- `Segment Mix`
- `What to do next`

- [ ] **Step 4: Keep the empty state simple**

If `load_all_data` fails, continue returning the same friendly empty state text, but do not leak raw exceptions.

- [ ] **Step 5: Run the page test**

Run:

```bash
python -m pytest tests/dashboard/test_insight_page.py -q
```

Expected: pass after the layout is rebuilt.

### Task 3: Add styling for the compact boardroom layout

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/styles/custom.css`

- [ ] **Step 1: Add page-scoped classes**

Add styling for:
- `.business-insight-shell`
- `.business-insight-hero`
- `.business-insight-hero__headline`
- `.business-insight-hero__lede`
- `.business-insight-kpi-row`
- `.business-insight-kpi-card`
- `.business-insight-segment-card`
- `.business-insight-recommendation`
- `.business-insight-chart-card`

Use the existing dashboard palette and shadows so the page matches the rest of the app.

- [ ] **Step 2: Make the layout responsive**

Add mobile rules so:
- the KPI row stacks cleanly
- the two-column body collapses to one column
- the hero text remains readable without overflow

- [ ] **Step 3: Keep the page visually lighter than the current view**

Use tighter vertical spacing than the old persona-card layout so the first screen fits the executive-summary goal.

### Task 4: Verify the page in tests and browser

**Files:**
- Modify: `tests/dashboard/test_insight_page.py`

- [ ] **Step 1: Run the dashboard test subset**

Run:

```bash
python -m pytest tests/dashboard tests/supervised tests/scripts -q
```

Expected: all tests pass.

- [ ] **Step 2: Open `/business` in the in-app browser**

Verify visually that:
- the hero is the first visible block
- the KPI cards appear before the charts
- the page shows the new `Segment Mix` donut
- the old `Persona Cards` title and wide grouped bar chart are gone

- [ ] **Step 3: Commit the implementation**

```bash
git add src/catfood_unsupervised/dashboard/pages/insight.py src/catfood_unsupervised/dashboard/styles/custom.css tests/dashboard/test_insight_page.py
git commit -m "feat: redesign business insight page"
```

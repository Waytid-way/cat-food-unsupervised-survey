# Enhanced Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new visualizations to the Dash dashboard: Top-3 Ranking Breakdown, Option Rating Distribution, K-Means Evaluation Table, and Segment Profile Deep-Dive — with structured redesign and CSS enhancements.

**Architecture:** Layered approach: new `styles/` and `shared.py` provide foundation; modify tab components in place; no breaking changes to existing layout or data flow.

**Tech Stack:** Dash + Plotly (Python), dash-bootstrap-components, CSS custom styling.

---

## File Structure

```
src/catfood_unsupervised/dashboard/
├── app.py                       # Modify: add CSS import
├── config.py                    # Keep (PALETTE, TAB_ITEMS, KPI_CARDS)
├── data_loader.py               # Keep (DashboardData dataclass)
├── styles/
│   ├── custom.css               # Create: enhanced card/chart styling
│   └── theme.py                 # Create: LIGHT_THEME, DARK_THEME constants
└── components/
    ├── kpi_banner.py            # Keep
    ├── tab_eda.py               # Modify: add Top-3 + Option Rating
    ├── tab_correlation.py        # Keep
    ├── tab_clustering.py        # Modify: add K-Means Table
    ├── tab_persona.py           # Modify: add Segment Profile Deep-Dive
    └── shared.py                # Create: render_summary_stats, segment_color_map
```

---

## Tasks

### Task 1: Create shared helpers

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/shared.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_persona.py:88`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_eda.py:79`

- [ ] **Step 1: Create shared.py**

```python
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def render_summary_stats(metrics: dict) -> dbc.Row:
    raw = metrics.get("row_counts", {}).get("raw_loaded", 0)
    completed = metrics.get("row_counts", {}).get("completed_top3", 0)
    rate = (completed / raw * 100) if raw > 0 else 0

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Raw Responses", className="text-muted mb-1"),
                        html.H4(str(raw), className="mb-0"),
                    ]),
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Completed (Top-3)", className="text-muted mb-1"),
                        html.H4(str(completed), className="mb-0"),
                    ]),
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Completion Rate", className="text-muted mb-1"),
                        html.H4(f"{rate:.1f}%", className="mb-0 text-success"),
                    ]),
                ),
                width=3,
            ),
        ],
        className="mb-3",
    )


def segment_color_map() -> dict[int, str]:
    return {
        1: "#2E86AB",
        2: "#F18F01",
    }
```

- [ ] **Step 2: Verify file imports correctly**

Run: `python -c "from catfood_unsupervised.dashboard.components.shared import render_summary_stats, segment_color_map; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/shared.py
git commit -m "feat(dashboard): add shared helpers — render_summary_stats, segment_color_map"
```

---

### Task 2: Create styles directory

**Files:**
- Create: `src/catfood_unsupervised/dashboard/styles/theme.py`
- Create: `src/catfood_unsupervised/dashboard/styles/custom.css`

- [ ] **Step 1: Create theme.py**

```python
from __future__ import annotations

LIGHT_THEME = {
    "background": "#F8F9FA",
    "card_bg": "#FFFFFF",
    "text_primary": "#212529",
    "text_muted": "#6C757D",
    "border": "#DEE2E6",
    "chart_bg": "#FFFFFF",
    "grid_line": "#EEEEEE",
}

DARK_THEME = {
    "background": "#212529",
    "card_bg": "#2D2D2D",
    "text_primary": "#F8F9FA",
    "text_muted": "#ADB5BD",
    "border": "#495057",
    "chart_bg": "#2D2D2D",
    "grid_line": "#495057",
}
```

- [ ] **Step 2: Create custom.css**

```css
/* Enhanced card styling */
.card {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: none;
    transition: box-shadow 0.2s ease;
}

.card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

/* Better plot backgrounds */
.js-plotly-plot .plotly .modebar {
    background: transparent;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2E86AB;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2E86AB;
}

/* Segment accent borders */
.segment-1-accent {
    border-left: 4px solid #2E86AB !important;
}

.segment-2-accent {
    border-left: 4px solid #F18F01 !important;
}

/* Better table styling */
.dash-table-container .dash-table {
    font-size: 0.9rem;
}

/* Metric highlight */
.metric-highlight {
    font-weight: 700;
    color: #2E86AB;
}

/* Tab content padding */
.tab-content {
    padding-top: 16px;
}
```

- [ ] **Step 3: Verify files exist**

Run: `ls src/catfood_unsupervised/dashboard/styles/`
Expected: `custom.css` and `theme.py`

- [ ] **Step 4: Commit**

```bash
git add src/catfood_unsupervised/dashboard/styles/theme.py src/catfood_unsupervised/dashboard/styles/custom.css
git commit -m "feat(dashboard): add styles — theme constants and custom CSS"
```

---

### Task 3: Modify tab_eda.py — add Top-3 Breakdown + Option Rating

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/components/tab_eda.py:79-100`

- [ ] **Step 1: Read existing file to understand current structure**

Read: `src/catfood_unsupervised/dashboard/components/tab_eda.py`

- [ ] **Step 2: Add imports and new render functions after line 78**

After existing imports, add:

```python
import plotly.graph_objects as go


def _render_top3_breakdown(clean_df: pd.DataFrame) -> dcc.Graph:
    option_cols = [clean_df.columns[i] for i in range(79, 89)]
    options = [f"Opt {i+1:02d}" for i in range(10)]

    rank1_counts = clean_df[clean_df.columns[79]].astype(str).str.strip().value_counts()
    rank2_counts = clean_df[clean_df.columns[80]].astype(str).str.strip().value_counts()
    rank3_counts = clean_df[clean_df.columns[81]].astype(str).str.strip().value_counts()

    fig = go.Figure()
    for rank_idx, (col_idx, label, color) in enumerate([
        (79, "Rank 1 (3 pts)", "#2E86AB"),
        (80, "Rank 2 (2 pts)", "#A23B72"),
        (81, "Rank 3 (1 pt)", "#F18F01"),
    ]):
        col = clean_df.columns[col_idx]
        vc = clean_df[col].astype(str).str.strip().value_counts()
        values = [vc.get(str(i), 0) for i in range(1, 11)]
        fig.add_trace(go.Bar(
            name=label,
            x=options,
            y=values,
            marker_color=color,
        ))

    fig.update_layout(
        title="Top-3 Ranking Breakdown by Option",
        barmode="group",
        height=350,
        plot_bgcolor="white",
        showlegend=True,
    )
    return dcc.Graph(figure=fig)


def _render_option_rating_heatmap(clean_df: pd.DataFrame) -> dcc.Graph:
    opt_cols = [c for c in clean_df.columns if c.startswith("option_") and "_attribute_" in c]
    opt_means = clean_df[opt_cols].mean()

    matrix = []
    for opt in range(1, 11):
        row = [opt_means.get(f"option_{opt:02d}_attribute_{attr:02d}", 0) for attr in range(1, 6)]
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"],
        y=[f"Opt {i:02d}" for i in range(1, 11)],
        colorscale="RdYlGn",
        zmin=1, zmax=5,
        colorbar_title="Rating",
    ))
    fig.update_layout(
        title="Option Rating Distribution (Mean Agreement 1-5)",
        height=400,
        plot_bgcolor="white",
    )
    return dcc.Graph(figure=fig)
```

- [ ] **Step 3: Append new sections to the return html.Div of render_eda_tab**

Find the closing `html.Div(id="tab_eda")` and add before the closing bracket:

```python
            html.H5("Top-3 Ranking Breakdown", className="section-header mt-4"),
            dbc.Row(dbc.Col(dcc.Graph(figure=_render_top3_breakdown(clean_df)), width=12)),
            html.H5("Option Rating Distribution", className="section-header mt-4"),
            dbc.Row(dbc.Col(dcc.Graph(figure=_render_option_rating_heatmap(clean_df)), width=12)),
```

- [ ] **Step 4: Verify the tab renders**

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_eda.py
git commit -m "feat(dashboard): tab_eda — add Top-3 breakdown and Option Rating heatmap"
```

---

### Task 4: Modify tab_clustering.py — add K-Means Evaluation Table

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/components/tab_clustering.py:1-141`

- [ ] **Step 1: Read existing file**

Read: `src/catfood_unsupervised/dashboard/components/tab_clustering.py`

- [ ] **Step 2: Add dash table import and render function after line 12**

After imports, add:

```python
from dash import dash_table as dt


def _render_kmeans_table(metrics: dict) -> dt.DataTable:
    ev = metrics.get("kmeans_evaluation", [])
    rows = []
    best_silhouette = -1
    best_db = float("inf")
    best_inertia = float("inf")

    for row in ev:
        k = row.get("k", 0)
        sil = row.get("silhouette_score", 0)
        db = row.get("davies_bouldin_score", 0)
        inert = row.get("inertia", 0)
        rows.append({"k": k, "Silhouette Score": round(sil, 3), "Davies-Bouldin Index": round(db, 3), "Inertia": round(inert, 1)})
        if sil > best_silhouette:
            best_silhouette = sil
        if db < best_db:
            best_db = db
        if inert < best_inertia:
            best_inertia = inert

    for row in rows:
        sil = row["Silhouette Score"]
        row["Silhouette Score"] = str(sil) + (" ★" if sil == best_silhouette else "")
        db = row["Davies-Bouldin Index"]
        row["Davies-Bouldin Index"] = str(db) + (" ★" if db == best_db else "")
        inert = row["Inertia"]
        row["Inertia"] = str(inert) + (" ★" if inert == best_inertia else "")

    return dt.DataTable(
        columns=[{"name": c, "id": c} for c in ["k", "Silhouette Score", "Davies-Bouldin Index", "Inertia"]],
        data=rows,
        style_table={"width": "60%"},
        style_cell={"textAlign": "center"},
        style_header={"background": "#F8F9FA", "fontWeight": "bold"},
    )
```

- [ ] **Step 3: Add K-Means Table section to render_clustering_tab return**

Find `dbc.Row(dbc.Col(dcc.Graph(figure=dendrogram_fig), width=12))` and add after:

```python
            html.H5("K-Means Evaluation Table", className="section-header mt-4"),
            dbc.Row(dbc.Col(_render_kmeans_table(metrics), width=12)),
```

- [ ] **Step 4: Verify imports**

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_clustering import render_clustering_tab; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_clustering.py
git commit -m "feat(dashboard): tab_clustering — add K-Means Evaluation Table"
```

---

### Task 5: Modify tab_persona.py — add Segment Profile Deep-Dive

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/components/tab_persona.py:57-89`

- [ ] **Step 1: Read existing file**

Read: `src/catfood_unsupervised/dashboard/components/tab_persona.py`

- [ ] **Step 2: Replace persona cards section with enhanced segment profiles**

After the line:
```python
        persona_cards.append(dbc.Col(card, width=6))
```

And before:
```python
    anomaly_banner = dbc.Alert(...)
```

Replace the loop and persona_cards with:

```python
    from catfood_unsupervised.dashboard.components.shared import segment_color_map

    seg_colors = segment_color_map()

    for segment_id in [1, 2]:
        seg_df = clean_df[clean_df["segment"] == segment_id]
        seg_size = len(seg_df)
        seg_pct = seg_size / len(clean_df) * 100
        color = seg_colors.get(segment_id, "#2E86AB")

        gender_col = clean_df.columns[74]
        age_col = clean_df.columns[73]
        marital_col = clean_df.columns[75]

        gender_vc = seg_df[gender_col].astype(str).str.strip().value_counts()
        age_vc = seg_df[age_col].astype(str).str.strip().value_counts()
        marital_vc = seg_df[marital_col].astype(str).str.strip().value_counts()

        gender_fig = go.Figure(data=[go.Pie(
            labels=list(gender_vc.index),
            values=list(gender_vc.values),
            marker_colors=["#2E86AB", "#A23B72", "#F18F01"][:len(gender_vc)],
        )])
        gender_fig.update_layout(title="Gender", height=200, margin=dict(t=30, b=30))

        age_fig = go.Figure(data=[go.Pie(
            labels=[str(x) for x in age_vc.index],
            values=list(age_vc.values),
        )])
        age_fig.update_layout(title="Age", height=200, margin=dict(t=30, b=30))

        marital_fig = go.Figure(data=[go.Pie(
            labels=list(marital_vc.index),
            values=list(marital_vc.values),
        )])
        marital_fig.update_layout(title="Marital", height=200, margin=dict(t=30, b=30))

        buy_labels = ["Buy Factor 1", "Buy Factor 2", "Buy Factor 3", "Buy Factor 4", "Buy Factor 5"]
        buy_means = [seg_df[clean_df.columns[5+i]].astype(float).mean() for i in range(5)]

        buy_fig = go.Figure(data=[go.Bar(
            x=buy_labels,
            y=buy_means,
            marker_color=color,
        )])
        buy_fig.update_layout(title="Buy Factor Preferences (Mean)", height=220, plot_bgcolor="white")

        strategy_info = SEGMENT_STRATEGIES.get(segment_id, {})

        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"Segment {segment_id}: {strategy_info.get('title', '')}", style={"color": color}),
                html.P(strategy_info.get("description", "")),
                html.Hr(),
                html.P(f"Size: {seg_size} ({seg_pct:.1f}%)", className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=gender_fig), width=4),
                    dbc.Col(dcc.Graph(figure=age_fig), width=4),
                    dbc.Col(dcc.Graph(figure=marital_fig), width=4),
                ]),
                dbc.Row(dbc.Col(dcc.Graph(figure=buy_fig), width=12)),
                html.Hr(),
                html.Strong("Strategy: "),
                html.P(strategy_info.get("strategy", ""), className="text-muted"),
            ]),
            className=f"shadow-sm mb-3 segment-{segment_id}-accent",
            style={"border-left": f"4px solid {color}"},
        )
        persona_cards.append(dbc.Col(card, width=6))
```

- [ ] **Step 3: Verify imports**

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_persona.py
git commit -m "feat(dashboard): tab_persona — add Segment Profile Deep-Dive with pie charts and buy factor bars"
```

---

### Task 6: Modify app.py — add CSS import

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/app.py:20-25`

- [ ] **Step 1: Read existing file**

Read: `src/catfood_unsupervised/dashboard/app.py`

- [ ] **Step 2: Add CSS import to external_stylesheets**

Change:
```python
    external_stylesheets=[dbc.themes.BOOTSTRAP],
```
To:
```python
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "/assets/custom.css",
    ],
```

- [ ] **Step 3: Add CSS serving route before dash_app initialization**

Before `dash_app = dash.Dash(...)`, add:

```python
@dash_app.server.route("/assets/<path:path>")
def serve_static(path):
    styles_dir = Path(__file__).parent / "styles"
    return send_from_directory(styles_dir, path)
```

Add import at top:
```python
from pathlib import Path
from flask import send_from_directory
```

- [ ] **Step 4: Verify app starts**

Run: `python -c "from catfood_unsupervised.dashboard.app import dash_app; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/app.py
git commit -m "feat(dashboard): app — serve custom.css from styles directory"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Top-3 Ranking Breakdown → Task 3
- ✅ Option Rating Distribution → Task 3
- ✅ K-Means Evaluation Table → Task 4
- ✅ Segment Profile Deep-Dive → Task 5
- ✅ CSS enhancements → Task 2 + Task 6
- ✅ Summary stats → Task 1

**2. Placeholder scan:** No TBD/TODO found. All steps have complete code.

**3. Type consistency:** All function names and signatures are unique and well-defined.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-05-10-dashboard-enhancement-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
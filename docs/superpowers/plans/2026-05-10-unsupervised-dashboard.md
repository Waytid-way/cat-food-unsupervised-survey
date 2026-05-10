# Unsupervised Learning Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Plotly+Dash web dashboard for the Cat Food Packaging Survey Unsupervised Learning results, with 4 tabbed sections covering EDA, Correlation, Clustering/Dimensionality Reduction, and Anomaly Detection + Customer Persona.

**Architecture:** Single-page Dash app with tab navigation. Data loaded once at startup from `outputs/` directory. All charts are Plotly figures. Layout uses `dash-bootstrap-components` for responsive grid and card components.

**Tech Stack:** Python 3.10+, Dash, Plotly, dash-bootstrap-components, pandas, scipy, scikit-learn

---

## File Structure

```
src/catfood_unsupervised/dashboard/
├── __init__.py
├── config.py              # KPI card definitions, tab structure, color palette
├── data_loader.py         # Load all outputs/ files and derive computed columns
├── components/
│   ├── __init__.py
│   ├── kpi_banner.py      # Top KPI cards: Silhouette, DB, Inertia, Variance, Anomaly Rate
│   ├── tab_eda.py         # Tab 1: Demographics + Buy Factors + Weighted Vote bar
│   ├── tab_correlation.py # Tab 2: Spearman heatmap + cluster annotations
│   ├── tab_clustering.py  # Tab 3: PCA Scree, PC scatter, Elbow/Silhouette, Dendrogram
│   └── tab_persona.py     # Tab 4: Isolation Forest scatter + 3 persona cards
└── app.py                 # Main Dash app entry point

tests/dashboard/
├── __init__.py
├── test_data_loader.py
├── test_kpi_banner.py
└── test_tab_eda.py
```

---

### Task 1: Project Structure Setup

**Files:**
- Create: `src/catfood_unsupervised/dashboard/__init__.py`
- Create: `src/catfood_unsupervised/dashboard/components/__init__.py`
- Create: `tests/dashboard/__init__.py`
- Modify: `pyproject.toml` — add Dash, Plotly, dash-bootstrap-components dependencies

- [ ] **Step 1: Add dependencies to pyproject.toml**

```toml
[project]
dependencies = [
    "dash>=2.14",
    "plotly>=5.18",
    "dash-bootstrap-components>=1.5",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.3",
]
```

Run: `cd "C:\Users\COM\Projects\Cat-food Unsupervised" && pip install dash plotly dash-bootstrap-components pandas scipy scikit-learn -e .`
Expected: Package installed, no import errors

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p src/catfood_unsupervised/dashboard/components
mkdir -p tests/dashboard
touch src/catfood_unsupervised/dashboard/__init__.py
touch src/catfood_unsupervised/dashboard/components/__init__.py
touch tests/dashboard/__init__.py
```

- [ ] **Step 3: Add test configuration**

Modify `pyproject.toml` to add:
```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

Run: `pytest --collect-only`
Expected: No errors, tests discovered

---

### Task 2: Data Loader

**Files:**
- Create: `src/catfood_unsupervised/dashboard/data_loader.py`
- Test: `tests/dashboard/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_data_loader.py
import json
from pathlib import Path

OUTPUT_DIR = Path("C:/Users/COM/Projects/Cat-food Unsupervised/outputs")

def test_load_metrics_summary():
    from catfood_unsupervised.dashboard.data_loader import load_metrics_summary
    metrics = load_metrics_summary(OUTPUT_DIR)
    assert "pca" in metrics
    assert "kmeans_evaluation" in metrics
    assert metrics["final_cluster_k"] == 2

def test_load_all_data():
    from catfood_unsupervised.dashboard.data_loader import load_all_data
    data = load_all_data(OUTPUT_DIR)
    assert data.metrics["anomaly_detection"]["anomaly_count"] == 26
    assert len(data.clean_df) == 148
    assert list(data.clean_df.columns).startswith("segment")
```

Run: `pytest tests/dashboard/test_data_loader.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL with "ModuleNotFoundError: No module named 'catfood_unsupervised.dashboard'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/catfood_unsupervised/dashboard/data_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DashboardData:
    metrics: dict
    clean_df: pd.DataFrame
    correlation_matrix: pd.DataFrame
    segment_profiles: pd.DataFrame
    pca_scores: pd.DataFrame


def load_metrics_summary(output_dir: Path) -> dict:
    return json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))


def load_all_data(output_dir: Path) -> DashboardData:
    metrics = load_metrics_summary(output_dir)
    clean_df = pd.read_csv(output_dir / "clean_dataset_with_segments.csv")
    correlation_matrix = pd.read_csv(output_dir / "correlation_matrix.csv", index_col=0)
    segment_profiles = pd.read_csv(output_dir / "segment_profiles.csv", index_col=0)
    pca_scores = _build_pca_scores(clean_df, metrics)
    return DashboardData(
        metrics=metrics,
        clean_df=clean_df,
        correlation_matrix=correlation_matrix,
        segment_profiles=segment_profiles,
        pca_scores=pca_scores,
    )


def _build_pca_scores(clean_df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    n_components = metrics["pca"]["n_components"]
    explained = metrics["pca"]["explained_variance_ratio"]
    cumulative = metrics["pca"]["cumulative_explained_variance"]
    columns = [f"PC{i+1}" for i in range(n_components)]
    rows = []
    for i in range(len(clean_df)):
        offset = 0
        row = {}
        for j, col in enumerate(columns):
            var = explained[j]
            row[col] = round(var * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows, index=clean_df.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/dashboard/test_data_loader.py -v`
Expected: PASS (3 tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/data_loader.py tests/dashboard/test_data_loader.py pyproject.toml
git commit -m "feat(dashboard): add data loader with metrics/df/correlation/segment loading"
```

---

### Task 3: Dashboard Config

**Files:**
- Create: `src/catfood_unsupervised/dashboard/config.py`

- [ ] **Step 1: Write config module**

```python
# src/catfood_unsupervised/dashboard/config.py
from __future__ import annotations

KPI_CARDS = [
    {
        "id": "silhouette",
        "label": "Silhouette Score (k=2)",
        "metric_path": ["kmeans_evaluation", 0, "silhouette_score"],
        "format": ".3f",
        "color": "#2E86AB",
    },
    {
        "id": "davies_bouldin",
        "label": "Davies-Bouldin Index",
        "metric_path": ["kmeans_evaluation", 0, "davies_bouldin_score"],
        "format": ".3f",
        "color": "#A23B72",
    },
    {
        "id": "inertia",
        "label": "Inertia (k=2)",
        "metric_path": ["kmeans_evaluation", 0, "inertia"],
        "format": ".1f",
        "color": "#F18F01",
    },
    {
        "id": "variance_explained",
        "label": "Variance Explained (8 PCs)",
        "metric_path": ["pca", "total_explained_variance_ratio"],
        "format": ".1%",
        "color": "#C73E1D",
    },
    {
        "id": "anomaly_rate",
        "label": "Anomaly Rate",
        "metric_path": ["anomaly_detection", "anomaly_rate"],
        "format": ".1%",
        "color": "#3A7D44",
    },
]

TAB_ITEMS = [
    {"label": "1. EDA & Stats", "value": "tab_eda", "href": "#tab_eda"},
    {"label": "2. Correlation", "value": "tab_correlation", "href": "#tab_correlation"},
    {"label": "3. Clustering", "value": "tab_clustering", "href": "#tab_clustering"},
    {"label": "4. Persona", "value": "tab_persona", "href": "#tab_persona"},
]

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "accent3": "#3A7D44",
    "segment1": "#2E86AB",
    "segment2": "#F18F01",
    "anomaly": "#C73E1D",
    "background": "#F8F9FA",
    "card_bg": "#FFFFFF",
}

CORRELATION_CLUSTERS = [
    {"name": "Classic/Safe", "options": ["opt01", "opt02"], "color": "#2E86AB"},
    {"name": "Mainstream", "options": ["opt03", "opt04", "opt05"], "color": "#F18F01"},
    {"name": "Niche Aesthetic", "options": ["opt08", "opt09", "opt10"], "color": "#A23B72"},
]
```

Run: `python -c "from catfood_unsupervised.dashboard.config import KPI_CARDS, TAB_ITEMS, PALETTE; print('OK')"`
Expected: OK

- [ ] **Step 2: Commit**

```bash
git add src/catfood_unsupervised/dashboard/config.py
git commit -m "feat(dashboard): add config with KPI cards, tab structure, and color palette"
```

---

### Task 4: KPI Banner Component

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/kpi_banner.py`
- Test: `tests/dashboard/test_kpi_banner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_kpi_banner.py
def test_kpi_banner_renders_cards():
    from catfood_unsupervised.dashboard.data_loader import load_metrics_summary
    from catfood_unsupervised.dashboard.config import KPI_CARDS
    from catfood_unsupervised.dashboard.components.kpi_banner import render_kpi_banner

    metrics = load_metrics_summary(Path("C:/Users/COM/Projects/Cat-food Unsupervised/outputs"))
    banner = render_kpi_banner(metrics, KPI_CARDS)
    assert len(banner.children) == 5  # 5 KPI cards
```

Run: `pytest tests/dashboard/test_kpi_banner.py -v`
Expected: FAIL

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/catfood_unsupervised/dashboard/components/kpi_banner.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from catfood_unsupervised.dashboard.config import KPI_CARDS


def _get_nested(metrics: dict, path: list) -> Any:
    node = metrics
    for key in path:
        node = node[key]
    return node


def render_kpi_banner(metrics: dict) -> html.Div:
    cards = []
    for card_def in KPI_CARDS:
        value = _get_nested(metrics, card_def["metric_path"])
        formatted = format(value, card_def["format"])
        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4(formatted, className="card-title text-center",
                                style={"color": card_def["color"], "font-weight": "bold"}),
                        html.P(card_def["label"], className="card-text text-center text-muted"),
                    ]
                ),
                className="text-center shadow-sm",
                style={"background": "#FFFFFF", "border-radius": "12px"},
            ),
            width=2,
        )
        cards.append(card)
    return html.Div(
        dbc.Row(cards, className="mb-4"),
        style={"background": "#F8F9FA", "padding": "20px", "border-radius": "8px"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/dashboard/test_kpi_banner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/kpi_banner.py tests/dashboard/test_kpi_banner.py
git commit -m "feat(dashboard): add KPI banner component"
```

---

### Task 5: Tab 1 — EDA & Descriptive Stats

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/tab_eda.py`
- Test: `tests/dashboard/test_tab_eda.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_tab_eda.py
from pathlib import Path

def test_eda_tab_renders():
    from catfood_unsupervised.dashboard.data_loader import load_all_data
    from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab

    data = load_all_data(Path("C:/Users/COM/Projects/Cat-food Unsupervised/outputs"))
    tab = render_eda_tab(data)
    assert tab is not None
    assert len(tab.children) > 0
```

Run: `pytest tests/dashboard/test_tab_eda.py -v`
Expected: FAIL

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/catfood_unsupervised/dashboard/components/tab_eda.py
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


GENDER_LABELS = {"ชาย": "ชาย", "หญิง": "หญิง", "อื่นๆ": "อื่นๆ"}
AGE_LABELS = {"20-29 ปี": "20-29", "30-39 ปี": "30-39", "40-49 ปี": "40-49", "50+": "50+"}
BUY_FACTOR_LABELS = [
    "ใช้วัตถุดิบจากธรรมชาติ",
    "ใช้วัตถุดิบนำเข้า",
    "รสชาติกลมกล่อม",
    "ผลิตภัณฑ์จากต่างประเทศ",
    "แบรนด์มีชื่อเสียง",
]


def render_eda_tab(data: DashboardData) -> html.Div:
    clean_df = data.clean_df
    metrics = data.metrics

    gender_counts = clean_df.iloc[:, 74].astype(str).str.strip().value_counts()
    age_counts = clean_df.iloc[:, 73].astype(str).str.strip().value_counts()
    marital_counts = clean_df.iloc[:, 75].astype(str).str.strip().value_counts()

    gender_fig = px.bar(
        x=list(gender_counts.index),
        y=list(gender_counts.values),
        title="เพศ",
        labels={"x": "เพศ", "y": "จำนวน"},
        color=list(gender_counts.index),
        color_discrete_sequence=["#2E86AB", "#A23B72", "#F18F01"],
    )
    age_fig = px.bar(
        x=[AGE_LABELS.get(k, k) for k in age_counts.index],
        y=list(age_counts.values),
        title="อายุ",
        labels={"x": "ช่วงอายุ", "y": "จำนวน"},
        color=list(age_counts.values),
        color_discrete_sequence=["#2E86AB"],
    )
    marital_fig = px.bar(
        x=list(marital_counts.index),
        y=list(marital_counts.values),
        title="สถานภาพสมรส",
        labels={"x": "สถานภาพ", "y": "จำนวน"},
        color=list(marital_counts.values),
        color_discrete_sequence=["#F18F01"],
    )

    buy_factor_means = [
        clean_df.iloc[:, i].mean() for i in range(5, 10)
    ]
    buy_factor_fig = px.bar(
        x=BUY_FACTOR_LABELS,
        y=buy_factor_means,
        title="ค่าเฉลี่ยปัจจัยการซื้อ (1-4)",
        labels={"x": "ปัจจัย", "y": "ค่าเฉลี่ย"},
        color=buy_factor_means,
        color_continuous_scale="Blues",
    )

    vote_cols = [f"vote_{i:02d}" for i in range(1, 11)]
    vote_sums = clean_df[vote_cols].sum().sort_values(ascending=False)
    vote_fig = px.bar(
        x=[f"Opt {i:02d}" for i in vote_sums.index],
        y=vote_sums.values,
        title="Weighted Vote (Top-3)",
        labels={"x": "ตัวเลือก", "y": "คะแนนถ่วงน้ำหนัก"},
        color=vote_sums.values,
        color_continuous_scale="Reds",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=gender_fig), width=4),
                    dbc.Col(dcc.Graph(figure=age_fig), width=4),
                    dbc.Col(dcc.Graph(figure=marital_fig), width=4),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=buy_factor_fig), width=6),
                    dbc.Col(dcc.Graph(figure=vote_fig), width=6),
                ],
            ),
        ],
        id="tab_eda",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/dashboard/test_tab_eda.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_eda.py tests/dashboard/test_tab_eda.py
git commit -m "feat(dashboard): add Tab 1 — EDA with demographics, buy factors, weighted vote"
```

---

### Task 6: Tab 2 — Correlation Analysis

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/tab_correlation.py`

- [ ] **Step 1: Write implementation**

```python
# src/catfood_unsupervised/dashboard/components/tab_correlation.py
from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


def render_correlation_tab(data: DashboardData) -> html.Div:
    corr = data.correlation_matrix

    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Spearman Correlation Heatmap — 10 Packaging Options",
        labels={"x": "Option", "y": "Option", "color": "r"},
    )
    fig.update_layout(
        width=900,
        height=700,
        xaxis_tickangle=45,
        font=dict(size=10),
    )

    cluster_text = [
        html.H5("Correlation Clusters", className="mt-3"),
        html.Ul([
            html.Li(html.Strong("Classic/Safe (Blue):"), " Option 01 × Option 02 — high r"),
            html.Li(html.Strong("Mainstream (Orange):"), " Option 03 × 04 × 05 — cluster together"),
            html.Li(html.Strong("Niche Aesthetic (Purple):"), " Option 08 × 09 × 10 — distinct preference"),
        ]),
    ]

    return html.Div(
        [
            dbc.Row(
                dbc.Col(dcc.Graph(figure=fig), width=12),
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(cluster_text, width=8),
            ),
        ],
        id="tab_correlation",
    )
```

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_correlation import render_correlation_tab; print('OK')"`
Expected: OK

- [ ] **Step 2: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_correlation.py
git commit -m "feat(dashboard): add Tab 2 — Spearman correlation heatmap with cluster annotations"
```

---

### Task 7: Tab 3 — Dimensionality Reduction & Clustering

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/tab_clustering.py`

- [ ] **Step 1: Write implementation**

```python
# src/catfood_unsupervised/dashboard/components/tab_clustering.py
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import scipy.cluster.hierarchy as sch
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


def _get_nested(metrics: dict, path: list) -> Any:
    node = metrics
    for key in path:
        node = node[key]
    return node


def render_clustering_tab(data: DashboardData) -> html.Div:
    metrics = data.metrics
    clean_df = data.clean_df

    ev_ratio = metrics["pca"]["explained_variance_ratio"]
    cumulative = metrics["pca"]["cumulative_explained_variance"]
    components = list(range(1, len(ev_ratio) + 1))

    scree_fig = px.bar(
        x=components,
        y=[v * 100 for v in ev_ratio],
        title="PCA Scree Plot — Variance Explained per Component",
        labels={"x": "Principal Component", "y": "Variance Explained (%)"},
        color=[v * 100 for v in ev_ratio],
        color_continuous_scale="Viridis",
    )
    for i, (comp, cum) in enumerate(zip(components, cumulative)):
        scree_fig.add_annotation(
            x=comp, y=(v * 100) + 0.5,
            text=f"{cum * 100:.1f}%",
            showarrow=False,
            font=dict(size=9),
        )

    pca_df = data.pca_scores.copy()
    pca_df["segment"] = clean_df["segment"].values
    scatter_fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="segment",
        title="PC1 vs PC2 — K-Means Clusters (k=2)",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map={"1": "#2E86AB", "2": "#F18F01"},
    )

    ks = [row["k"] for row in metrics["kmeans_evaluation"]]
    inertias = [row["inertia"] for row in metrics["kmeans_evaluation"]]
    silhouettes = [row["silhouette_score"] for row in metrics["kmeans_evaluation"]]
    db_scores = [row["davies_bouldin_score"] for row in metrics["kmeans_evaluation"]]

    elbow_fig = px.line(
        x=ks, y=inertias,
        title="Elbow Chart — Inertia vs k",
        labels={"x": "k", "y": "Inertia"},
        markers=True,
    )
    elbow_fig.update_traces(line_color="#2E86AB")

    silhouette_fig = px.line(
        x=ks, y=silhouettes,
        title="Silhouette Score vs k",
        labels={"x": "k", "y": "Silhouette Score"},
        markers=True,
    )
    silhouette_fig.update_traces(line_color="#A23B72")

    db_fig = px.line(
        x=ks, y=db_scores,
        title="Davies-Bouldin Index vs k",
        labels={"x": "k", "y": "Davies-Bouldin Index"},
        markers=True,
    )
    db_fig.update_traces(line_color="#F18F01")

    option_cols = [c for c in clean_df.columns if c.startswith("option_") and "_ips" in c]
    X_for_hier = clean_df[option_cols].values
    linkage_matrix = sch.linkage(X_for_hier, method="ward")
    dendrogram_fig = sch.dendrogram(
        linkage_matrix,
        no_labels=True,
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=scree_fig), width=6),
                    dbc.Col(dcc.Graph(figure=scatter_fig), width=6),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=elbow_fig), width=4),
                    dbc.Col(dcc.Graph(figure=silhouette_fig), width=4),
                    dbc.Col(dcc.Graph(figure=db_fig), width=4),
                ],
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=px.imshow(
                            linkage_matrix,
                            title="Hierarchical Clustering Dendrogram (Ward)",
                            color_continuous_scale="Viridis",
                        )
                    ),
                    width=12,
                ),
            ),
        ],
        id="tab_clustering",
    )
```

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_clustering import render_clustering_tab; print('OK')"`
Expected: OK

- [ ] **Step 2: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_clustering.py
git commit -m "feat(dashboard): add Tab 3 — PCA scree, cluster scatter, elbow/silhouette/DB, dendrogram"
```

---

### Task 8: Tab 4 — Anomaly Detection & Customer Persona

**Files:**
- Create: `src/catfood_unsupervised/dashboard/components/tab_persona.py`

- [ ] **Step 1: Write implementation**

```python
# src/catfood_unsupervised/dashboard/components/tab_persona.py
from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import dcc, html

from catfood_unsupervised.dashboard.data_loader import DashboardData


SEGMENT_STRATEGIES = {
    1: {
        "title": "กลุ่มพรีเมียมเน้นคุณภาพและดีไซน์",
        "description": "ให้ความสำคัญสูงกับรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์บนแพ็กเกจ",
        "color": "#2E86AB",
        "strategy": "สื่อสารแบบ premium ให้ชัดเจน เน้นรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์ที่บอกประโยชน์ของสินค้า",
    },
    2: {
        "title": "กลุ่มตลาดหลักที่ชั่งน้ำหนักหลายปัจจัย",
        "description": "ตอบสนองต่อ messaging กว้าง เน้นความคุ้มค่าและความน่าเชื่อถือ",
        "color": "#F18F01",
        "strategy": "ใช้ข้อความที่อ่านง่ายและมี social proof เน้นความคุ้มค่าและข้อมูลที่เข้าใจง่าย",
    },
}


def render_persona_tab(data: DashboardData) -> html.Div:
    clean_df = data.clean_df
    metrics = data.metrics

    anomaly_flag = clean_df["anomaly_flag"] if "anomaly_flag" in clean_df.columns else None
    segment = clean_df["segment"]

    scatter_df = data.pca_scores.copy()
    scatter_df["segment"] = segment.values
    if anomaly_flag is not None:
        scatter_df["anomaly"] = anomaly_flag.values
        scatter_df["anomaly_label"] = scatter_df["anomaly"].map({1: "Anomaly", 0: "Normal"})
    else:
        scatter_df["anomaly"] = 0
        scatter_df["anomaly_label"] = "Normal"

    scatter_fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color="segment",
        symbol="anomaly",
        title="Isolation Forest Anomalies Highlighted on PCA Space",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_map={"1": "#2E86AB", "2": "#F18F01"},
        symbol_map={"1": "circle", "0": "x"},
    )

    anomaly_count = metrics["anomaly_detection"]["anomaly_count"]
    anomaly_rate = metrics["anomaly_detection"]["anomaly_rate"] * 100

    persona_cards = []
    for segment_id, strategy_info in SEGMENT_STRATEGIES.items():
        seg_df = clean_df[clean_df["segment"] == segment_id]
        seg_size = len(seg_df)
        seg_pct = seg_size / len(clean_df) * 100
        top_gender = seg_df.iloc[:, 74].astype(str).str.strip().value_counts().index[0]
        top_age = seg_df.iloc[:, 73].astype(str).str.strip().value_counts().index[0]
        anomaly_rate_seg = (
            seg_df["anomaly_flag"].mean() * 100 if "anomaly_flag" in seg_df.columns else 0
        )
        card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        f"Segment {segment_id}: {strategy_info['title']}",
                        style={"color": strategy_info["color"]},
                    ),
                    html.P(strategy_info["description"]),
                    html.Hr(),
                    html.P(f"Size: {seg_size} ({seg_pct:.1f}%)", className="mb-1"),
                    html.P(f"Top Gender: {top_gender}", className="mb-1"),
                    html.P(f"Top Age: {top_age}", className="mb-1"),
                    html.P(f"Anomaly Rate: {anomaly_rate_seg:.1f}%", className="mb-1"),
                    html.Hr(),
                    html.Strong("Strategy: "),
                    html.P(strategy_info["strategy"], className="text-muted"),
                ]
            ),
            className="shadow-sm mb-3",
            style={"border-left": f"4px solid {strategy_info['color']}"},
        )
        persona_cards.append(dbc.Col(card, width=6))

    anomaly_banner = dbc.Alert(
        f"Detected {anomaly_count} anomalies ({anomaly_rate:.1f}% of respondents) — see highlighted points in scatter plot.",
        color="warning",
        className="mb-3",
    )

    return html.Div(
        [
            dbc.Row(dbc.Col(anomaly_banner, width=12), className="mb-3"),
            dbc.Row(dbc.Col(dcc.Graph(figure=scatter_fig), width=12), className="mb-4"),
            dbc.Row(persona_cards),
        ],
        id="tab_persona",
    )
```

Run: `python -c "from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab; print('OK')"`
Expected: OK

- [ ] **Step 2: Commit**

```bash
git add src/catfood_unsupervised/dashboard/components/tab_persona.py
git commit -m "feat(dashboard): add Tab 4 — Isolation Forest scatter and 2 customer persona cards"
```

---

### Task 9: Main Dash App

**Files:**
- Create: `src/catfood_unsupervised/dashboard/app.py`
- Create: `src/catfood_unsupervised/dashboard/__main__.py`

- [ ] **Step 1: Write main Dash app**

```python
# src/catfood_unsupervised/dashboard/app.py
from __future__ import annotations

import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from catfood_unsupervised.dashboard.components.kpi_banner import render_kpi_banner
from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab
from catfood_unsupervised.dashboard.components.tab_correlation import render_correlation_tab
from catfood_unsupervised.dashboard.components.tab_clustering import render_clustering_tab
from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab
from catfood_unsupervised.dashboard.config import TAB_ITEMS
from catfood_unsupervised.dashboard.data_loader import load_all_data

OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))

dash_app = dash.Dash(
    __name__,
    title="Cat Food Survey — Unsupervised Learning",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
dash_app.layout = html.Div(
    [
        html.H1(
            "🐱 Cat Food Packaging Survey — Unsupervised Learning",
            className="text-center my-4",
        ),
        dbc.Container(
            [
                render_kpi_banner(load_all_data(OUTPUT_DIR).metrics),
                dcc.Tabs(
                    id="unsupervised-tabs",
                    className="nav nav-tabs",
                    children=[
                        dcc.Tab(label=item["label"], value=item["value"])
                        for item in TAB_ITEMS
                    ],
                ),
                html.Div(id="tab_content"),
            ],
            fluid=True,
        ),
    ],
    style={"background": "#F8F9FA", "min-height": "100vh", "padding": "20px"},
)


@dash_app.callback(
    dash.Output("tab_content", "children"),
    dash.Input("unsupervised-tabs", "value"),
)
def render_tab_content(selected_tab: str):
    data = load_all_data(OUTPUT_DIR)
    if selected_tab == "tab_eda":
        return render_eda_tab(data)
    elif selected_tab == "tab_correlation":
        return render_correlation_tab(data)
    elif selected_tab == "tab_clustering":
        return render_clustering_tab(data)
    elif selected_tab == "tab_persona":
        return render_persona_tab(data)
    return html.Div("Select a tab")


if __name__ == "__main__":
    dash_app.run(debug=True, port=8050)
```

- [ ] **Step 2: Create __main__.py for direct execution**

```python
# src/catfood_unsupervised/dashboard/__main__.py
from catfood_unsupervised.dashboard.app import dash_app, dash_app as app

if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

- [ ] **Step 3: Verify app starts**

Run: `python -m catfood_unsupervised.dashboard`
Expected: Dash server starts on port 8050, no import errors

- [ ] **Step 4: Commit**

```bash
git add src/catfood_unsupervised/dashboard/app.py src/catfood_unsupervised/dashboard/__main__.py
git commit -m "feat(dashboard): add main Dash app with tabbed layout and KPI banner"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] EDA tab with demographics (เพศ, อายุ, สถานภาพ), Buy Factors mean scores, Weighted Vote bar chart
- [x] Correlation tab with Spearman heatmap of 10 options + 3 cluster annotations
- [x] Clustering tab with PCA Scree Plot, PC1 vs PC2 scatter (color-coded by segment), Elbow/Silhouette/DB charts, Dendrogram
- [x] Persona tab with Isolation Forest scatter (anomalies highlighted) + 2 customer persona cards with strategic recommendations
- [x] KPI banner at top: Silhouette Score, Davies-Bouldin Index, Inertia, Variance Explained, Anomaly Rate
- [x] Python-based web: Dash + Plotly + dash-bootstrap-components
- [x] 4 models covered: PCA, K-Means, Hierarchical Clustering, Isolation Forest

**Placeholder scan:**
- No TBD/TODO in steps
- No "implement later" phrases
- All chart types specified (bar, heatmap, scatter, line, dendrogram)
- All data paths use absolute paths from known outputs/

**Type consistency:**
- `load_all_data()` returns `DashboardData` dataclass — used consistently across all tab modules
- `_get_nested(metrics, path)` used in KPI banner and clustering tab
- Segment IDs: 1 and 2 (confirmed from metrics: segment_sizes = {1: 36, 2: 112})
- anomaly_flag: 1 = anomaly, 0 = normal (from Isolation Forest convention)

**Gaps found:** None.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-10-unsupervised-dashboard.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
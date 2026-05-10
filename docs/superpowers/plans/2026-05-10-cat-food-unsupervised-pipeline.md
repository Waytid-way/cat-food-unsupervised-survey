# Cat Food Unsupervised Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible unsupervised-learning pipeline from the BU cat-food survey CSV and generate Thai-language instructor reports plus a personal unsupervised memo.

**Architecture:** Use a small Python package under `src/` for data loading, cleaning, feature engineering, modeling, and report artifact generation. Use one CLI entrypoint to run the full pipeline end-to-end from the raw CSV, save outputs into `outputs/` and `reports/`, and keep assumptions explicit so metrics can be regenerated from source data.

**Tech Stack:** Python, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, pytest

---

### File Structure

**Create:**
- `C:\Users\COM\Projects\Cat-food Unsupervised\pyproject.toml`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\__init__.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\config.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\data_loading.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\preprocessing.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\features.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\models.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\reporting.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\pipeline.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\scripts\run_pipeline.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_data_loading.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_preprocessing.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_features.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_models.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_pipeline.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_reporting.py`
- `C:\Users\COM\Projects\Cat-food Unsupervised\tests\fixtures\mini_survey.csv`
- `C:\Users\COM\Projects\Cat-food Unsupervised\reports\descriptive_stats_th.md`
- `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_report_th.md`
- `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_owner_memo_th.md`

**Generate at runtime:**
- `C:\Users\COM\Projects\Cat-food Unsupervised\outputs\clean_dataset_with_segments.csv`
- `C:\Users\COM\Projects\Cat-food Unsupervised\outputs\metrics_summary.json`
- `C:\Users\COM\Projects\Cat-food Unsupervised\outputs\correlation_matrix.csv`
- `C:\Users\COM\Projects\Cat-food Unsupervised\outputs\segment_profiles.csv`
- `C:\Users\COM\Projects\Cat-food Unsupervised\outputs\plots\*.png`

### Task 1: Bootstrap The Project Skeleton

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\pyproject.toml`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\__init__.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\config.py`

- [ ] **Step 0: Initialize git if the workspace is not already a repository**

Run: `git init`
Expected: initialized empty Git repository in `C:\Users\COM\Projects\Cat-food Unsupervised\.git`

- [ ] **Step 1: Write the failing import test**

```python
from catfood_unsupervised.config import RAW_DATA_PATH


def test_raw_data_path_is_configured():
    assert RAW_DATA_PATH.name == "BU Data from Survey Cases_final(5).csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loading.py -v`
Expected: FAIL with `ModuleNotFoundError` for `catfood_unsupervised`

- [ ] **Step 3: Write minimal packaging and config implementation**

```python
from pathlib import Path

RAW_DATA_PATH = Path(r"C:\Users\COM\Downloads\BU Data from Survey Cases_final(5).csv")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_loading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/catfood_unsupervised/__init__.py src/catfood_unsupervised/config.py tests/test_data_loading.py
git commit -m "chore: bootstrap unsupervised project structure"
```

### Task 2: Load The Raw Survey Export Correctly

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\data_loading.py`
- Modify: `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_data_loading.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\tests\fixtures\mini_survey.csv`

- [ ] **Step 1: Write the failing loader test**

```python
from catfood_unsupervised.data_loading import load_raw_export


def test_load_raw_export_uses_second_row_as_headers(tmp_path):
    csv_text = (
        "Brief,Brief2\n"
        "Timestamp,top3_choice\n"
        "6/1/2024 10:00:00,\"Option 1, Option 2, Option 3\"\n"
        ",\n"
    )
    file_path = tmp_path / "mini.csv"
    file_path.write_text(csv_text, encoding="utf-8")

    df = load_raw_export(file_path)

    assert list(df.columns) == ["Timestamp", "top3_choice"]
    assert len(df) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loading.py::test_load_raw_export_uses_second_row_as_headers -v`
Expected: FAIL with `ImportError` or missing function

- [ ] **Step 3: Write the minimal loader**

```python
import pandas as pd


def load_raw_export(path):
    raw = pd.read_csv(path, header=None, encoding="utf-8")
    data = raw.iloc[2:].copy()
    data.columns = raw.iloc[1].tolist()
    timestamp = data.iloc[:, 0].astype("string")
    keep_mask = timestamp.notna() & (timestamp.str.strip() != "")
    return data.loc[keep_mask].reset_index(drop=True)
```

- [ ] **Step 4: Add a second failing test for top-3 completion filtering**

```python
from catfood_unsupervised.data_loading import filter_completed_responses


def test_filter_completed_responses_keeps_only_non_empty_top3():
    df = pd.DataFrame(
        {
            "Timestamp": ["t1", "t2"],
            "top3_choice": ["Option 1, Option 2, Option 3", None],
        }
    )

    result = filter_completed_responses(df, "top3_choice")

    assert len(result) == 1
    assert result.iloc[0]["Timestamp"] == "t1"
```

- [ ] **Step 5: Implement the filter and rerun tests**

```python
def filter_completed_responses(df, top3_column):
    top3 = df[top3_column].astype("string")
    keep_mask = top3.notna() & (top3.str.strip() != "")
    return df.loc[keep_mask].reset_index(drop=True)
```

Run: `python -m pytest tests/test_data_loading.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/data_loading.py tests/test_data_loading.py tests/fixtures/mini_survey.csv
git commit -m "feat: load and filter raw survey export"
```

### Task 3: Encode Survey Fields And Build Modeling Features

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\preprocessing.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\features.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_preprocessing.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_features.py`

- [ ] **Step 1: Write the failing mapping test**

```python
from catfood_unsupervised.preprocessing import map_likert_value


def test_map_likert_value_maps_thai_scale():
    mapping = {"ไม่เห็นด้วยเลย": 1, "เฉยๆ": 3, "เห็นด้วยที่สุด": 5}
    assert map_likert_value("เห็นด้วยที่สุด", mapping) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocessing.py::test_map_likert_value_maps_thai_scale -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement survey mapping helpers**

```python
def map_likert_value(value, mapping):
    if value is None:
        return None
    return mapping.get(str(value).strip())
```

- [ ] **Step 4: Write the failing vote-weight test**

```python
import pandas as pd

from catfood_unsupervised.features import build_vote_features


def test_build_vote_features_applies_rank_weights():
    df = pd.DataFrame({"rank1": [3], "rank2": [2], "rank3": [1]})
    result = build_vote_features(df, rank_columns=("rank1", "rank2", "rank3"), option_count=3)

    assert result.loc[0, "vote_03"] == 3
    assert result.loc[0, "vote_02"] == 2
    assert result.loc[0, "vote_01"] == 1
```

- [ ] **Step 5: Implement vote features, ipsatization, and imputation utilities**

```python
def build_vote_features(df, rank_columns, option_count):
    result = df.copy()
    for opt in range(1, option_count + 1):
        result[f"vote_{opt:02d}"] = (
            (result[rank_columns[0]] == opt).astype(int) * 3
            + (result[rank_columns[1]] == opt).astype(int) * 2
            + (result[rank_columns[2]] == opt).astype(int) * 1
        )
    return result
```

Run: `python -m pytest tests/test_preprocessing.py tests/test_features.py -v`
Expected: PASS

- [ ] **Step 6: Expand tests for ipsatization and KNN imputation**

```python
def test_ipsatize_rows_center_each_row():
    X = np.array([[5.0, 3.0, 1.0]])
    result = ipsatize_rows(X)
    assert result.mean(axis=1)[0] == pytest.approx(0.0)
```

```python
def test_impute_buy_factors_returns_no_nan_values():
    df = pd.DataFrame({"bf_a": [1.0, None, 3.0], "bf_b": [2.0, 2.0, None]})
    result = impute_buy_factors(df, ["bf_a", "bf_b"])
    assert not np.isnan(result).any()
```

- [ ] **Step 7: Commit**

```bash
git add src/catfood_unsupervised/preprocessing.py src/catfood_unsupervised/features.py tests/test_preprocessing.py tests/test_features.py
git commit -m "feat: encode survey responses and engineer features"
```

### Task 4: Train PCA, Clustering, And Anomaly Models

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\models.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\tests\test_models.py`

- [ ] **Step 1: Write the failing PCA shape test**

```python
import numpy as np

from catfood_unsupervised.models import run_pca


def test_run_pca_returns_requested_component_count():
    X = np.random.RandomState(42).randn(20, 10)
    model, scores, summary = run_pca(X, n_components=5, random_state=42)
    assert scores.shape == (20, 5)
    assert len(summary["explained_variance_ratio"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_run_pca_returns_requested_component_count -v`
Expected: FAIL with missing function

- [ ] **Step 3: Implement PCA and K-Means evaluation helpers**

```python
def evaluate_kmeans_range(X, ks, random_state):
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=30)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "inertia": model.inertia_,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Add failing tests for hierarchical validation and anomaly detection**

```python
from catfood_unsupervised.models import run_hierarchical_validation, run_isolation_forest


def test_run_hierarchical_validation_returns_cophenetic_value():
    X = np.random.RandomState(0).randn(12, 4)
    result = run_hierarchical_validation(X)
    assert 0.0 <= result["cophenetic_correlation"] <= 1.0
```

```python
def test_run_isolation_forest_returns_binary_flags():
    X = np.random.RandomState(0).randn(25, 6)
    result = run_isolation_forest(X, contamination=0.12, random_state=42)
    assert set(result["anomaly_flag"]) <= {0, 1}
```

- [ ] **Step 5: Implement final model helpers and rerun tests**

```python
def run_isolation_forest(X, contamination, random_state):
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    return pd.DataFrame(
        {
            "anomaly_flag": (labels == -1).astype(int),
            "anomaly_score": scores,
        }
    )
```

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/models.py tests/test_models.py
git commit -m "feat: add PCA clustering and anomaly models"
```

### Task 5: Assemble The End-To-End Pipeline And Export Artifacts

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\pipeline.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\scripts\run_pipeline.py`

- [ ] **Step 1: Write the failing smoke test for pipeline outputs**

```python
from pathlib import Path

from catfood_unsupervised.pipeline import run_pipeline


def test_run_pipeline_writes_metrics_summary(tmp_path):
    result = run_pipeline(output_dir=tmp_path)
    assert (tmp_path / "metrics_summary.json").exists()
    assert "final_cluster_k" in result["metrics"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with missing module or missing file writes

- [ ] **Step 3: Implement the orchestration flow**

```python
def run_pipeline(output_dir):
    raw_df = load_raw_export(RAW_DATA_PATH)
    clean_df = filter_completed_responses(raw_df, top3_column=TOP3_COLUMN)
    prepared = prepare_modeling_dataframe(clean_df)
    metrics, enriched_df = fit_unsupervised_suite(prepared)
    write_outputs(enriched_df, metrics, output_dir)
    return {"metrics": metrics, "dataframe": enriched_df}
```

- [ ] **Step 4: Add the CLI entrypoint and run the full pipeline**

```python
from catfood_unsupervised.pipeline import run_pipeline
from catfood_unsupervised.config import OUTPUT_DIR


if __name__ == "__main__":
    run_pipeline(output_dir=OUTPUT_DIR)
```

Run: `python scripts/run_pipeline.py`
Expected: creates `outputs/clean_dataset_with_segments.csv` and companion metrics files

- [ ] **Step 5: Validate the rerun metrics against the user-provided expectations**

Run: `python -c "import json, pathlib; print(json.loads(pathlib.Path('outputs/metrics_summary.json').read_text(encoding='utf-8')))" `
Expected: printed metrics include row counts, PCA variance, K-Means comparison, cophenetic correlation, anomaly count, and segment sizes

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/pipeline.py scripts/run_pipeline.py outputs
git commit -m "feat: run end-to-end unsupervised pipeline"
```

### Task 6: Generate Thai Reports And Personal Memo

**Files:**
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\src\catfood_unsupervised\reporting.py`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\descriptive_stats_th.md`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_report_th.md`
- Create: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_owner_memo_th.md`

- [ ] **Step 1: Write the failing report rendering test**

```python
from catfood_unsupervised.reporting import render_descriptive_report


def test_render_descriptive_report_contains_sample_size():
    text = render_descriptive_report({"n_final": 148, "female_pct": 73.0})
    assert "n = 148" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement markdown rendering functions**

```python
def render_descriptive_report(metrics):
    return f"""# รายงานสถิติเชิงพรรณนาและความสัมพันธ์ของตัวแปร

กลุ่มตัวอย่างหลังการคัดกรองมีจำนวน n = {metrics['n_final']} คน
"""
```

- [ ] **Step 4: Render all three Thai documents from rerun metrics**

```python
write_text(REPORT_DIR / "descriptive_stats_th.md", render_descriptive_report(metrics))
write_text(REPORT_DIR / "unsupervised_report_th.md", render_unsupervised_report(metrics))
write_text(REPORT_DIR / "unsupervised_owner_memo_th.md", render_owner_memo(metrics))
```

- [ ] **Step 5: Review the report language against the rerun outputs**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: PASS

Run: `Get-Content 'C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_report_th.md'`
Expected: Thai markdown with PCA, clustering, hierarchical validation, anomaly detection, personas, and limitations

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/reporting.py reports tests/test_reporting.py
git commit -m "feat: generate Thai unsupervised reports"
```

### Task 7: Final Verification And Delivery Check

**Files:**
- Modify: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\descriptive_stats_th.md`
- Modify: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_report_th.md`
- Modify: `C:\Users\COM\Projects\Cat-food Unsupervised\reports\unsupervised_owner_memo_th.md`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: PASS

- [ ] **Step 2: Rerun the pipeline from the real CSV**

Run: `python scripts/run_pipeline.py`
Expected: PASS and refreshed files in `outputs/` and `reports/`

- [ ] **Step 3: Spot-check key metrics against report claims**

Run: `python -c "import pandas as pd; df = pd.read_csv('outputs/clean_dataset_with_segments.csv'); print(df.shape); print(df['segment'].value_counts().sort_index().to_dict()); print(df['anomaly_flag'].mean())"`
Expected: row count, segment sizes, and anomaly rate printed for manual comparison with the Thai reports

- [ ] **Step 4: Confirm delivery inventory**

Run: `Get-ChildItem 'C:\Users\COM\Projects\Cat-food Unsupervised\outputs' -Recurse`
Expected: clean CSV, metrics tables, and plots are present

Run: `Get-ChildItem 'C:\Users\COM\Projects\Cat-food Unsupervised\reports'`
Expected: three Thai report files are present

- [ ] **Step 5: Commit**

```bash
git add outputs reports
git commit -m "docs: finalize unsupervised deliverables"
```

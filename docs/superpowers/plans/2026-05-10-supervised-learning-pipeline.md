# Supervised Learning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible supervised classifier that predicts survey segment labels from the cleaned dataset, compares Logistic Regression, Random Forest, Gradient Boosting, and RBF SVM, and exposes the results in dashboard-ready artifacts and Thai-language reports.

**Architecture:** Read the canonical cleaned dataset from `outputs/clean_dataset_with_segments.csv`, explicitly remove anomaly rows before training, and whitelist only the approved feature groups so segment leakage cannot slip into the classifier. Train every model through one shared evaluation harness that returns the same metrics, per-class scores, and persisted artifacts, then let the dashboard read those saved outputs instead of retraining. Keep the code tolerant of the current 2-segment output, but never hard-code the class count so a later rerun with a different number of segments still works.

**Tech Stack:** Python, pandas, numpy, scikit-learn, plotly, dash, pytest, pickle

---

## File Structure

**Create:**
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\__init__.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\config.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\data_loading.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\features.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\models.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\pipeline.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\reporting.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\supervised_data_loader.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\components\tab_supervised.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\scripts\run_supervised_pipeline.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_data_loading.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_features.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_models.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_pipeline.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_reporting.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\dashboard\test_tab_supervised.py`

**Modify:**
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\app.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\config.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\components\__init__.py`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\bootstrap.py` only if a small compatibility tweak is needed for a new tab
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\docs\supervised-team-handoff.md` to point at the supervised runbook after implementation is done

**Runtime outputs to generate:**
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\metrics_summary.json`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\model_comparison.csv`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\confusion_matrix.csv`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\feature_importance.csv`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\predictions.csv`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\outputs\supervised\best_model.pkl`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\reports\supervised\supervised_model_report_th.md`
- `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\reports\supervised\supervised_owner_memo_th.md`

### Task 1: Lock The Supervised Data Contract

**Files:**
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\config.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\data_loading.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_data_loading.py`

- [ ] **Step 1: Write the failing data-contract test**

```python
from pathlib import Path

import pandas as pd

from catfood_unsupervised.supervised.data_loading import load_supervised_dataset


def test_load_supervised_dataset_filters_anomalies_and_keeps_segment_labels(tmp_path: Path):
    fixture = pd.DataFrame(
        {
            "segment": [1, 2, 1],
            "anomaly_flag": [0, 1, 0],
            "age": ["20-29ปี", "30-39ปี", "20-29ปี"],
            "gender": ["หญิง", "ชาย", "หญิง"],
        }
    )
    source = tmp_path / "supervised_fixture.csv"
    fixture.to_csv(source, index=False)

    df = load_supervised_dataset(source)

    assert df["anomaly_flag"].eq(0).all()
    assert sorted(df["segment"].unique().tolist()) == [1]
```

- [ ] **Step 2: Run the test and confirm it fails for the right reason**

Run: `python -m pytest tests/supervised/test_data_loading.py::test_load_supervised_dataset_filters_anomalies_and_keeps_segment_labels -v`

Expected: FAIL because `load_supervised_dataset` does not exist yet.

- [ ] **Step 3: Implement the loader and contract constants**

```python
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPERVISED_INPUT_PATH = PROJECT_ROOT / "outputs" / "clean_dataset_with_segments.csv"
SUPERVISED_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
SUPERVISED_REPORT_DIR = PROJECT_ROOT / "reports" / "supervised"

TARGET_COLUMN = "segment"
ANOMALY_COLUMN = "anomaly_flag"
LEAKAGE_PREFIXES = ("PC",)
LEAKAGE_COLUMNS = {
    "segment",
    "anomaly_flag",
    "top3_rank_1",
    "top3_rank_2",
    "top3_rank_3",
}


def load_supervised_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {TARGET_COLUMN, ANOMALY_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df.loc[df[ANOMALY_COLUMN].eq(0)].copy()
    filtered[TARGET_COLUMN] = pd.to_numeric(filtered[TARGET_COLUMN], errors="raise").astype(int)
    return filtered.reset_index(drop=True)
```

- [ ] **Step 4: Add a second test for the feature whitelist**

```python
from catfood_unsupervised.supervised.features import build_supervised_feature_frame


def test_build_supervised_feature_frame_excludes_leakage_columns():
    df = pd.DataFrame(
        {
            "segment": [1, 2],
            "anomaly_flag": [0, 0],
            "PC1": [0.1, 0.2],
            "PC2": [0.2, 0.3],
            "age": ["20-29ปี", "30-39ปี"],
            "gender": ["หญิง", "ชาย"],
            "marital": ["โสด ไม่มีแฟน", "แต่งงานแล้ว"],
        }
    )

    X, y = build_supervised_feature_frame(df)

    assert "segment" not in X.columns
    assert "anomaly_flag" not in X.columns
    assert not any(col.startswith("PC") for col in X.columns)
    assert y.name == "segment"
```

- [ ] **Step 5: Implement the feature-frame builder and rerun the tests**

```python
import pandas as pd

from catfood_unsupervised.supervised.data_loading import ANOMALY_COLUMN, TARGET_COLUMN


def build_supervised_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN].astype(int).rename(TARGET_COLUMN)
    X = df.drop(columns=[TARGET_COLUMN, ANOMALY_COLUMN], errors="ignore").copy()
    X = X.loc[:, ~X.columns.str.startswith("PC")]
    categorical = [column for column in ["age", "gender", "marital"] if column in X.columns]
    X = pd.get_dummies(X, columns=categorical, dtype=int)
    return X, y
```

Run: `python -m pytest tests/supervised/test_data_loading.py tests/supervised/test_features.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/supervised/config.py src/catfood_unsupervised/supervised/data_loading.py src/catfood_unsupervised/supervised/features.py tests/supervised/test_data_loading.py tests/supervised/test_features.py
git commit -m "feat: define supervised data contract"
```

### Task 2: Build The Model Suite And Evaluation Harness

**Files:**
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\models.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_models.py`
- Modify: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\features.py`

- [ ] **Step 1: Write the failing model-comparison test**

```python
from sklearn.datasets import make_classification

from catfood_unsupervised.supervised.models import build_model_suite, evaluate_model_suite


def test_evaluate_model_suite_returns_four_models_and_ranked_metrics():
    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=7,
    )

    suite = build_model_suite(random_state=7)
    comparison, fitted_models, predictions = evaluate_model_suite(suite, X, y, random_state=7)

    assert comparison["model_name"].tolist() == [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm_rbf",
    ]
    assert set(fitted_models) == set(comparison["model_name"])
    assert set(predictions) == set(comparison["model_name"])
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/supervised/test_models.py::test_evaluate_model_suite_returns_four_models_and_ranked_metrics -v`

Expected: FAIL because the model suite functions do not exist yet.

- [ ] **Step 3: Implement the shared model builders**

```python
from collections.abc import Mapping

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model_suite(random_state: int) -> Mapping[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }
```

```python
def evaluate_model_suite(models, X, y, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    rows = []
    fitted = {}
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted[model_name] = model
        predictions[model_name] = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=X_test.index)
        rows.append(
            {
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "weighted_f1", "accuracy", "model_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return comparison, fitted, predictions
```

- [ ] **Step 4: Add tests for feature importance and split behavior**

```python
from catfood_unsupervised.supervised.features import make_stratified_split
from catfood_unsupervised.supervised.models import extract_feature_importance


def test_make_stratified_split_preserves_class_presence():
    X = pd.DataFrame({"a": range(30), "b": range(30, 60)})
    y = pd.Series([1] * 20 + [2] * 10, name="segment")

    split = make_stratified_split(X, y, random_state=7)

    assert set(split["y_train"].unique()) == {1, 2}
    assert set(split["y_test"].unique()) == {1, 2}
```

```python
def test_extract_feature_importance_returns_sorted_feature_scores():
    # Use the best estimator from the suite once the implementation exists.
    ...
```

- [ ] **Step 5: Implement split helpers, importance extraction, and rerun tests**

```python
def make_stratified_split(X: pd.DataFrame, y: pd.Series, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
```

```python
from sklearn.inspection import permutation_importance


def extract_feature_importance(model, feature_names, X_test, y_test, random_state: int):
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = abs(estimator.coef_).ravel()
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state)
        scores = result.importances_mean

    importance = pd.DataFrame({"feature": list(feature_names), "importance": scores})
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)
```

Run: `python -m pytest tests/supervised/test_models.py tests/supervised/test_features.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/supervised/models.py src/catfood_unsupervised/supervised/features.py tests/supervised/test_models.py tests/supervised/test_features.py
git commit -m "feat: train and evaluate supervised model suite"
```

### Task 3: Orchestrate Training, Persist Artifacts, And Generate Thai Reports

**Files:**
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\pipeline.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\supervised\reporting.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\scripts\run_supervised_pipeline.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_pipeline.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\supervised\test_reporting.py`

- [ ] **Step 1: Write the failing end-to-end pipeline test**

```python
from pathlib import Path

from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


def test_run_supervised_pipeline_writes_all_outputs(tmp_path: Path):
    input_path = Path("C:/Users/wootty/Downloads/mylastproject/cat-food-unsupervised-survey/outputs/clean_dataset_with_segments.csv")
    output_dir = tmp_path / "supervised_outputs"
    report_dir = tmp_path / "supervised_reports"

    result = run_supervised_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
    )

    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert (report_dir / "supervised_model_report_th.md").exists()
    assert result["best_model_name"] in result["metrics"]["model_comparison"]["model_name"]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/supervised/test_pipeline.py::test_run_supervised_pipeline_writes_all_outputs -v`

Expected: FAIL because `run_supervised_pipeline` does not exist yet.

- [ ] **Step 3: Implement the pipeline and artifact writers**

```python
import json
import pickle
from pathlib import Path

from catfood_unsupervised.supervised.config import (
    SUPERVISED_INPUT_PATH,
    SUPERVISED_OUTPUT_DIR,
    SUPERVISED_REPORT_DIR,
)
from catfood_unsupervised.supervised.data_loading import load_supervised_dataset
from catfood_unsupervised.supervised.features import build_supervised_feature_frame, make_stratified_split
from catfood_unsupervised.supervised.models import (
    build_model_suite,
    evaluate_model_suite,
    extract_feature_importance,
)


def run_supervised_pipeline(
    input_path: str | Path = SUPERVISED_INPUT_PATH,
    output_dir: str | Path = SUPERVISED_OUTPUT_DIR,
    report_dir: str | Path = SUPERVISED_REPORT_DIR,
    *,
    random_state: int = 42,
) -> dict:
    df = load_supervised_dataset(input_path)
    X, y = build_supervised_feature_frame(df)
    split = make_stratified_split(X, y, random_state=random_state)
    model_suite = build_model_suite(random_state=random_state)
    comparison, fitted_models, predictions = evaluate_model_suite(
        model_suite, split["X_train"], split["y_train"], random_state=random_state
    )
    best_model_name = comparison.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]
    best_predictions = predictions[best_model_name]
    feature_importance = extract_feature_importance(
        best_model,
        X.columns,
        split["X_test"],
        split["y_test"],
        random_state=random_state,
    )
    metrics = {
        "row_count": int(len(df)),
        "target_classes": sorted(y.unique().tolist()),
        "best_model_name": best_model_name,
        "model_comparison": comparison.to_dict(orient="records"),
    }
    _write_outputs(output_dir, best_model, comparison, best_predictions, feature_importance, metrics)
    _write_reports(report_dir, metrics, comparison, feature_importance)
    return {"metrics": metrics, "best_model_name": best_model_name}
```

```python
def _write_outputs(output_dir, best_model, comparison, predictions, feature_importance, metrics):
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(base / "model_comparison.csv", index=False)
    predictions.to_csv(base / "predictions.csv", index=False)
    feature_importance.to_csv(base / "feature_importance.csv", index=False)
    (base / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (base / "best_model.pkl").open("wb") as fh:
        pickle.dump(best_model, fh)
```

```python
def _write_reports(report_dir, metrics, comparison, feature_importance):
    # Render Thai markdown from the saved metrics and write both documents.
    ...
```

- [ ] **Step 4: Add the CLI entrypoint**

```python
from catfood_unsupervised.supervised.config import SUPERVISED_INPUT_PATH, SUPERVISED_OUTPUT_DIR, SUPERVISED_REPORT_DIR
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


if __name__ == "__main__":
    run_supervised_pipeline(
        input_path=SUPERVISED_INPUT_PATH,
        output_dir=SUPERVISED_OUTPUT_DIR,
        report_dir=SUPERVISED_REPORT_DIR,
        random_state=42,
    )
```

- [ ] **Step 5: Implement Thai report rendering and rerun the tests**

```python
def render_supervised_model_report(context):
    return f"""# รายงานการเรียนรู้แบบมีผู้สอน

## ภาพรวม
- จำนวนตัวอย่างที่ใช้ฝึก: {context['row_count']} รายการ
- จำนวนคลาสเป้าหมาย: {len(context['target_classes'])}
- โมเดลที่ดีที่สุด: {context['best_model_name']}

## Model Comparison
{comparison_table_here}
"""
```

Run:

```bash
python -m pytest tests/supervised/test_pipeline.py tests/supervised/test_reporting.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/supervised/pipeline.py src/catfood_unsupervised/supervised/reporting.py scripts/run_supervised_pipeline.py tests/supervised/test_pipeline.py tests/supervised/test_reporting.py
git commit -m "feat: orchestrate supervised pipeline and reports"
```

### Task 4: Add The Supervised Dashboard Tab

**Files:**
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\supervised_data_loader.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\components\tab_supervised.py`
- Modify: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\config.py`
- Modify: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\app.py`
- Modify: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\src\catfood_unsupervised\dashboard\components\__init__.py`
- Create: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\tests\dashboard\test_tab_supervised.py`

- [ ] **Step 1: Write the failing dashboard-tab test**

```python
from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_dashboard_bundle
from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab
from catfood_unsupervised.supervised.config import SUPERVISED_OUTPUT_DIR


def test_render_supervised_tab_contains_model_comparison_and_roc():
    bundle = load_supervised_dashboard_bundle(SUPERVISED_OUTPUT_DIR)
    tab = render_supervised_tab(bundle)

    assert tab is not None
    assert "Model Comparison" in tab.children[0].children[0].children
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/dashboard/test_tab_supervised.py -v`

Expected: FAIL because the loader and component do not exist yet.

- [ ] **Step 3: Implement the dashboard bundle loader**

```python
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SupervisedDashboardBundle:
    metrics: dict
    comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


def load_supervised_dashboard_bundle(output_dir: str | Path) -> SupervisedDashboardBundle:
    base = Path(output_dir)
    metrics = json.loads((base / "metrics_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(base / "model_comparison.csv")
    feature_importance = pd.read_csv(base / "feature_importance.csv")
    predictions = pd.read_csv(base / "predictions.csv")
    return SupervisedDashboardBundle(metrics, comparison, feature_importance, predictions)
```

```python
def render_supervised_tab(bundle: SupervisedDashboardBundle) -> html.Div:
    # Build model comparison bar chart, confusion matrix heatmap, per-class metrics table,
    # feature importance chart, and ROC curve from the saved artifacts.
    ...
```

- [ ] **Step 4: Wire the new tab into the dashboard app**

```python
TAB_ITEMS = [
    {"label": "1. EDA & Stats", "value": "tab_eda", "href": "#tab_eda"},
    {"label": "2. Correlation", "value": "tab_correlation", "href": "#tab_correlation"},
    {"label": "3. Clustering", "value": "tab_clustering", "href": "#tab_clustering"},
    {"label": "4. Persona", "value": "tab_persona", "href": "#tab_persona"},
    {"label": "5. Supervised", "value": "tab_supervised", "href": "#tab_supervised"},
]
```

```python
if selected_tab == "tab_supervised":
    return render_supervised_tab(supervised_bundle)
```

- [ ] **Step 5: Make the tab render from real artifacts and rerun the test**

Run:

```bash
python -m pytest tests/dashboard/test_tab_supervised.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/catfood_unsupervised/dashboard/supervised_data_loader.py src/catfood_unsupervised/dashboard/components/tab_supervised.py src/catfood_unsupervised/dashboard/config.py src/catfood_unsupervised/dashboard/app.py src/catfood_unsupervised/dashboard/components/__init__.py tests/dashboard/test_tab_supervised.py
git commit -m "feat: add supervised dashboard tab"
```

### Task 5: Final Verification And Handoff

**Files:**
- Modify: `C:\Users\wootty\Downloads\mylastproject\cat-food-unsupervised-survey\docs\supervised-team-handoff.md`

- [ ] **Step 1: Run the supervised pipeline on the real cleaned dataset**

Run:

```bash
python scripts/run_supervised_pipeline.py
```

Expected: `outputs/supervised/` contains `metrics_summary.json`, `model_comparison.csv`, `feature_importance.csv`, `predictions.csv`, and `best_model.pkl`.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with no new regressions.

- [ ] **Step 3: Launch the dashboard and verify the new supervised tab**

Run:

```bash
python -m catfood_unsupervised.dashboard
```

Expected: the new `Supervised` tab shows model comparison, confusion matrix, per-class precision/recall/F1, feature importance, and ROC curve without callback errors.

- [ ] **Step 4: Update the handoff note with the final command sequence**

Add a short section to `docs/supervised-team-handoff.md` that tells the next engineer:
- which script to run,
- which files are generated,
- which dashboard tab to inspect,
- and which metrics are the source of truth for the business team.

- [ ] **Step 5: Commit the final handoff updates**

```bash
git add docs/supervised-team-handoff.md
git commit -m "docs: finalize supervised handoff guidance"
```

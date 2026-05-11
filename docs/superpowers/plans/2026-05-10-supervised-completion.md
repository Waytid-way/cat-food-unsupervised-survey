# Supervised Completion & Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the supervised slice so it can train on the cleaned survey export, score a new customer row, stay stable if the class count changes, and ship portable artifacts and Thai reports that match the intent of `GOAL.md`.

**Architecture:** Keep the current supervised pipeline as the backbone. Split the work into three layers: canonical schema and validation, model training/scoring, and presentation/reporting. Replace brittle index-based feature selection with an explicit schema contract, make metrics multiclass-safe, and ensure every artifact can be regenerated from the current workspace without stale absolute paths. The existing supervised dashboard tab stays the source of truth for model comparison and interpretation; the new scoring path handles "new customer" prediction.

**Tech Stack:** Python, pandas, numpy, scikit-learn, matplotlib, plotly, dash, pytest, pickle

---

## File Structure

**Create:**
- `src/catfood_unsupervised/supervised/schema.py`
- `src/catfood_unsupervised/supervised/scoring.py`
- `scripts/predict_supervised_segment.py`
- `tests/supervised/test_schema.py`
- `tests/supervised/test_scoring.py`

**Modify:**
- `src/catfood_unsupervised/supervised/config.py`
- `src/catfood_unsupervised/supervised/data_loading.py`
- `src/catfood_unsupervised/supervised/features.py`
- `src/catfood_unsupervised/supervised/models.py`
- `src/catfood_unsupervised/supervised/pipeline.py`
- `src/catfood_unsupervised/supervised/reporting.py`
- `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- `docs/supervised-team-handoff.md`
- `README.md` if the run commands need to be surfaced at the top level

**Runtime outputs to regenerate:**
- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/confusion_matrix.png`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `outputs/supervised/best_model.pkl`
- `reports/supervised/supervised_model_report_th.md`
- `reports/supervised/supervised_owner_memo_th.md`

---

### Task 1: Lock the supervised data contract

**Files:**
- Create: `src/catfood_unsupervised/supervised/schema.py`
- Modify: `src/catfood_unsupervised/supervised/config.py`
- Modify: `src/catfood_unsupervised/supervised/data_loading.py`
- Modify: `src/catfood_unsupervised/supervised/features.py`
- Test: `tests/supervised/test_schema.py`
- Test: `tests/supervised/test_data_loading.py`

The current implementation works, but the feature contract is still driven by column positions. This task makes the supervised slice explicit and safer to rerun against future exports.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_build_supervised_feature_frame_excludes_leakage_columns(supervised_fixture_path):
    df = load_supervised_dataset(supervised_fixture_path)
    X, y = build_supervised_feature_frame(df)

    assert TARGET_COLUMN not in X.columns
    assert ANOMALY_COLUMN not in X.columns
    assert not any(column.startswith("PC") for column in X.columns)
    assert not any(column.startswith("vote_") for column in X.columns)
    assert not any(column.startswith("top3_rank_") for column in X.columns)
    assert y.name == TARGET_COLUMN
```

Run:
`python -m pytest tests/supervised/test_schema.py tests/supervised/test_data_loading.py -q`

Expected: fail until the schema helpers and named feature list exist.

- [ ] **Step 2: Implement the schema helper**

Add `schema.py` with:
- `TARGET_COLUMN = "segment"`
- `ANOMALY_COLUMN = "anomaly_flag"`
- `CANONICAL_FEATURE_COLUMNS` as an explicit tuple of the 18 current survey feature headers copied from the cleaned CSV header snapshot
- `LEAKAGE_COLUMNS` for `segment`, `anomaly_flag`, `top3_rank_*`, `vote_*`, and any PCA output columns
- `validate_supervised_frame(df)` that checks required columns, missing feature columns, and stable ordering

- [ ] **Step 3: Update loader and feature builder to use the schema**

Change `load_supervised_dataset()` to:
- keep only rows where `anomaly_flag == 0`
- coerce `segment` to `int`
- preserve the original row order after filtering

Change `build_supervised_feature_frame()` to:
- select features by canonical name, not by index
- return `X` with a stable, schema-defined column order
- keep all feature values as strings so the current one-hot encoder still works

- [ ] **Step 4: Re-run the focused tests**

Run:
`python -m pytest tests/supervised/test_schema.py tests/supervised/test_data_loading.py -q`

Expected: PASS.

---

### Task 2: Add a scoring path for a new customer row

**Files:**
- Create: `src/catfood_unsupervised/supervised/scoring.py`
- Create: `scripts/predict_supervised_segment.py`
- Test: `tests/supervised/test_scoring.py`
- Modify: `src/catfood_unsupervised/supervised/pipeline.py`

This is the missing business-facing piece in `GOAL.md`: the code can train on segments, but it still needs a simple way to predict the segment for a new customer without rerunning training.

- [ ] **Step 1: Write the failing scoring test**

```python
def test_predict_segments_returns_probabilities(supervised_fixture_path, tmp_path):
    df = load_supervised_dataset(supervised_fixture_path)
    X, _ = build_supervised_feature_frame(df)
    new_customer = pd.DataFrame([X.iloc[0].to_dict()])

    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
        random_state=7,
        test_size=0.25,
    )

    scored = predict_supervised_segment(
        model_path=tmp_path / "out" / "best_model.pkl",
        input_frame=new_customer,
    )

    assert "predicted_segment" in scored.columns
    assert any(column.startswith("prob_class_") for column in scored.columns)
    assert scored.filter(like="prob_class_").sum(axis=1).iloc[0] == pytest.approx(1.0)
```

Run:
`python -m pytest tests/supervised/test_scoring.py -q`

Expected: fail until the scoring module exists.

- [ ] **Step 2: Implement the reusable scorer**

Add a small library surface that:
- loads `best_model.pkl`
- applies the same schema validation and preprocessing path used by training
- returns a dataframe with `predicted_segment` plus one `prob_class_<label>` column per class
- accepts both a one-row frame and a batch CSV

- [ ] **Step 3: Add the CLI wrapper**

`scripts/predict_supervised_segment.py` should support:
- `--model-path`
- `--input-csv`
- `--output-csv`

Example usage:
```bash
python scripts/predict_supervised_segment.py ^
  --model-path outputs/supervised/best_model.pkl ^
  --input-csv data/new_customer.csv ^
  --output-csv outputs/supervised/new_customer_predictions.csv
```

- [ ] **Step 4: Re-run the scoring tests**

Run:
`python -m pytest tests/supervised/test_scoring.py -q`

Expected: PASS.

---

### Task 3: Make metrics and artifacts portable

**Files:**
- Modify: `src/catfood_unsupervised/supervised/models.py`
- Modify: `src/catfood_unsupervised/supervised/pipeline.py`
- Modify: `src/catfood_unsupervised/supervised/reporting.py`
- Test: `tests/supervised/test_pipeline.py`
- Test: `tests/supervised/test_reporting.py`

The current pipeline already writes the main artifacts, but the saved metrics still carry a stale absolute `input_path`, and the artifact set does not yet include the PNG confusion matrix that `GOAL.md` asks for.

- [ ] **Step 1: Write the failing artifact tests**

```python
def test_run_supervised_pipeline_writes_confusion_matrix_png(tmp_path, supervised_fixture_path):
    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
        random_state=7,
        test_size=0.25,
    )

    assert (tmp_path / "out" / "confusion_matrix.png").exists()
    assert not str(result["metrics"]["input_path"]).startswith("C:\\Users\\wootty\\Downloads")
```

Run:
`python -m pytest tests/supervised/test_pipeline.py tests/supervised/test_reporting.py -q`

Expected: fail until the PNG artifact and portable metadata are in place.

- [ ] **Step 2: Extend the pipeline output contract**

Update `run_supervised_pipeline()` so it:
- writes `confusion_matrix.csv` and `confusion_matrix.png`
- stores workspace-neutral metadata in `metrics_summary.json`
- keeps `classification_report` and `model_comparison` as JSON-safe structures
- remains tolerant to any number of class labels
- computes ROC AUC as binary ROC when there are 2 classes, and multiclass OvR summary metrics when there are 3+ classes

- [ ] **Step 3: Render the confusion matrix image**

Use `matplotlib`/`ConfusionMatrixDisplay` so the PNG is generated locally without adding a heavy export dependency.

- [ ] **Step 4: Refresh report text**

Update the Thai markdown reports so they:
- list the final artifact set accurately
- explain which metric is the source of truth for model selection
- mention the new scoring script for external use

- [ ] **Step 5: Re-run the pipeline and reporting tests**

Run:
`python -m pytest tests/supervised/test_pipeline.py tests/supervised/test_reporting.py -q`

Expected: PASS.

---

### Task 4: Refresh the dashboard and handoff docs

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- Modify: `docs/supervised-team-handoff.md`
- Modify: `README.md` if the top-level run instructions need a supervised section

The supervised tab already shows model comparison, confusion matrix, feature importance, and ROC curves. This task keeps it aligned with the hardened artifacts and adds a clear pointer for the new scoring path.

- [ ] **Step 1: Update the dashboard bundle loader for the revised artifact set**

The loader should still read `metrics_summary.json`, `model_comparison.csv`, `confusion_matrix.csv`, `feature_importance.csv`, and `predictions.csv`, but it should also handle the new portable metadata keys and any multiclass label count.

- [ ] **Step 2: Add a small callout for scoring new customers**

Keep the tab artifact-driven, but add a short panel that points to:
- `scripts/predict_supervised_segment.py`
- the `best_model.pkl` artifact
- the final source-of-truth metrics

- [ ] **Step 3: Keep the visualizations multiclass-safe**

The confusion matrix and ROC plot should continue to render if the class count changes from 2 to 3+ in a later rerun.

- [ ] **Step 4: Update the handoff note**

`docs/supervised-team-handoff.md` should end up with:
- the one command to run the supervised pipeline
- the files that matter for review
- the scoring command for a new customer row
- the dashboard tab to inspect for a quick sanity check

- [ ] **Step 5: Re-run the dashboard test**

Run:
`python -m pytest tests/dashboard/test_tab_supervised.py -q`

Expected: PASS.

---

### Task 5: Final verification

**Files:**
- None expected unless a last-doc tweak is needed

- [ ] **Step 1: Regenerate the supervised outputs from the real cleaned dataset**

Run:
`python scripts/run_supervised_pipeline.py`

Expected:
- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/confusion_matrix.png`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `outputs/supervised/best_model.pkl`
- `reports/supervised/supervised_model_report_th.md`
- `reports/supervised/supervised_owner_memo_th.md`

- [ ] **Step 2: Run the supervised test slice**

Run:
`python -m pytest tests/supervised tests/dashboard/test_tab_supervised.py -q`

Expected: PASS.

- [ ] **Step 3: Open the dashboard and sanity-check the supervised tab**

Confirm the tab still shows:
- model comparison
- confusion matrix
- per-class precision/recall/F1
- feature importance
- ROC plot

- [ ] **Step 4: Mark the plan done**

When the above checks pass, the supervised slice is ready for submission or handoff.

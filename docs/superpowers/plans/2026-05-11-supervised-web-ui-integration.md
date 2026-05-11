# Supervised Web UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the supervised slice into a real end-to-end web flow that can score a new customer row, persist each prediction, and show model quality plus prediction history in the Dash UI.

**Architecture:** Keep the sklearn pipeline as the source of truth for training and inference, but make the saved model artifact carry the preprocessing steps with it so the UI never rebuilds encoders or scalers at runtime. Use Dash callbacks as the backend boundary instead of a second API stack, and store prediction history in a lightweight SQLite database so the supervised tab and the business-insight page can read the same history. Keep the new pieces small and focused: one shared scoring module, one persistence module, and one frontend composition layer.

**Tech Stack:** Python, pandas, scikit-learn, Dash, Plotly, `sqlite3`, `pickle`, pytest.

---

## File Structure

**Create:**
- `src/catfood_unsupervised/supervised/scoring.py`
- `src/catfood_unsupervised/supervised/history_store.py`
- `scripts/predict_supervised_segment.py`
- `src/catfood_unsupervised/dashboard/components/supervised_form.py`
- `src/catfood_unsupervised/dashboard/components/supervised_results.py`
- `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`
- `src/catfood_unsupervised/dashboard/supervised_callbacks.py`
- `tests/supervised/test_scoring.py`
- `tests/supervised/test_history_store.py`
- `tests/dashboard/test_supervised_callbacks.py`
- `tests/dashboard/test_business_insight.py`

**Modify:**
- `src/catfood_unsupervised/supervised/config.py`
- `src/catfood_unsupervised/supervised/schema.py`
- `src/catfood_unsupervised/supervised/data_loading.py`
- `src/catfood_unsupervised/supervised/features.py`
- `src/catfood_unsupervised/supervised/models.py`
- `src/catfood_unsupervised/supervised/pipeline.py`
- `src/catfood_unsupervised/supervised/reporting.py`
- `src/catfood_unsupervised/dashboard/app.py`
- `src/catfood_unsupervised/dashboard/config.py`
- `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- `docs/supervised-team-handoff.md`
- `README.md`

---

### Task 1: Lock the supervised model artifact

**Files:**
- Modify: `src/catfood_unsupervised/supervised/config.py`
- Modify: `src/catfood_unsupervised/supervised/schema.py`
- Modify: `src/catfood_unsupervised/supervised/data_loading.py`
- Modify: `src/catfood_unsupervised/supervised/features.py`
- Modify: `src/catfood_unsupervised/supervised/models.py`
- Modify: `src/catfood_unsupervised/supervised/pipeline.py`
- Test: `tests/supervised/test_pipeline.py`
- Test: `tests/supervised/test_model_artifacts.py`

The goal of this task is to make the saved supervised artifact usable on raw UI input without rebuilding preprocessing by hand. The full fitted `Pipeline` object should be saved as `best_model.pkl`, and the metrics file should publish the exact feature list and class labels used during training.

- [ ] **Step 1: Write the failing artifact tests**

```python
def test_best_model_is_full_pipeline(tmp_path, supervised_fixture_path):
    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
        random_state=7,
        test_size=0.25,
    )

    with (tmp_path / "out" / "best_model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert "class_labels" in result["metrics"]
    assert "feature_columns" in result["metrics"]
```

Run:
`python -m pytest tests/supervised/test_pipeline.py tests/supervised/test_model_artifacts.py -q`

Expected: fail until the artifact contract is explicit and the saved object is the full fitted pipeline.

- [ ] **Step 2: Update the model contract**

Persist the fitted `Pipeline` object with `pickle.dump()` so preprocessing travels with the estimator. Add the following metadata to `metrics_summary.json`:
- `feature_columns`
- `class_labels`
- `best_model_path`
- `scoring_entrypoint`

Keep the metrics JSON-safe and workspace-neutral so the dashboard can reload it from any checkout.

- [ ] **Step 3: Keep feature selection schema-driven**

Make `load_supervised_dataset()` and `build_supervised_feature_frame()` rely on the canonical feature contract in `schema.py` instead of hard-coded positions. The loader should continue filtering `anomaly_flag == 0`, and the feature frame should preserve the stable, canonical column order so the saved pipeline and the UI always see the same schema.

- [ ] **Step 4: Re-run the artifact tests**

Run:
`python -m pytest tests/supervised/test_pipeline.py tests/supervised/test_model_artifacts.py -q`

Expected: PASS.

---

### Task 2: Add a reusable backend scoring and history layer

**Files:**
- Create: `src/catfood_unsupervised/supervised/scoring.py`
- Create: `src/catfood_unsupervised/supervised/history_store.py`
- Create: `scripts/predict_supervised_segment.py`
- Test: `tests/supervised/test_scoring.py`
- Test: `tests/supervised/test_history_store.py`

This task is the business-facing core of the supervised slice. The same scoring path must be usable from the dashboard callback and from a CLI command, and every prediction should be written to a local SQLite history store so the Business Insight page can reuse it later.

- [ ] **Step 1: Write the failing scoring and history tests**

```python
def test_predict_supervised_segment_returns_probabilities(supervised_fixture_path, tmp_path):
    result = run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
        random_state=7,
        test_size=0.25,
    )

    scored = predict_supervised_segment(
        model_path=tmp_path / "out" / "best_model.pkl",
        input_frame=pd.DataFrame([{
            "age_enc": 2,
            "gender_enc": 0,
            "marital_enc": 3,
            "bf_natural": 4,
            "bf_imported": 3,
            "bf_taste": 5,
            "bf_foreign_brand": 2,
            "bf_famous_brand": 4,
            "pkg_premium_look": 5,
            "pkg_cat_image": 4,
            "pkg_food_image": 4,
            "pkg_ingredient_image": 3,
            "pkg_eco_friendly": 3,
            "pkg_source_symbol": 4,
            "pkg_benefit_symbol": 4,
            "pkg_certified": 3,
            "pkg_style_enc": 2,
            "pkg_inf_enc": 1,
        }]),
    )

    assert "predicted_segment" in scored.columns
    assert any(column.startswith("prob_class_") for column in scored.columns)
    assert scored.filter(like="prob_class_").sum(axis=1).iloc[0] == pytest.approx(1.0)
```

```python
def test_prediction_history_round_trips(tmp_path):
    db_path = tmp_path / "prediction_history.sqlite"
    init_history_store(db_path)
    append_prediction_history(
        db_path,
        source="dashboard",
        model_name="random_forest",
        predicted_segment=1,
        probability_map={"prob_class_1": 0.87, "prob_class_2": 0.13},
        input_payload={"age_enc": 2, "gender_enc": 0},
    )

    rows = fetch_recent_predictions(db_path, limit=1)
    assert rows[0]["predicted_segment"] == 1
    assert rows[0]["model_name"] == "random_forest"
```

Run:
`python -m pytest tests/supervised/test_scoring.py tests/supervised/test_history_store.py -q`

Expected: fail until the scorer, history store, and CLI exist.

- [ ] **Step 2: Implement the shared scoring service**

Create `scoring.py` with a small surface that:
- loads the saved `best_model.pkl`
- validates raw input against `FEATURE_COLUMNS`
- converts one-row form payloads and batch CSV input into the same feature frame
- returns `predicted_segment` plus one `prob_class_<label>` column per class
- raises a clear error when required input fields are missing or malformed

Keep this logic free of Dash imports so the CLI and the web UI can both call it.

- [ ] **Step 3: Implement the SQLite history store**

Create `history_store.py` with helpers that:
- initialize a `prediction_history` table on first use
- write timestamped prediction rows with the raw input, predicted label, probabilities, and model name
- fetch the most recent rows for display in the UI
- optionally expose a tiny aggregate summary for the Business Insight page

Use the standard library `sqlite3` module so the repo does not need a new persistence dependency.

- [ ] **Step 4: Add the CLI wrapper**

Create `scripts/predict_supervised_segment.py` with:
- `--model-path`
- `--input-csv`
- `--output-csv`

The script should load the saved pipeline, score the provided rows, write the scored CSV, and optionally append the run to the history database.

- [ ] **Step 5: Re-run the backend tests**

Run:
`python -m pytest tests/supervised/test_scoring.py tests/supervised/test_history_store.py -q`

Expected: PASS.

---

### Task 3: Build the interactive supervised UI

**Files:**
- Modify: `src/catfood_unsupervised/dashboard/app.py`
- Modify: `src/catfood_unsupervised/dashboard/config.py`
- Modify: `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- Modify: `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- Create: `src/catfood_unsupervised/dashboard/components/supervised_form.py`
- Create: `src/catfood_unsupervised/dashboard/components/supervised_results.py`
- Create: `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`
- Create: `src/catfood_unsupervised/dashboard/supervised_callbacks.py`
- Test: `tests/dashboard/test_tab_supervised.py`
- Test: `tests/dashboard/test_supervised_callbacks.py`
- Test: `tests/dashboard/test_business_insight.py`

This task turns the static supervised charts into a real user flow. The supervised page should accept a new customer row, run the shared scorer, show the predicted segment and probabilities, and read back recent prediction history from SQLite. The business-insight page should consume that same history store so the dashboard reflects actual usage, not only training-time outputs.

- [ ] **Step 1: Write the failing UI tests**

```python
def test_render_supervised_tab_shows_prediction_form():
    tab = render_supervised_tab(bundle)
    rendered_text = " ".join(_collect_text(tab))

    assert "Predict" in rendered_text
    assert "Confidence" in rendered_text
    assert "Best Model" in rendered_text
    assert "Recent predictions" in rendered_text
```

```python
def test_business_insight_reads_prediction_history(tmp_path):
    db_path = tmp_path / "history.sqlite"
    init_history_store(db_path)
    append_prediction_history(
        db_path,
        source="dashboard",
        model_name="random_forest",
        predicted_segment=1,
        probability_map={"prob_class_0": 0.18, "prob_class_1": 0.82},
        input_payload={
            "age_enc": 2,
            "gender_enc": 0,
            "marital_enc": 3,
            "bf_natural": 4,
            "bf_imported": 3,
        },
    )
    tab = render_business_insight_tab(history_db_path=db_path)
    rendered_text = " ".join(_collect_text(tab))

    assert "Prediction history" in rendered_text
    assert "random_forest" in rendered_text
```

Run:
`python -m pytest tests/dashboard/test_tab_supervised.py tests/dashboard/test_supervised_callbacks.py tests/dashboard/test_business_insight.py -q`

Expected: fail until the new form, callback, and history-driven page exist.

- [ ] **Step 2: Add the supervised form and result components**

Create a small field-spec module or form component that maps friendly labels to the canonical supervised feature names. The supervised tab should show:
- an input form for age, gender, marital status, buy factors, and packaging features
- a `Predict` button
- a predicted segment result card
- a confidence/probability visualization
- a compact model-metrics strip with Accuracy, Macro F1, and Weighted F1

Keep the unsupervised indicators visually separate from the supervised indicators so the page never mixes the two stories.

- [ ] **Step 3: Wire the Dash callback**

Create `supervised_callbacks.py` and have `app.py` register it. The callback should:
- read form values
- call `predict_supervised_segment()`
- append the prediction to SQLite
- refresh the recent history panel
- return a clean error message when a field is missing or invalid

This keeps the data-processing logic out of the presentation file and makes the callback testable.

- [ ] **Step 4: Add the Business Insight page**

Add `tab_business_insight.py` and update `dashboard/config.py` so the app can show a prediction-history view. This page should summarize:
- the number of predictions by segment
- the recent prediction rows
- any simple trend or mix chart that helps business users see how the supervised tool is being used

If the repository keeps the tab-based layout, this page can live alongside the supervised tab rather than requiring a full router rewrite.

- [ ] **Step 5: Re-run the UI tests**

Run:
`python -m pytest tests/dashboard/test_tab_supervised.py tests/dashboard/test_supervised_callbacks.py tests/dashboard/test_business_insight.py -q`

Expected: PASS.

---

### Task 4: Document, publish, and verify end to end

**Files:**
- Modify: `docs/supervised-team-handoff.md`
- Create: `README.md`
- Modify: `src/catfood_unsupervised/dashboard/app.py` if the final dashboard title or tab order changes during integration

This task closes the loop for the project deliverables: the repo should explain how to train, score a new customer, and inspect results without digging through source code.

- [ ] **Step 1: Refresh the handoff docs**

Update `docs/supervised-team-handoff.md` to include:
- the one command that runs the supervised pipeline
- the `best_model.pkl` path
- the CLI command for scoring a new customer row
- the SQLite history path
- the dashboard tab or page to inspect for a quick sanity check

Create `README.md` if it is still missing and add a concise run guide for the supervised flow.

- [ ] **Step 2: Regenerate outputs**

Run:
`python scripts/run_supervised_pipeline.py`

Expected outputs:
- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/confusion_matrix.png`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `outputs/supervised/best_model.pkl`
- `reports/supervised/supervised_model_report_th.md`
- `reports/supervised/supervised_owner_memo_th.md`

- [ ] **Step 3: Run the supervised and dashboard test slices**

Run:
`python -m pytest tests/supervised tests/dashboard -q`

Expected: PASS.

- [ ] **Step 4: Do a live browser sanity check**

Open the dashboard in the browser and confirm that the supervised tab:
- loads the form
- accepts a sample customer row
- returns a prediction and probability
- records the prediction in history
- keeps the supervised metrics separate from the unsupervised ones

Treat any browser console error as a blocker until it is explained or fixed.

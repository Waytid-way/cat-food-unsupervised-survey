# Presentation Regeneration and Code Consistency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the single presentation deck for EDA, Unsupervised, Supervised, and Business, then verify that every slide claim, code snippet, and metric matches the current merged codebase and report artifacts.

**Architecture:** Treat `presentation/full-presentation-slides.md` as the human-readable source of truth and keep `presentation/full-presentation-slides.txt` as the plain-text mirror. The deck should remain a 20-slide story arc, but every numeric claim must come from current outputs (`outputs/metrics_summary.json`, `outputs/supervised/metrics_summary.json`) and every process claim must point to real functions in the repo. The audit phase should be explicit: slide inventory first, code-reference validation second, deck rewrite third, then a final consistency pass against `pytest` and `git diff`.

**Tech Stack:** Markdown, plain text export, Python, pandas, scikit-learn, Dash, Plotly, pytest, rg, git.

---

## File Structure

**Modify:**
- `presentation/full-presentation-slides.md`
- `presentation/full-presentation-slides.txt`

**Read:**
- `outputs/metrics_summary.json`
- `outputs/segment_profiles.csv`
- `reports/unsupervised_report_th.md`
- `reports/unsupervised_owner_memo_th.md`
- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `reports/supervised/supervised_model_report_th.md`
- `reports/supervised/supervised_owner_memo_th.md`
- `src/catfood_unsupervised/data_loading.py`
- `src/catfood_unsupervised/features.py`
- `src/catfood_unsupervised/pipeline.py`
- `src/catfood_unsupervised/reporting.py`
- `src/catfood_unsupervised/supervised/config.py`
- `src/catfood_unsupervised/supervised/schema.py`
- `src/catfood_unsupervised/supervised/features.py`
- `src/catfood_unsupervised/supervised/models.py`
- `src/catfood_unsupervised/supervised/pipeline.py`
- `src/catfood_unsupervised/supervised/reporting.py`
- `src/catfood_unsupervised/supervised/scoring.py`
- `src/catfood_unsupervised/supervised/history_store.py`
- `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- `src/catfood_unsupervised/dashboard/supervised_callbacks.py`
- `src/catfood_unsupervised/dashboard/components/supervised_form.py`
- `src/catfood_unsupervised/dashboard/components/supervised_results.py`
- `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`
- `scripts/predict_supervised_segment.py`
- `tests/supervised/test_scoring.py`
- `tests/supervised/test_reporting.py`
- `tests/dashboard/test_tab_supervised.py`
- `tests/dashboard/test_business_insight.py`
- `tests/dashboard/test_supervised_callbacks.py`

---

### Task 1: Lock the source-of-truth matrix before editing the deck

**Files:**
- Read: `outputs/metrics_summary.json`
- Read: `outputs/supervised/metrics_summary.json`
- Read: `reports/unsupervised_report_th.md`
- Read: `reports/supervised/supervised_model_report_th.md`
- Read: `src/catfood_unsupervised/data_loading.py`
- Read: `src/catfood_unsupervised/features.py`
- Read: `src/catfood_unsupervised/pipeline.py`
- Read: `src/catfood_unsupervised/reporting.py`
- Read: `src/catfood_unsupervised/supervised/scoring.py`
- Read: `src/catfood_unsupervised/supervised/history_store.py`
- Read: `src/catfood_unsupervised/supervised/reporting.py`
- Read: `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- Read: `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- Read: `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`

The goal of this task is to prevent slide drift by mapping each slide family to the exact artifact or function that supports it.

**Slide-to-source matrix**

| Slide family | Allowed claims | Source of truth |
|---|---|---|
| Slides 1-5: EDA | Raw row counts, completed-response count, demographic summary, buy-factor summary, top-3 choices, correlation signal | `outputs/metrics_summary.json`, `reports/unsupervised_report_th.md`, `src/catfood_unsupervised/reporting.py` |
| Slides 6-9: Unsupervised | PCA, K-Means sweep, hierarchical validation, anomaly detection, segment personas | `outputs/metrics_summary.json`, `outputs/segment_profiles.csv`, `src/catfood_unsupervised/pipeline.py`, `src/catfood_unsupervised/features.py`, `src/catfood_unsupervised/reporting.py` |
| Slides 10-15: Supervised | Segment target, leakage control, feature contract, model training, metrics, scoring flow, dashboard bundle, SQLite history | `src/catfood_unsupervised/supervised/schema.py`, `src/catfood_unsupervised/supervised/features.py`, `src/catfood_unsupervised/supervised/models.py`, `src/catfood_unsupervised/supervised/pipeline.py`, `src/catfood_unsupervised/supervised/scoring.py`, `src/catfood_unsupervised/supervised/history_store.py`, `src/catfood_unsupervised/dashboard/supervised_data_loader.py`, `src/catfood_unsupervised/dashboard/components/tab_supervised.py`, `src/catfood_unsupervised/dashboard/supervised_callbacks.py` |
| Slides 16-18: Business | How prediction history is used, what actions each segment gets, how the packaging strategy translates to campaign language | `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`, `src/catfood_unsupervised/dashboard/components/supervised_results.py`, `reports/supervised/supervised_owner_memo_th.md` |
| Slides 19-20: Backup | Metric definitions and code appendix only | the same source files above; no new logic |

- [ ] **Step 1: Verify the current deck size and headline inventory**

Run:
`Select-String -Path .\presentation\full-presentation-slides.md -Pattern '^## Slide \d+' | Measure-Object`

Expected: `Count : 20`

Run:
`Select-String -Path .\presentation\full-presentation-slides.txt -Pattern '^## Slide \d+' | Measure-Object`

Expected: `Count : 20`

- [ ] **Step 2: Verify the code entrypoints that the slides will reference**

Run:
`rg -n "def load_raw_export|def filter_completed_responses|def build_vote_features|def ipsatize_rows|def run_pipeline|def write_reports_from_output_dir|def predict_supervised_segment|def fetch_recent_prediction_history|def render_supervised_tab|def render_business_insight_tab|def load_supervised_dashboard_bundle|def score_and_store_supervised_row" src scripts`

Expected: every function listed above is present exactly once in the repo and the search returns the real implementation files.

- [ ] **Step 3: Lock the numeric anchors used in the deck**

Use the current outputs as the only allowed numbers in the presentation:
- raw survey rows: `167`
- completed top-3 rows: `148`
- unsupervised PCA components: `8`
- unsupervised explained variance: `0.686`
- best unsupervised `k`: `2`
- silhouette: `0.193`
- Davies-Bouldin: `2.032`
- cophenetic correlation: `0.473`
- anomaly count: `26`
- anomaly rate: `17.6%`
- supervised training rows: `122`
- best supervised model: `random_forest`
- supervised accuracy: `0.88`
- supervised macro F1: `0.797`
- supervised weighted F1: `0.864`
- supervised ROC AUC: `0.759`

These values must appear in the slides only if they are still present in the current `outputs/*` and `reports/*` files.

---

### Task 2: Rewrite the combined deck so the story and code agree

**Files:**
- Modify: `presentation/full-presentation-slides.md`
- Modify: `presentation/full-presentation-slides.txt`

This task keeps the existing 20-slide structure but refreshes the wording so the slide claims match the merged supervised workflow and the current report artifacts. The content must stay balanced at roughly `70/30` between business/storytelling and technical proof.

- [ ] **Step 1: Rewrite Slides 1-5 as the EDA story**

Keep these claims:
- Project framing: cat food packaging survey turned into an AI-driven marketing campaign system
- Data overview: raw export, completed responses, and why the survey is worth analyzing
- Cleaning step: promote the real header row, remove empty export rows, keep only completed top-3 responses
- Findings: demographic distribution, buy-factor summary, top-3 packaging rankings, correlation signal

Use the exact code path names in the snippet section:
```python
raw_df = load_raw_export(data_path)
top3_column = raw_df.columns[TOP3_COLUMN_INDEX]
completed_df = filter_completed_responses(raw_df, top3_column=top3_column)
```

The slide text must explain that EDA is not a separate toy exercise; it is the base dataset for both unsupervised segmentation and supervised training.

- [ ] **Step 2: Rewrite Slides 6-9 as the unsupervised pipeline story**

Keep these claims:
- unsupervised comes first because the customer groups are not known in advance
- pipeline order: feature encoding -> ipsatization -> imputation -> PCA -> K-Means -> hierarchical validation -> Isolation Forest
- PCA keeps enough structure for clustering while reducing dimensionality
- K-Means selects `k=2`
- the clusters are useful but not perfectly separated
- anomaly detection finds a small but important minority of outliers
- the final personas are business-friendly and explainable

Use the exact code path names in the snippet section:
```python
vote_features = build_vote_features(
    rankings,
    rank_columns=rankings.columns.tolist(),
    option_count=OPTION_COUNT,
)

ipsatized_option_ratings = pd.DataFrame(
    ipsatize_rows(option_ratings.to_numpy()),
    index=completed_df.index,
    columns=[f"{column}_ips" for column in option_ratings.columns],
).fillna(0.0)
```

The slide wording must stay consistent with `src/catfood_unsupervised/pipeline.py`, especially the `run_pipeline()` flow and the final metrics file names.

- [ ] **Step 3: Rewrite Slides 10-15 as the supervised system story**

Keep these claims:
- supervised uses `outputs/clean_dataset_with_segments.csv`
- rows with `anomaly_flag = 1` are excluded from training
- `segment` is the target
- leakage columns are excluded from the model contract
- the feature frame is canonical and stable
- the training suite compares Logistic Regression, Random Forest, Gradient Boosting, and RBF SVM
- `macro_f1` is the primary selection metric
- the scoring path saves each prediction to SQLite
- the dashboard uses the same bundle as the CLI scoring path

Use the exact code path names in the snippet section:
```python
prediction_frame = predict_supervised_segment(bundle.model_path, input_payload)
append_prediction_history(
    bundle.history_db_path,
    source="dashboard",
    model_name=str(bundle.metrics.get("best_model_name", bundle.model_path.stem)),
    predicted_segment=predicted_segment,
    probability_map=probability_map,
    input_payload=input_payload,
)
```

The slide text must name the real functions:
- `build_supervised_field_specs`
- `predict_supervised_segment`
- `load_supervised_dashboard_bundle`
- `score_and_store_supervised_row`
- `render_supervised_tab`
- `render_business_insight_tab`

Also keep the dashboard claims aligned with the current UI layout: prediction form, result panel, recent history, model comparison, confusion matrix, feature importance, ROC, and business insight history.

- [ ] **Step 4: Rewrite Slides 16-18 as the business story**

Keep these claims:
- Segment 1 is the premium-quality audience
- Segment 2 is the main-market audience
- opt03 is the most defensible packaging direction
- the business use case is not just classification, but packaging and campaign action
- the dashboard and SQLite history make the model usable for non-technical stakeholders

The narrative must match `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`, which reads prediction history and summarizes usage.

- [ ] **Step 5: Keep Slides 19-20 as backup only**

Slide 19 must explain the metrics in plain language:
- silhouette
- Davies-Bouldin
- cophenetic correlation
- accuracy
- macro F1
- weighted F1
- ROC AUC

Slide 20 must stay a code appendix with only the exact function names used in the deck and no extra invented code paths.

- [ ] **Step 6: Sync the text export**

`presentation/full-presentation-slides.txt` must mirror the same slide ordering, slide titles, and claims as the Markdown deck.
It does not need to be byte-identical, but it must preserve:
- all 20 slide headings
- the same slide order
- the same metrics
- the same function names
- the same business takeaway

---

### Task 3: Audit every code snippet and slide claim against the real repo

**Files:**
- Read: `presentation/full-presentation-slides.md`
- Read: `presentation/full-presentation-slides.txt`
- Read: `src/catfood_unsupervised/data_loading.py`
- Read: `src/catfood_unsupervised/features.py`
- Read: `src/catfood_unsupervised/pipeline.py`
- Read: `src/catfood_unsupervised/reporting.py`
- Read: `src/catfood_unsupervised/supervised/schema.py`
- Read: `src/catfood_unsupervised/supervised/features.py`
- Read: `src/catfood_unsupervised/supervised/models.py`
- Read: `src/catfood_unsupervised/supervised/pipeline.py`
- Read: `src/catfood_unsupervised/supervised/reporting.py`
- Read: `src/catfood_unsupervised/supervised/scoring.py`
- Read: `src/catfood_unsupervised/supervised/history_store.py`
- Read: `src/catfood_unsupervised/dashboard/supervised_data_loader.py`
- Read: `src/catfood_unsupervised/dashboard/components/tab_supervised.py`
- Read: `src/catfood_unsupervised/dashboard/components/tab_business_insight.py`

This task is the actual consistency check. It should catch wrong function names, stale artifact names, stale metrics, and any slide that implies a behavior the code does not have.

- [ ] **Step 1: Verify every snippet name exists in code**

Run:
`rg -n "load_raw_export|filter_completed_responses|build_vote_features|ipsatize_rows|run_pipeline|write_reports_from_output_dir|build_supervised_field_specs|predict_supervised_segment|append_prediction_history|load_supervised_dashboard_bundle|score_and_store_supervised_row|render_supervised_tab|render_business_insight_tab" src presentation`

Expected: every function name used in the slides is backed by a real definition in `src/`.

- [ ] **Step 2: Verify report filenames and artifact names**

Run:
`rg -n "metrics_summary.json|model_comparison.csv|confusion_matrix.csv|feature_importance.csv|predictions.csv|best_model.pkl|supervised_model_report_th.md|supervised_owner_memo_th.md|unsupervised_report_th.md|unsupervised_owner_memo_th.md" presentation src reports`

Expected: the deck references only files that already exist in the repo or are written by the current pipeline.

- [ ] **Step 3: Run the full test suite after the slide edits**

Run:
`python -m pytest -q`

Expected: `58 passed, 1 warning`

The warning is the existing scikit-learn convergence warning in the supervised test suite and is acceptable unless the underlying model behavior changes.

- [ ] **Step 4: Inspect the diff for accidental scope creep**

Run:
`git diff --stat`

Expected: only `presentation/full-presentation-slides.md` and `presentation/full-presentation-slides.txt` change unless the audit uncovered a real upstream bug in the source code.

Run:
`git diff --check`

Expected: no whitespace or line-ending errors.

---

### Task 4: Finalize the deck and hand off a clean consistency report

**Files:**
- Modify: `presentation/full-presentation-slides.md`
- Modify: `presentation/full-presentation-slides.txt`
- Report: final status only

The final handoff should say whether the slide deck is now synchronized with the merged codebase, which metrics were locked, and whether anything remained ambiguous.

- [ ] **Step 1: Recount the slide inventory one last time**

Run:
`Select-String -Path .\presentation\full-presentation-slides.md -Pattern '^## Slide \d+' | Measure-Object`

Expected: `Count : 20`

Run:
`Select-String -Path .\presentation\full-presentation-slides.txt -Pattern '^## Slide \d+' | Measure-Object`

Expected: `Count : 20`

- [ ] **Step 2: Commit the deck refresh**

Run:
`git add presentation/full-presentation-slides.md presentation/full-presentation-slides.txt`

Run:
`git commit -m "docs: regenerate combined presentation and audit slide-code consistency"`

Expected: one focused commit that only covers the regenerated deck.

- [ ] **Step 3: Report the result back in plain language**

The report must answer these questions:
- Is the combined deck still 20 slides?
- Do the slide claims match the merged code?
- Do the code snippets refer to real functions?
- Do the numeric claims come from current outputs and reports?
- Did the test suite stay green after the deck refresh?

---

## Self-Review

**1. Spec coverage**
- Combined deck regeneration: covered by Task 2 and Task 4.
- EDA, Unsupervised, Supervised, and Business coverage: covered by the slide-family matrix and Task 2.
- Code snippet validation: covered by Task 1 and Task 3.
- Consistency check against current code and reports: covered by Task 3.
- Final validation and commit: covered by Task 4.

**2. Placeholder scan**
- No `TBD`, `TODO`, or vague "add appropriate" placeholders are present.
- Every task names the exact files, function names, commands, and expected outputs.

**3. Type and name consistency**
- `render_supervised_tab` is used consistently instead of a guessed wrapper name.
- `render_business_insight_tab` is used consistently for the business tab.
- `predict_supervised_segment`, `append_prediction_history`, `load_supervised_dashboard_bundle`, and `score_and_store_supervised_row` are named the same way in the plan and the code references.


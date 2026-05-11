# Full Presentation - EDA + Unsupervised + Supervised + Business
## Cat Food Packaging Survey | AI-Driven Marketing Campaign System
**Total slides: 20**

---

## Slide 1 - Title

**Project:** Cat Food Packaging Survey

**Subtitle:** AI-Driven Marketing Campaign System for EDA, Segmentation, Prediction, and Business Decision

**One-line story:** turn survey answers into customer segments, then turn segments into a reusable scoring system for marketing.

**Team / course / date:** add final presentation metadata here.

---

## Slide 2 - Problem Statement

**Business question**
- Which packaging design fits which customer group?
- Can we turn survey insights into a system that scores new customers automatically?

**System promise**
- EDA tells us what the data looks like.
- Unsupervised learning finds customer segments and anomalies.
- Supervised learning turns those segments into a prediction model.
- Business insight translates model output into campaign actions.

**End-to-end flow**
```text
Raw survey -> EDA -> Unsupervised segmentation -> Supervised classifier -> Dashboard + business action
```

---

## Slide 3 - Data Overview

**Source**
- Google Form export from the cat food packaging survey
- Raw survey rows loaded: **167**
- Completed top-3 responses used for modeling: **148**

**What is inside the survey**
- Demographics
- Buy factors
- Packaging importance
- Option ratings for 10 packaging choices

**Feature footprint**
- 50 Likert rating columns
- 5 buy-factor columns
- 8 packaging-importance columns
- 5 demographic columns

**EDA goal**
- understand who answered
- check response quality
- see which packaging choices and purchase factors dominate

---

## Slide 4 - EDA Cleaning

**Why this step matters**
- The raw Google Form export contains metadata rows and incomplete responses.
- We only want respondents who completed the top-3 choice question.

**Key cleaning rule**
- Promote the true header row
- Remove empty exported rows
- Keep only completed top-3 responses

**Code snippet**
```python
raw_df = load_raw_export(data_path)
top3_column = raw_df.columns[TOP3_COLUMN_INDEX]
completed_df = filter_completed_responses(raw_df, top3_column=top3_column)
```

**Effect**
- 167 raw rows -> 148 valid responses
- This becomes the shared base for both unsupervised analysis and supervised training

---

## Slide 5 - EDA Findings

**Demographic snapshot**
- Female: **73.0%**
- Age 30-39: **41.9%**
- Single / not married: **35.1%**

**Purchase drivers**
- Taste: **4.43**
- Natural ingredients: **4.26**
- Famous brand: **4.06**

**Top packaging choices**
- opt03: **251**
- opt02: **138**
- opt01: **114**

**Correlation signal**
- opt03 - opt04: **r = 0.625**
- opt01 - opt02: **r = 0.568**
- opt05 - opt09: **r = 0.635**

**Interpretation**
- opt03 is the clear front-runner.
- Taste and natural ingredients dominate the decision lens.
- Preferences are structured, not random, so segmentation is worth doing.

---

## Slide 6 - Unsupervised Overview

**Why unsupervised first**
- We do not know the true customer groups before modeling.
- The survey is designed to discover patterns, not only predict labels.

**Pipeline**
```text
Encode -> Ipsatize -> Impute -> PCA -> K-Means -> Hierarchical validation -> Isolation Forest
```

**What this gives us**
- customer segments
- anomaly flags
- a business-friendly persona view

**Reference outputs**
- `outputs/metrics_summary.json`
- `outputs/segment_profiles.csv`
- `reports/unsupervised_report_th.md`

---

## Slide 7 - Unsupervised Feature Engineering

**What the code does**
- Convert top-3 rankings into weighted vote features
- Reduce response-style bias with ipsatization
- Fill missing buy-factor values before clustering
- Build one merged feature matrix for clustering and anomaly detection

**Code snippet**
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

**Why ipsatization matters**
- It subtracts each respondent's own mean rating.
- That helps separate true preference from the habit of giving high or low scores to everything.

---

## Slide 8 - Unsupervised Evaluation

**PCA**
- Input features: 50 Likert ratings
- Selected components: **8**
- Total explained variance: **68.6%**

**K-Means sweep**
- Best `k = 2`
- Silhouette score: **0.193**
- Davies-Bouldin: **2.032**

**Hierarchical validation**
- Ward linkage
- Cophenetic correlation: **0.473**

**Interpretation**
- The clusters are usable, but not sharply separated.
- This is a pragmatic segmentation, not a mathematically perfect split.

---

## Slide 9 - Segments & Anomalies

**Final segments**

| Segment | Persona | n | Share |
|---|---:|---:|---:|
| 1 | Premium Quality Seekers | 36 | 24.3% |
| 2 | Main Market | 112 | 75.7% |

**Segment 1**
- more quality / design / natural-ingredient oriented
- stronger premium signal
- better fit for premium packaging language

**Segment 2**
- larger mass-market base
- more practical and trust-driven
- better fit for clear, simple, confidence-building packaging

**Anomaly detection**
- Isolation Forest found **26 anomalies**
- Anomaly rate: **17.6%**

**Caution**
- The segment boundary is useful, but there is still overlap.

---

## Slide 10 - Bridge to Supervised

**Why supervised is needed**
- Unsupervised tells us the group structure.
- Supervised lets us score a new customer row instantly.

**Training data**
- `outputs/clean_dataset_with_segments.csv`
- Remove `anomaly_flag = 1`
- Target column: `segment`

**What must be avoided**
- Leakage columns like `vote_*`, `PC*`, `anomaly_score`
- Top-3 ranking columns as direct target leaks

**Business result**
- a reusable classifier for dashboard scoring and batch prediction

---

## Slide 11 - Supervised Feature Contract

**Feature policy**
- Only approved survey features can enter the model.
- Missing values are handled as `__missing__`.
- Everything is converted to string for one-hot encoding.

**Code snippet**
```python
feature_columns = get_supervised_feature_columns(df)
X = df.loc[:, feature_columns].copy().fillna("__missing__").astype(str)
y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
```

**Why this matters**
- The model stays consistent across training, scoring, and dashboard input.
- The same feature contract prevents accidental leakage.

**Model input shape**
- 18 supervised features
- 2 classes: segment 1 and segment 2

---

## Slide 12 - Supervised Training Workflow

**Training setup**
- Clean rows after anomaly removal: **122**
- Holdout split uses stratification to preserve class balance
- Candidate models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - RBF SVM

**Model selection rule**
- Primary: `macro F1`
- Then: `weighted F1`
- Then: `accuracy`

**Code snippet**
```python
comparison, fitted_models, predictions = evaluate_model_suite(
    model_builders,
    X_train,
    y_train,
    X_test,
    y_test,
)
best_model_name = str(comparison.iloc[0]["model_name"])
```

**Design choice**
- Logistic, Random Forest, and SVM use `class_weight="balanced"` where appropriate.

---

## Slide 13 - Supervised Results

**Best model**
- `random_forest`
- Accuracy: **0.88**
- Macro F1: **0.797**
- Weighted F1: **0.864**
- ROC AUC: **0.759**

**Model comparison**

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.88 | 0.797 | 0.864 |
| Gradient Boosting | 0.80 | 0.762 | 0.811 |
| SVM (RBF) | 0.72 | 0.636 | 0.727 |
| Logistic Regression | 0.52 | 0.500 | 0.552 |

**Holdout confusion matrix**
- Actual segment 1: 3 correct, 3 missed
- Actual segment 2: 19 correct, 0 false positive

**Per-class interpretation**
- Segment 1 recall is only **0.50**
- Segment 2 recall is **1.00**
- The model is stronger on the majority class than the minority class

---

## Slide 14 - Scoring Flow

**What happens when we score a new row**
- Load the trained model artifact
- Prepare the input frame with the same feature contract
- Predict the segment and probability distribution
- Save the result to SQLite for traceability

**Code snippet**
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

**CLI support**
- `scripts/predict_supervised_segment.py`
- Batch scoring with `--model-path`, `--input-csv`, `--output-csv`, and optional `--history-db`

---

## Slide 15 - Dashboard Usage

**Supervised tab includes**
- prediction form
- result card with confidence chart
- per-class metrics table
- model comparison chart
- confusion matrix
- feature importance
- ROC curve
- recent prediction history

**How the dashboard loads**
```python
bundle = load_supervised_dashboard_bundle(output_dir)
render_supervised_tab(bundle, recent_history=...)
```

**Why this matters**
- Business users can test a row, inspect the result, and review the usage trail in one place.

---

## Slide 16 - Business Strategy

**Segment 1 - Premium Quality Seekers**
- Lead with premium look and feel
- Use stronger natural / quality cues
- Make the packaging look trustworthy, designed, and distinctive

**Segment 2 - Main Market**
- Keep the message simple and clear
- Emphasize taste, trust, and value
- Use packaging that is easy to understand at a glance

**Packaging implication**
- opt03 is the core design direction because it wins the top-3 vote ranking across both segments.

**Campaign implication**
- keep one main design, then tune the message layer for premium vs mass-market audiences

---

## Slide 17 - Limitations

**Unsupervised limitations**
- Silhouette score is only **0.193**
- Davies-Bouldin is **2.032**
- Cophenetic correlation is **0.473**

**What this means**
- The clusters are useful for business storytelling, but they are not perfectly separated natural groups.

**Supervised limitations**
- Segment 1 recall is only **0.50**
- The minority class still needs more examples

**Recommended wording**
- call this a pragmatic segmentation system, not a perfect cluster boundary

---

## Slide 18 - Final Takeaway

**What the project delivers**
- EDA gives a clean read on survey behavior.
- Unsupervised learning turns survey preferences into two actionable personas.
- Supervised learning operationalizes the result into a scoring model.
- The dashboard connects the model to business usage and audit history.

**Final recommendation**
- Use opt03 as the main design direction.
- Tailor communication by segment.
- Collect more premium examples in the next survey round to improve minority-class recall.

---

## Slide 19 - Backup: Metric Definitions

**Unsupervised metrics**
- Silhouette: how well a sample fits its cluster vs others
- Davies-Bouldin: lower is better; measures compactness and separation
- Cophenetic correlation: how well hierarchical clustering preserves distances

**Supervised metrics**
- Accuracy: overall correct predictions
- Macro F1: average F1 across classes with equal class weight
- Weighted F1: F1 weighted by support
- ROC AUC: probability ranking quality for binary classification

---

## Slide 20 - Backup: Code Appendix

**EDA / cleaning**
```python
raw_df = load_raw_export(data_path)
completed_df = filter_completed_responses(raw_df, top3_column=top3_column)
```

**Unsupervised orchestration**
```python
feature_bundle = _build_feature_bundle(completed_df, top3_column=top3_column)
pca_bundle = _run_pca_pipeline(feature_bundle["ipsatized_option_ratings"])
```

**Supervised scoring**
```python
prediction_frame = predict_supervised_segment(model_path, input_data)
```

**Dashboard state**
```python
bundle = load_supervised_dashboard_bundle(output_dir)
```

**Use this slide only if the panel asks for the exact code path.**

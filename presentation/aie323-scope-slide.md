# AIE323 Scope Slide - Supervised ML

**Project Name:** AIE323 Supervised ML - AI-Driven Marketing Campaign System (Cat Food Packaging Survey)

**Objective**
- Predict `segment` for a new customer from survey answers.
- Convert the unsupervised result into a reusable supervised classifier.
- Support dashboard scoring and business review.

**Scope**
- Input: `outputs/clean_dataset_with_segments.csv`
- Remove `anomaly_flag = 1` before training.
- Target: `segment`
- Use only approved supervised features.
- Do not use leakage columns such as `vote_*`, `PC*`, or `anomaly_score`.
- Models: Logistic Regression, Random Forest, Gradient Boosting, RBF SVM.
- Outputs: `best_model.pkl`, `metrics_summary.json`, `model_comparison.csv`, `confusion_matrix.csv`, `feature_importance.csv`, `predictions.csv`.
- UI: Dash supervised tab, prediction form, and SQLite history.

**Methodology**
1. Load clean segmented data.
2. Build the feature frame from approved survey columns.
3. Split train/test with stratification.
4. Train 4 candidate models.
5. Select the best model by macro F1.
6. Generate confusion matrix, per-class metrics, feature importance, and ROC AUC.
7. Expose the scoring form in Dash and save each prediction to SQLite.

**ขั้นตอนเก็บข้อมูล**
- Survey data comes from the Google Form export.
- The unsupervised pipeline creates segment labels and anomaly flags.
- The supervised pipeline trains only on clean rows.
- Dashboard options are populated from values observed in the dataset.

**วิธีวัด model + ผลทดลอง**
- Use a holdout split for evaluation.
- Measure Accuracy, Macro F1, Weighted F1, and per-class precision/recall/F1.
- Select the best model by macro F1.
- The resulting model is ready for customer scoring in the dashboard.

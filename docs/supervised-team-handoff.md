# Supervised Team Handoff

## What this pipeline does

- Uses `outputs/clean_dataset_with_segments.csv` as the supervised training source.
- Filters out rows with `anomaly_flag = 1` before training.
- Predicts the unsupervised `segment` label with four models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - RBF SVM
- Writes model artifacts and Thai-language reports for business review.

## How to run it

```bash
python scripts/run_supervised_pipeline.py
```

## Source of truth files

- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `outputs/supervised/best_model.pkl`

## Reports for business users

- `reports/supervised/supervised_model_report_th.md`
- `reports/supervised/supervised_owner_memo_th.md`

## Dashboard check

Open the dashboard and inspect the `Supervised` tab.

## Key metrics to trust

- `best_model_name`
- `best_model_accuracy`
- `best_model_macro_f1`
- `best_model_weighted_f1`
- `classification_report`

## Notes for the next engineer

- Keep `segment` as the target variable for classification.
- Do not use `segment`, `anomaly_flag`, `top3_rank_*`, `vote_*`, or `PC*` as features.
- The current dataset has two segments, but the code should stay tolerant to a future rerun with more classes.

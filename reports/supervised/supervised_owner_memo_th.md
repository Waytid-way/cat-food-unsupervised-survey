# Supervised Owner Memo

- Source of truth model: `random_forest`
- Accuracy on holdout set: 0.880
- Macro F1 on holdout set: 0.797
- Use the Supervised dashboard tab to review model comparison, confusion matrix, and feature importance.
- If you export the model for external use, keep `best_model.pkl` and `metrics_summary.json` together.

## What the metrics mean
- If the confusion matrix is skewed in one class, the segment labels may still overlap and feature engineering should be revisited.
- If feature importance concentrates on a few columns, the survey contains strong signals and the model can be simplified later.

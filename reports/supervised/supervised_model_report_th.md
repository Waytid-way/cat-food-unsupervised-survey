# Supervised Learning Model Report

## Executive Summary
- Training rows used: 122
- Encoded feature count: 18
- Best model: random_forest
- Accuracy: 0.880
- Macro F1: 0.797
- Weighted F1: 0.864
- ROC AUC: 0.588

## Model Justification
- Selected `random_forest` because it ranked first on macro F1, weighted F1, and accuracy in the holdout comparison.
- Holdout performance for the selected model: accuracy 0.880, macro F1 0.797, weighted F1 0.864.
- Anomaly rows were excluded before training, so the comparison is based on clean segment labels only.
- The model suite uses the approved supervised feature contract and avoids leakage columns.

## Model Comparison
| model | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- |
| random_forest | 0.880 | 0.797 | 0.864 |
| gradient_boosting | 0.760 | 0.671 | 0.760 |
| svm_rbf | 0.720 | 0.636 | 0.727 |
| logistic_regression | 0.680 | 0.603 | 0.694 |

## Per-class metrics
| class | precision | recall | f1_score | support |
| --- | --- | --- | --- | --- |
| Segment 1 | 1.000 | 0.500 | 0.667 | 6 |
| Segment 2 | 0.864 | 1.000 | 0.927 | 19 |

## Confusion Matrix
|  | pred_1 | pred_2 |
| --- | --- | --- |
| actual_1 | 3 | 3 |
| actual_2 | 0 | 19 |

## Feature Importance
| feature | importance_mean | importance_std |
| --- | --- | --- |
| บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพแมว] | 0.0960 | 0.0265 |
| บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพวัตถุดิบและส่วนผสมจริงให้เห็น] | 0.0640 | 0.0265 |
| สำหรับบรรจุภัณฑ์อาหารแมว คุณชอบภาพแบบใด | 0.0400 | 0.0000 |
| บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์มีภาพอาหารเม็ด รูปทรงของอาหารเม็ดจริงให้เห็น] | 0.0320 | 0.0160 |
| บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม] | 0.0320 | 0.0349 |
| บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [มีสัญลักษณ์์สื่อถึงประโยชน์หรือฟังก์ชั่น เช่น ช่วยลดก้อนขน] | 0.0240 | 0.0196 |
| อายุของคุณ | 0.0200 | 0.0200 |
| บรรจุภัณฑ์ (packaging) มีผลต่อการตัดสินใจซื้อใจหรือไม่ | 0.0160 | 0.0196 |
| สถานภาพสมรส | 0.0160 | 0.0196 |
| เพศของคุณ | 0.0120 | 0.0183 |

## Business Guidance
- Use `random_forest` as the source of truth for segment prediction.
- Revisit feature engineering if the confusion matrix shows a strong class bias.
- Use the top three features from the importance table as the narrative backbone in dashboard and slides.

## Output Files
- `outputs/supervised/metrics_summary.json`
- `outputs/supervised/model_comparison.csv`
- `outputs/supervised/confusion_matrix.csv`
- `outputs/supervised/feature_importance.csv`
- `outputs/supervised/predictions.csv`
- `outputs/supervised/best_model.pkl`

# AGENT.md

## Project Overview

This workspace is for the BU cat food packaging survey unsupervised-learning pipeline.

Primary source data:
- `C:\Users\COM\Downloads\BU Data from Survey Cases_final(5).csv`

Primary objective:
- Turn the raw Google Form export into a reproducible unsupervised-learning workflow that produces cleaned data, segment labels, anomaly flags, visual artifacts, and Thai-language reports for the instructor.

## Deliverables

The project should produce these outputs:
- `clean_dataset_with_segments.csv` for downstream supervised learning and dashboard work
- Thai descriptive statistics and correlation report
- Thai clustering / dimensionality reduction / anomaly report
- Thai customer persona and strategic recommendation report
- A personal unsupervised memo for the owner of this workspace

## Required Analysis Flow

Follow this sequence unless the user explicitly changes it:
1. Read the raw CSV exactly as exported from Google Forms / Excel.
2. Reconstruct headers correctly and remove empty export rows.
3. Keep only valid respondents with non-empty timestamp.
4. Keep only respondents who completed the top-3 choice question.
5. Encode survey answers into numeric features.
6. Build rank-weighted vote features.
7. Ipsatize option-attribute Likert scores before PCA.
8. Impute buy-factor missing values with `KNNImputer(n_neighbors=5)` unless rerun evidence justifies a change.
9. Run PCA on standardized ipsatized option features.
10. Run K-Means over PCA outputs and compare candidate `k`.
11. Run hierarchical clustering as a validation view.
12. Run Isolation Forest for anomaly detection.
13. Export final dataset with segment and anomaly fields.
14. Write Thai-language reports from the rerun results, not from assumed numbers.

## Working Rules

- Prefer reproducible Python scripts over notebook-only logic.
- Keep raw data immutable; write all outputs into the workspace.
- Treat the user-provided metrics in chat as hypotheses to verify, not as ground truth.
- Save intermediate artifacts that support reporting: tables, metrics, and plots.
- Document limitations honestly when cluster separation is weak or overlapping.
- Keep language in reports suitable for instructor review in Thai.

## Expected Workspace Structure

- `src/` for pipeline code
- `scripts/` for entrypoints
- `tests/` for validation
- `outputs/` for generated CSV, JSON, and figure artifacts
- `reports/` for Thai report markdown files
- `docs/superpowers/plans/` for planning documents

## Decision Priorities

When tradeoffs appear, optimize in this order:
1. Reproducibility
2. Correctness of filtering / encoding
3. Honest interpretation of unsupervised results
4. Clear Thai communication for reports
5. Convenience

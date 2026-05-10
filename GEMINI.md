# Cat-food Unsupervised

Instructional context for the BU cat food packaging survey unsupervised-learning pipeline.

## Project Overview

This project implements a reproducible unsupervised-learning workflow to analyze cat food survey data. It transforms raw Google Form exports into segment labels, anomaly flags, and customer persona reports.

### Key Technologies
- **Python 3.10+**: Core logic.
- **Pandas**: Data manipulation and loading.
- **Scikit-learn**: KNN Imputation, PCA, K-Means, and Isolation Forest.
- **SciPy**: Hierarchical clustering validation.
- **Pytest**: Automated testing.

### Core Architecture
The pipeline is organized into modular components within `src/catfood_unsupervised/`:
- `config.py`: Project-wide path and file configurations.
- `data_loading.py`: Specialized logic for reading multi-row header exports from Google Forms.
- `preprocessing.py`: Thai Likert scale mapping and KNN-based missing value imputation.
- `features.py`: Feature engineering including rank-weighted vote construction and row-wise ipsatization.
- `models.py`: Clustering (PCA + K-Means), hierarchical validation, and anomaly detection (Isolation Forest).

## Building and Running

### Setup
Install dependencies and the project in editable mode:
```bash
pip install -e .
```

### Running Tests
Execute the test suite using pytest:
```bash
pytest
```

### Analysis Pipeline
The analysis flow should follow this sequence:
1. **Load**: Read `BU Data from Survey Cases_final(5).csv`.
2. **Filter**: Retain only valid respondents with timestamps and completed top-3 choices.
3. **Encode**: Map Thai survey answers to numeric Likert scales.
4. **Impute**: Fill missing "buy factor" values using `KNNImputer(n_neighbors=5)`.
5. **Feature Engineering**: Build rank-weighted votes and ipsatize Likert scores.
6. **Model**:
   - Run PCA on standardized features.
   - Run K-Means on PCA outputs (evaluate optimal `k`).
   - Run Hierarchical clustering for validation.
   - Run Isolation Forest for anomaly detection.
7. **Export**: Save `clean_dataset_with_segments.csv` with segment and anomaly fields.
8. **Report**: Generate Thai-language markdown reports.

## Development Conventions

- **Reproducibility**: Prefer Python scripts (`src/` and entrypoints) over notebook-only workflows.
- **Immutability**: Treat raw data as immutable; all outputs must be written to `outputs/` or `reports/`.
- **Thai Reporting**: All deliverables for the instructor (descriptive stats, personas, strategic recommendations) must be written in Thai.
- **Validation**: Every major code change should be accompanied by a test in `tests/`.
- **Honest Interpretation**: Clearly document limitations if cluster separation is weak or overlapping.

## Workspace Structure

- `src/`: Core pipeline package.
- `tests/`: Unit tests and fixtures.
- `outputs/`: (TODO) Location for generated CSVs, JSON, and plots.
- `reports/`: (TODO) Location for Thai report markdown files.
- `docs/superpowers/plans/`: Implementation plans and pipeline documentation.
- `AGENT.md`: High-level project objectives and deliverables.
- `GEMINI.md`: This instructional context file.

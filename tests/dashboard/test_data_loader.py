from pathlib import Path

from catfood_unsupervised.config import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"

def test_load_metrics_summary():
    from catfood_unsupervised.dashboard.data_loader import load_metrics_summary
    metrics = load_metrics_summary(OUTPUT_DIR)
    assert "pca" in metrics
    assert "kmeans_evaluation" in metrics
    assert metrics["final_cluster_k"] == 2

def test_load_all_data():
    from catfood_unsupervised.dashboard.data_loader import load_all_data
    data = load_all_data(OUTPUT_DIR)
    assert data.metrics["anomaly_detection"]["anomaly_count"] == 26
    assert len(data.clean_df) == 148
    assert "segment" in data.clean_df.columns

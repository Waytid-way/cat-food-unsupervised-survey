from __future__ import annotations

KPI_CARDS = [
    {
        "id": "silhouette",
        "label": "Silhouette Score (k=2)",
        "metric_path": ["kmeans_evaluation", 0, "silhouette_score"],
        "format": ".3f",
        "color": "#2E86AB",
    },
    {
        "id": "davies_bouldin",
        "label": "Davies-Bouldin Index",
        "metric_path": ["kmeans_evaluation", 0, "davies_bouldin_score"],
        "format": ".3f",
        "color": "#A23B72",
    },
    {
        "id": "inertia",
        "label": "Inertia (k=2)",
        "metric_path": ["kmeans_evaluation", 0, "inertia"],
        "format": ".1f",
        "color": "#F18F01",
    },
    {
        "id": "variance_explained",
        "label": "Variance Explained (8 PCs)",
        "metric_path": ["pca", "total_explained_variance_ratio"],
        "format": ".1%",
        "color": "#C73E1D",
    },
    {
        "id": "anomaly_rate",
        "label": "Anomaly Rate",
        "metric_path": ["anomaly_detection", "anomaly_rate"],
        "format": ".1%",
        "color": "#3A7D44",
    },
]

TAB_ITEMS = [
    {"label": "1. EDA & Stats", "value": "tab_eda", "href": "#tab_eda"},
    {"label": "2. Correlation", "value": "tab_correlation", "href": "#tab_correlation"},
    {"label": "3. Clustering", "value": "tab_clustering", "href": "#tab_clustering"},
    {"label": "4. Persona", "value": "tab_persona", "href": "#tab_persona"},
]

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "accent3": "#3A7D44",
    "segment1": "#2E86AB",
    "segment2": "#F18F01",
    "anomaly": "#C73E1D",
    "background": "#F8F9FA",
    "card_bg": "#FFFFFF",
}

CORRELATION_CLUSTERS = [
    {"name": "Classic/Safe", "options": ["opt01", "opt02"], "color": "#2E86AB"},
    {"name": "Mainstream", "options": ["opt03", "opt04", "opt05"], "color": "#F18F01"},
    {"name": "Niche Aesthetic", "options": ["opt08", "opt09", "opt10"], "color": "#A23B72"},
]

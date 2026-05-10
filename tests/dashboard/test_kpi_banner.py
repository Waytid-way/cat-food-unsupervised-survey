from pathlib import Path

def test_kpi_banner_renders_cards():
    from catfood_unsupervised.dashboard.data_loader import load_metrics_summary
    from catfood_unsupervised.dashboard.config import KPI_CARDS
    from catfood_unsupervised.dashboard.components.kpi_banner import render_kpi_banner

    metrics = load_metrics_summary(Path("C:/Users/COM/Projects/Cat-food Unsupervised/outputs"))
    banner = render_kpi_banner(metrics)
    assert len(banner.children.children) == 5  # 5 KPI cards
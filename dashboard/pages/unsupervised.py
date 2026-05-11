from __future__ import annotations

from catfood_unsupervised.dashboard.components.tab_business_insight import render_business_insight_tab
from catfood_unsupervised.dashboard.components.tab_clustering import render_clustering_tab as render_unsupervised_page
from catfood_unsupervised.dashboard.components.tab_correlation import render_correlation_tab
from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab
from catfood_unsupervised.dashboard.components.tab_persona import render_persona_tab

__all__ = [
    "render_business_insight_tab",
    "render_correlation_tab",
    "render_eda_tab",
    "render_persona_tab",
    "render_unsupervised_page",
]

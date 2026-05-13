from __future__ import annotations

from dash import register_page

from catfood_unsupervised.dashboard.config import SUPERVISED_OUTPUT_DIR
from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab
from catfood_unsupervised.dashboard.supervised_data_loader import (
    load_supervised_runtime_bundle,
)


def _render():
    bundle = load_supervised_runtime_bundle(SUPERVISED_OUTPUT_DIR)
    return render_supervised_tab(bundle)


register_page(__name__, path="/supervised", title="Supervised Learning", layout=_render)

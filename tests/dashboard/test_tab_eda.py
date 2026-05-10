from pathlib import Path

def test_eda_tab_renders():
    from catfood_unsupervised.config import PROJECT_ROOT
    from catfood_unsupervised.dashboard.data_loader import load_all_data
    from catfood_unsupervised.dashboard.components.tab_eda import render_eda_tab

    data = load_all_data(PROJECT_ROOT / "outputs")
    tab = render_eda_tab(data)
    assert tab is not None
    assert len(tab.children) > 0

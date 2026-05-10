from __future__ import annotations

from pathlib import Path

from dash import dcc

from catfood_unsupervised.dashboard.components.tab_supervised import render_supervised_tab
from catfood_unsupervised.dashboard.supervised_data_loader import load_supervised_dashboard_bundle
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


def test_render_supervised_tab_contains_scoring_workflow(tmp_path: Path, supervised_fixture_path):
    output_dir = tmp_path / "supervised_outputs"
    report_dir = tmp_path / "supervised_reports"
    run_supervised_pipeline(
        input_path=supervised_fixture_path,
        output_dir=output_dir,
        report_dir=report_dir,
        random_state=7,
        test_size=0.25,
    )

    bundle = load_supervised_dashboard_bundle(output_dir)
    tab = render_supervised_tab(bundle)
    rendered_text = " ".join(_collect_text(tab))
    dropdowns = _collect_components(tab, dcc.Dropdown)

    assert tab.id == "tab_supervised"
    assert "Predict" in rendered_text
    assert "Confidence" in rendered_text
    assert "Recent predictions" in rendered_text
    assert "Best Model" in rendered_text
    assert "Accuracy" in rendered_text
    assert "Macro F1" in rendered_text
    assert "Weighted F1" in rendered_text
    assert len(dropdowns) == len(FEATURE_COLUMNS)
    assert all(dropdown.options for dropdown in dropdowns)


def _collect_text(component) -> list[str]:
    texts: list[str] = []
    if isinstance(component, str):
        return [component]

    children = getattr(component, "children", None)
    if children is None:
        return texts
    if isinstance(children, (list, tuple)):
        for child in children:
            texts.extend(_collect_text(child))
    else:
        texts.extend(_collect_text(children))
    return texts


def _collect_components(component, component_type) -> list[object]:
    matches: list[object] = []
    if isinstance(component, component_type):
        matches.append(component)

    children = getattr(component, "children", None)
    if children is None:
        return matches
    if isinstance(children, (list, tuple)):
        for child in children:
            matches.extend(_collect_components(child, component_type))
    else:
        matches.extend(_collect_components(children, component_type))
    return matches

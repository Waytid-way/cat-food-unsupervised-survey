from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.dashboard.supervised_data_loader import (
    load_supervised_dashboard_bundle,
)
from catfood_unsupervised.dashboard.components.tab_supervised import (
    render_supervised_tab,
)
from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline


def test_render_supervised_tab_contains_key_sections(tmp_path: Path, supervised_fixture_path):
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

    assert tab is not None
    assert tab.id == "tab_supervised"
    assert bundle.metrics["best_model_name"] in [row["model_name"] for row in bundle.comparison.to_dict(orient="records")]

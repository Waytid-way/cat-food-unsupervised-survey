from __future__ import annotations

from pathlib import Path
import os

from dash import html, dcc, register_page
import dash_bootstrap_components as dbc
import plotly.express as px

from catfood_unsupervised.dashboard.data_loader import load_all_data

UNSUPERVISED_OUTPUT_DIR = Path(os.environ.get("CATFOOD_OUTPUT_DIR", "outputs"))


def _render():
    try:
        data = load_all_data(UNSUPERVISED_OUTPUT_DIR)
        segment_profiles = data.segment_profiles
    except Exception:
        return html.Div("Segment data not available. Please run the pipeline first.")

    persona_cards = []
    for segment_id in segment_profiles.index:
        row = segment_profiles.loc[segment_id]
        color = "#2E86AB" if segment_id == 1 else "#F18F01"
        size = int(row.get("segment_size", 0))

        vote_cols = [c for c in segment_profiles.columns if c.startswith("vote_")]
        votes = row[vote_cols].values
        top_vote_idx = int(votes.argmax()) + 1

        buy_factor_cols = [c for c in segment_profiles.columns if c.startswith("buy_factor_")]
        buy_factors = row[buy_factor_cols].values
        top_buy_idx = int(buy_factors.argmax()) + 1

        pkg_cols = [c for c in segment_profiles.columns if c.startswith("packaging_importance_")]
        pkg_vals = row[pkg_cols].values
        top_pkg_idx = int(pkg_vals.argmax()) + 1

        persona_cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H4(f"Segment {segment_id}", style={"color": color}),
                    html.P(f"Size: {size} respondents ({size * 100 / 148:.0f}%)", className="text-muted mb-1"),
                    html.P(f"Top Voted Option: Option {top_vote_idx}", className="text-muted mb-1"),
                    html.P(f"Top Buy Factor: Factor {top_buy_idx}", className="text-muted mb-1"),
                    html.P(f"Top Packaging Importance: Pkg {top_pkg_idx}", className="text-muted mb-1"),
                ]),
                className=f"ind-card",
                style={"border-left": f"4px solid {color}"},
            )
        )

    vote_cols = [c for c in segment_profiles.columns if c.startswith("vote_")]
    vote_df = segment_profiles[vote_cols].copy()
    vote_df.columns = [f"Opt {i+1}" for i in range(len(vote_cols))]
    vote_df = vote_df.reset_index().melt(id_vars="segment", var_name="Option", value_name="Avg Votes")
    vote_df = vote_df.rename(columns={"index": "Segment"})

    vote_fig = px.bar(
        vote_df,
        x="Option",
        y="Avg Votes",
        color="segment",
        title="Average Votes per Option by Segment",
        color_discrete_map={1: "#2E86AB", 2: "#F18F01"},
        barmode="group",
    )
    vote_fig.update_layout(plot_bgcolor="white")

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Persona Cards", className="page-title"), html.P("Segment profile deep-dive", className="text-muted")]), className="ind-card"), width=12),
                ],
                className="mb-4",
            ),
            dbc.Row(persona_cards, className="mb-4 g-3") if persona_cards else dbc.Row([]),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=vote_fig, className="chart-card")]), className="chart-card"), width=12),
                ],
                className="mb-4",
            ),
        ],
        className="shell-content",
    )


register_page(__name__, path="/business", title="Business Insight", layout=_render)
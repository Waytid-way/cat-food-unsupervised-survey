from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class SupervisedFieldSpec:
    column: str
    field_id: str
    label: str
    hint: str
    options: list[str]


def build_supervised_field_specs(
    feature_options: dict[str, list[str]],
) -> list[SupervisedFieldSpec]:
    specs: list[SupervisedFieldSpec] = []
    for index, column in enumerate(FEATURE_COLUMNS, start=1):
        specs.append(
            SupervisedFieldSpec(
                column=column,
                field_id=f"supervised-feature-{index:02d}",
                label=f"Question {index:02d}",
                hint=column,
                options=list(feature_options.get(column, [])),
            )
        )
    return specs


def render_supervised_form(field_specs: Iterable[SupervisedFieldSpec] | None = None) -> html.Div:
    specs = list(field_specs or build_supervised_field_specs())
    field_cards = [_render_field_card(spec) for spec in specs]
    rows = []
    for row_index in range(0, len(field_cards), 2):
        rows.append(
            dbc.Row(
                [
                    dbc.Col(field_cards[row_index], width=6),
                    dbc.Col(field_cards[row_index + 1], width=6) if row_index + 1 < len(field_cards) else dbc.Col(width=6),
                ],
                className="g-3 mb-3",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Supervised input", className="supervised-form__eyebrow"),
                    html.H4("Build a customer row and score it in one click", className="supervised-form__title"),
                    html.P(
                        "Pick a value for each canonical survey feature. Each dropdown is populated from the training data, so the row you enter matches the supervised schema exactly.",
                        className="supervised-form__lede",
                    ),
                ],
                className="supervised-form__intro",
            ),
            *rows,
            html.Div(
                [
                    html.Button(
                        "Predict",
                        id="supervised-predict-button",
                        n_clicks=0,
                        className="btn btn-primary supervised-form__button",
                    ),
                    html.Div(
                        "All fields are required. Values are entered as the survey answer text.",
                        className="supervised-form__note",
                    ),
                ],
                className="supervised-form__actions",
            ),
        ],
        id="supervised-input-form",
    )


def field_input_id(spec: SupervisedFieldSpec) -> str:
    return spec.field_id


def _render_field_card(spec: SupervisedFieldSpec) -> html.Div:
    dropdown_options = [
        {"label": option, "value": option}
        for option in spec.options
    ]
    return html.Div(
        [
            html.Label(spec.label, className="supervised-form__label"),
            html.Div(spec.hint, className="supervised-form__hint"),
            dcc.Dropdown(
                id=spec.field_id,
                options=dropdown_options,
                placeholder="Select a survey answer",
                clearable=False,
                className="supervised-form__input",
            ),
        ],
        className="supervised-form__field",
    )

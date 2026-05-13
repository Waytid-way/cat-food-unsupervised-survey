from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest

from dash import dcc, html

from catfood_unsupervised.dashboard.bootstrap import dbc
from catfood_unsupervised.shared.survey_values import normalize_survey_value
from catfood_unsupervised.supervised.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class SupervisedFieldSpec:
    column: str
    field_id: str
    label: str
    options: tuple[str, ...]
    default_index: int
    widget_kind: str
    section_key: str


_SECTION_ORDER = ("product", "packaging", "profile")
_SECTION_TITLES = {
    "product": "คุณสมบัติอาหารและรสชาติ",
    "packaging": "บรรจุภัณฑ์และการสื่อสาร",
    "profile": "ข้อมูลผู้ตอบ",
}
_SECTION_DESCRIPTIONS = {
    "product": "สเกลเลื่อนด้านล่างช่วยให้เลือกคำตอบจากน้อยไปมากอย่างเป็นระบบ",
    "packaging": "ใช้สเกลเลื่อนกับคำถามเชิงความชอบ และใช้ช่องเลือกกับข้อมูลพื้นฐาน",
    "profile": "ข้อมูลประชากรศาสตร์ยังคงใช้ช่องเลือกเหมือนเดิมเพื่อให้อ่านง่าย",
}
_LIKERT_ORDER = ("น้อยที่สุด", "น้อย", "ปานกลาง", "เฉยๆ", "มาก", "มากที่สุด")
_BINARY_ORDER = ("ไม่มีผล", "มีผล")
_PACKAGING_IMAGE_ORDER = (
    "ภาพการ์ตูน หรือลายเส้น",
    "ได้ทั้งสองแบบ หากถูกใจ",
    "ภาพแมวจริง หรือแมวสมจริง (AI)",
)
_AGE_ORDER = ("20-29ปี", "30-39ปี", "40-49ปี", "50ปี ขึ้นไป")
_GENDER_ORDER = ("ชาย", "หญิง", "อื่นๆ")
_MARITAL_ORDER = ("โสด ไม่มีแฟน", "มีแฟนแต่ยังไม่แต่งงาน", "แต่งงานแล้ว", "หย่าร้าง/เป็นม่าย")
_PROFILE_COLUMNS = {FEATURE_COLUMNS[15], FEATURE_COLUMNS[16], FEATURE_COLUMNS[17]}
_SHORT_LABELS = {
    FEATURE_COLUMNS[0]: "ใช้วัตถุดิบจากธรรมชาติ",
    FEATURE_COLUMNS[1]: "ใช้วัตถุดิบนำเข้า",
    FEATURE_COLUMNS[2]: "รสชาติกลมกล่อมถูกปากแมว",
    FEATURE_COLUMNS[3]: "เป็นสินค้านำเข้าจากต่างประเทศ",
    FEATURE_COLUMNS[4]: "แบรนด์มีชื่อเสียง",
    FEATURE_COLUMNS[5]: "บรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อหรือไม่",
    FEATURE_COLUMNS[6]: "ชอบภาพแบบใดบนบรรจุภัณฑ์",
    FEATURE_COLUMNS[7]: "บรรจุภัณฑ์ดูดีพรีเมียม",
    FEATURE_COLUMNS[8]: "บรรจุภัณฑ์มีภาพแมว",
    FEATURE_COLUMNS[9]: "บรรจุภัณฑ์มีภาพอาหารเม็ด",
    FEATURE_COLUMNS[10]: "บรรจุภัณฑ์มีภาพวัตถุดิบจริง",
    FEATURE_COLUMNS[11]: "บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม",
    FEATURE_COLUMNS[12]: "สื่อถึงแหล่งผลิตหรือที่มา",
    FEATURE_COLUMNS[13]: "สื่อถึงประโยชน์หรือฟังก์ชัน",
    FEATURE_COLUMNS[14]: "มีการการันตีหรือรางวัล",
    FEATURE_COLUMNS[15]: "อายุของคุณ",
    FEATURE_COLUMNS[16]: "เพศของคุณ",
    FEATURE_COLUMNS[17]: "สถานภาพสมรส",
}


def build_supervised_field_specs(
    feature_options: Mapping[str, Sequence[str]]
) -> list[SupervisedFieldSpec]:
    specs: list[SupervisedFieldSpec] = []
    for index, column in enumerate(FEATURE_COLUMNS, start=1):
        options = _sort_supervised_options(column, feature_options.get(column, ()))
        section_key = _section_for_column(column)
        specs.append(
            SupervisedFieldSpec(
                column=column,
                field_id=f"supervised-field-{index:02d}",
                label=_SHORT_LABELS.get(column, column),
                options=options,
                default_index=0,
                widget_kind="dropdown",
                section_key=section_key,
            )
        )
    return specs


def render_supervised_form(
    feature_options: Mapping[str, Sequence[str]],
    *,
    model_path_text: str,
    history_path_text: str,
) -> html.Div:
    specs = build_supervised_field_specs(feature_options)
    sections = _group_specs_by_section(specs)

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5("Supervised Prediction", className="page-title"),
                        html.P(
                            "Upload a retrained `best_model.pkl`, choose answers from dropdowns, and predict a segment.",
                            className="text-muted",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("Upload Model", className="mb-2"),
                            dcc.Upload(
                                id="supervised-model-upload",
                                children=html.Div(
                                    [
                                        html.Strong("Drop or select a retrained model file"),
                                        html.Div("Accepted file: `best_model.pkl`"),
                                    ]
                                ),
                                multiple=False,
                                className="supervised-upload",
                            ),
                            html.Div(
                                [
                                    html.Div(f"Current model path: {model_path_text}", className="text-muted"),
                                    html.Div(f"History DB: {history_path_text}", className="text-muted"),
                                ],
                                className="mt-2",
                            ),
                            html.Div(id="supervised-model-status", className="mt-2"),
                        ]
                    ),
                    className="mb-3",
                ),
                html.Div(
                    [_render_section(section_key, section_specs) for section_key, section_specs in sections],
                    className="supervised-form-sections",
                ),
                html.Div(
                    [
                        html.Button(
                            "Predict",
                            id="supervised-predict-button",
                            type="button",
                            n_clicks=0,
                            className="shell-primary-button mt-3",
                        )
                    ],
                    className="d-flex justify-content-end",
                ),
            ]
        ),
        className="ind-card",
    )


def _group_specs_by_section(
    specs: Sequence[SupervisedFieldSpec],
) -> list[tuple[str, tuple[SupervisedFieldSpec, ...]]]:
    grouped: dict[str, list[SupervisedFieldSpec]] = {key: [] for key in _SECTION_ORDER}
    for spec in specs:
        grouped.setdefault(spec.section_key, []).append(spec)
    return [
        (section_key, tuple(grouped.get(section_key, ())))
        for section_key in _SECTION_ORDER
        if grouped.get(section_key)
    ]


def _render_section(section_key: str, specs: Sequence[SupervisedFieldSpec]) -> dbc.Card:
    title = _SECTION_TITLES.get(section_key, section_key.title())
    description = _SECTION_DESCRIPTIONS.get(section_key, "")
    rows = _build_field_rows(specs) if specs else []
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="mb-1"),
                html.P(description, className="text-muted mb-3"),
                html.Div(rows, className="supervised-form-grid"),
            ]
        ),
        className="mb-3 supervised-section-card",
    )


def _build_field_rows(specs: Sequence[SupervisedFieldSpec]) -> list[dbc.Row]:
    rows: list[dbc.Row] = []
    iterator = iter(specs)
    for left_spec, right_spec in zip_longest(iterator, iterator):
        cols = [_build_field_col(left_spec)]
        if right_spec is not None:
            cols.append(_build_field_col(right_spec))
        rows.append(dbc.Row(cols, className="g-3 mb-3"))
    return rows


def _build_field_col(spec: SupervisedFieldSpec | None):
    if spec is None:
        return dbc.Col(html.Div(), width=6)

    if not spec.options:
        return dbc.Col(
            html.Div(
                [
                    html.Label(spec.label, htmlFor=spec.field_id, className="form-label"),
                    html.Div("No options available.", className="text-muted"),
                ]
            ),
            width=6,
        )

    return _build_dropdown_field_col(spec)


def _build_dropdown_field_col(spec: SupervisedFieldSpec) -> dbc.Col:
    dropdown_options = [{"label": value, "value": value} for value in spec.options]
    default_value = spec.options[spec.default_index] if spec.options else None
    return dbc.Col(
        html.Div(
            [
                html.Label(spec.label, htmlFor=spec.field_id, className="form-label"),
                dcc.Dropdown(
                    id=spec.field_id,
                    options=dropdown_options,
                    value=default_value,
                    clearable=False,
                    placeholder=f"Select {spec.label}",
                ),
            ]
        ),
        width=6,
    )


def _section_for_column(column: str) -> str:
    index = FEATURE_COLUMNS.index(column)
    if index <= 4:
        return "product"
    if index <= 14:
        return "packaging"
    return "profile"


def _sort_supervised_options(column: str, options: Sequence[object]) -> tuple[str, ...]:
    normalized_options = [str(value).strip() for value in options if normalize_survey_value(value) is not None]
    if column == FEATURE_COLUMNS[15]:
        return _sort_by_order(normalized_options, _AGE_ORDER)
    if column == FEATURE_COLUMNS[16]:
        return _sort_by_order(normalized_options, _GENDER_ORDER)
    if column == FEATURE_COLUMNS[17]:
        return _sort_by_order(normalized_options, _MARITAL_ORDER)
    if column == FEATURE_COLUMNS[5]:
        return _sort_by_order(normalized_options, _BINARY_ORDER)
    if column == FEATURE_COLUMNS[6]:
        return _sort_by_order(normalized_options, _PACKAGING_IMAGE_ORDER)
    return _sort_by_order(normalized_options, _LIKERT_ORDER)


def _sort_by_order(values: Sequence[str], preferred_order: Sequence[str]) -> tuple[str, ...]:
    preferred_index = {value: index for index, value in enumerate(preferred_order)}
    enumerated = list(enumerate(values))
    sorted_values = sorted(
        enumerated,
        key=lambda item: (
            preferred_index.get(item[1], len(preferred_order) + item[0]),
            item[0],
        ),
    )
    return tuple(item[1] for item in sorted_values)


__all__ = [
    "SupervisedFieldSpec",
    "build_supervised_field_specs",
    "render_supervised_form",
]

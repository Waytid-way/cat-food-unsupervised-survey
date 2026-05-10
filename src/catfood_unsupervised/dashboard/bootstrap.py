from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dash import html

try:  # pragma: no cover - exercised indirectly when dependency is installed
    import dash_bootstrap_components as dbc  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in this workspace
    class _ThemeNamespace(SimpleNamespace):
        BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

    def _merge_class_name(*values: Any) -> str | None:
        classes = [str(value).strip() for value in values if value]
        merged = " ".join(part for part in classes if part)
        return merged or None

    def _wrap(tag: str):
        def factory(children: Any = None, **kwargs: Any):
            class_name = kwargs.pop("className", None)
            style = kwargs.pop("style", None)
            component_id = kwargs.pop("id", None)

            if tag == "Container":
                fluid = kwargs.pop("fluid", False)
                class_name = _merge_class_name(
                    class_name, "container-fluid" if fluid else "container"
                )
            elif tag == "Row":
                class_name = _merge_class_name(class_name, "row")
            elif tag == "Col":
                width = kwargs.pop("width", None)
                if width is not None:
                    class_name = _merge_class_name(class_name, f"col-{width}")
            elif tag == "Card":
                class_name = _merge_class_name(class_name, "card")
            elif tag == "CardBody":
                class_name = _merge_class_name(class_name, "card-body")
            elif tag == "Alert":
                class_name = _merge_class_name(class_name, "alert")

            return html.Div(
                children,
                id=component_id,
                className=class_name,
                style=style,
            )

        return factory

    dbc = SimpleNamespace(  # type: ignore[assignment]
        themes=_ThemeNamespace(),
        Row=_wrap("Row"),
        Col=_wrap("Col"),
        Container=_wrap("Container"),
        Card=_wrap("Card"),
        CardBody=_wrap("CardBody"),
        Alert=_wrap("Alert"),
    )

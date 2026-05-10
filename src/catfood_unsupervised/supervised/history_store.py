from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_TABLE_NAME = "prediction_history"
_EMPTY_COLUMNS = [
    "id",
    "created_at",
    "source",
    "model_name",
    "predicted_segment",
    "probability",
    "raw_input",
]


def initialize_prediction_history_db(db_path: str | Path) -> None:
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_segment INTEGER NOT NULL,
                probability_json TEXT NOT NULL,
                raw_input_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def append_prediction_history(
    db_path: str | Path,
    *,
    source: str,
    model_name: str,
    predicted_segment: int,
    probability: Mapping[str, Any] | None = None,
    raw_input: Mapping[str, Any] | None = None,
    probability_map: Mapping[str, Any] | None = None,
    input_payload: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> None:
    initialize_prediction_history_db(db_path)

    payload_probability = probability_map if probability_map is not None else probability
    payload_input = input_payload if input_payload is not None else raw_input
    if payload_probability is None:
        payload_probability = {}
    if payload_input is None:
        payload_input = {}

    timestamp = created_at or datetime.now(timezone.utc)
    timestamp_text = timestamp.astimezone(timezone.utc).isoformat()

    with sqlite3.connect(Path(db_path)) as connection:
        connection.execute(
            f"""
            INSERT INTO {_TABLE_NAME} (
                created_at,
                source,
                model_name,
                predicted_segment,
                probability_json,
                raw_input_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp_text,
                str(source),
                str(model_name),
                int(predicted_segment),
                json.dumps(dict(payload_probability), ensure_ascii=False, default=str),
                json.dumps(dict(payload_input), ensure_ascii=False, default=str),
            ),
        )
        connection.commit()


def fetch_recent_prediction_history(
    db_path: str | Path,
    limit: int = 20,
) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    with sqlite3.connect(path) as connection:
        query = f"""
            SELECT id, created_at, source, model_name, predicted_segment, probability_json, raw_input_json
            FROM {_TABLE_NAME}
            ORDER BY id DESC
            LIMIT ?
        """
        rows = connection.execute(query, (int(limit),)).fetchall()

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    frame = pd.DataFrame(
        rows,
        columns=[
            "id",
            "created_at",
            "source",
            "model_name",
            "predicted_segment",
            "probability_json",
            "raw_input_json",
        ],
    )
    frame["probability"] = frame["probability_json"].map(_safe_json_loads)
    frame["raw_input"] = frame["raw_input_json"].map(_safe_json_loads)
    return frame.loc[:, _EMPTY_COLUMNS]


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"value": value}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}

from __future__ import annotations

from pathlib import Path

from catfood_unsupervised.supervised.history_store import (
    append_prediction_history,
    fetch_recent_prediction_history,
    initialize_prediction_history_db,
)


def test_prediction_history_round_trips_one_record(tmp_path: Path):
    db_path = tmp_path / "history.sqlite3"
    initialize_prediction_history_db(db_path)

    append_prediction_history(
        db_path=db_path,
        source="cli",
        model_name="best_model.pkl",
        predicted_segment=1,
        probability={"1": 0.8, "2": 0.2},
        raw_input={"question_a": "high", "question_b": "low"},
    )

    history = fetch_recent_prediction_history(db_path, limit=10)

    assert len(history) == 1
    row = history.iloc[0]
    assert row["source"] == "cli"
    assert row["model_name"] == "best_model.pkl"
    assert row["predicted_segment"] == 1
    assert row["probability"] == {"1": 0.8, "2": 0.2}
    assert row["raw_input"] == {"question_a": "high", "question_b": "low"}


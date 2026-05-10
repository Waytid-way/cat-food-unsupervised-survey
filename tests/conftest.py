from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from catfood_unsupervised.supervised.config import ANOMALY_COLUMN, TARGET_COLUMN


@pytest.fixture()
def supervised_fixture_path(tmp_path: Path) -> Path:
    path = tmp_path / "supervised_fixture.csv"
    columns = [f"col_{index}" for index in range(100)]
    columns[0] = "Timestamp"
    columns[72] = "top3"
    columns[73] = "age"
    columns[74] = "gender"
    columns[75] = "marital"
    columns[98] = TARGET_COLUMN
    columns[99] = ANOMALY_COLUMN

    rows = []
    for row_index in range(12):
        is_segment_one = row_index < 6
        row = [f"value_{row_index}_{col_index}" for col_index in range(100)]
        row[0] = f"2024-01-01 10:{row_index:02d}:00"
        row[5] = "มากที่สุด" if is_segment_one else "น้อย"
        row[6] = "มาก" if is_segment_one else "น้อย"
        row[7] = "มากที่สุด" if is_segment_one else "น้อย"
        row[8] = "มาก" if is_segment_one else "น้อย"
        row[9] = "มากที่สุด" if is_segment_one else "น้อย"
        row[10] = "มีผล" if is_segment_one else "ไม่มีผล"
        row[11] = "ภาพแมว" if is_segment_one else "ภาพอาหารเม็ด"
        for col_index in range(12, 20):
            row[col_index] = "มาก" if is_segment_one else "น้อย"
        row[73] = "20-29ปี" if is_segment_one else "30-39ปี"
        row[74] = "หญิง" if is_segment_one else "ชาย"
        row[75] = "โสด ยังไม่แต่งงาน" if is_segment_one else "แต่งงานแล้ว"
        row[98] = 1 if is_segment_one else 2
        row[99] = 0
        rows.append(row)

    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path

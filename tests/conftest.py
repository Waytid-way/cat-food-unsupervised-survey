from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from catfood_unsupervised.supervised.schema import ANOMALY_COLUMN, TARGET_COLUMN


@pytest.fixture()
def supervised_fixture_path(tmp_path: Path) -> Path:
    source_columns = pd.read_csv(
        Path(__file__).resolve().parents[1] / "outputs" / "clean_dataset_with_segments.csv",
        nrows=0,
    ).columns.tolist()
    path = tmp_path / "supervised_fixture.csv"

    rows: list[dict[str, object]] = []
    for row_index in range(12):
        is_segment_one = row_index < 6
        row = {column: f"value_{row_index}_{column}" for column in source_columns}
        row["Timestamp"] = f"2024-01-01 10:{row_index:02d}:00"
        row["คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [ใช้วัตถุดิบจากธรรมชาติ]"] = (
            "มากที่สุด" if is_segment_one else "น้อย"
        )
        row["คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [ใช้วัตถุดิบนำเข้าจากต่างประเทศ เช่น เนื้อปลาทูน่าจากญี่ปุ่น]"] = (
            "มาก" if is_segment_one else "น้อย"
        )
        row["คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [รสชาติกลมกล่อมอร่อยถูกปากแมว เช่น เทไว้แล้วแมวกินหมดไม่เหลือ, หยิบถุงแล้วแมวรอกิน]"] = (
            "มากที่สุด" if is_segment_one else "น้อย"
        )
        row["คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [เป็นผลิตภัณฑ์จากต่างประเทศ เช่น ญี่ปุ่น, อเมริกา]"] = (
            "มาก" if is_segment_one else "น้อย"
        )
        row["คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [แบรนด์มีชื่อเสียงเป็นที่รู้จัก]"] = (
            "มากที่สุด" if is_segment_one else "น้อย"
        )
        row["บรรจุภัณฑ์ (packaging) มีผลต่อการตัดสินใจซื้อใจหรือไม่"] = (
            "มีผล" if is_segment_one else "ไม่มีผล"
        )
        row["สำหรับบรรจุภัณฑ์อาหารแมว คุณชอบภาพแบบใด"] = (
            "ภาพแมว" if is_segment_one else "ภาพอาหารเม็ด"
        )
        for column in source_columns[12:20]:
            row[column] = "มาก" if is_segment_one else "น้อย"
        row["อายุของคุณ"] = "20-29ปี" if is_segment_one else "30-39ปี"
        row["เพศของคุณ"] = "หญิง" if is_segment_one else "ชาย"
        row["สถานภาพสมรส"] = (
            "โสด ยังไม่แต่งงาน" if is_segment_one else "แต่งงานแล้ว"
        )
        row["จากตัวเลือกทั้งหมด คุณชอบการออกแบบบรรจุภัณฑ์อาหารแมวสำเร็จรูปแบบใดมากที่สุด 3 อันดับแรก"] = (
            "Option 3, Option 6, Option 8"
            if is_segment_one
            else "Option 1, Option 2, Option 4"
        )
        row["top3_rank_1"] = 3 if is_segment_one else 1
        row["top3_rank_2"] = 6 if is_segment_one else 2
        row["top3_rank_3"] = 8 if is_segment_one else 4
        for vote_index in range(1, 11):
            row[f"vote_{vote_index:02d}"] = vote_index if is_segment_one else 0
        for pc_index in range(1, 9):
            row[f"PC{pc_index}"] = round(pc_index * 0.1, 3) if is_segment_one else 0.0
        row[TARGET_COLUMN] = 1 if is_segment_one else 2
        row[ANOMALY_COLUMN] = 0
        row["anomaly_score"] = 0.399 if is_segment_one else 0.123
        rows.append(row)

    pd.DataFrame(rows, columns=source_columns).to_csv(path, index=False)
    return path

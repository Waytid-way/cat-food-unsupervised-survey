from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from catfood_unsupervised.supervised.config import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR


REPORT_FILENAMES = {
    "model_report": "supervised_model_report_th.md",
    "owner_memo": "supervised_owner_memo_th.md",
}


@dataclass(frozen=True)
class SupervisedReportContext:
    metrics: dict[str, Any]
    comparison: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


def load_report_context(output_dir: str | Path) -> SupervisedReportContext:
    base = Path(output_dir)
    metrics = json.loads((base / "metrics_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(base / "model_comparison.csv")
    confusion_matrix = pd.read_csv(base / "confusion_matrix.csv", index_col=0)
    feature_importance = pd.read_csv(base / "feature_importance.csv")
    predictions = pd.read_csv(base / "predictions.csv")
    return SupervisedReportContext(
        metrics=metrics,
        comparison=comparison,
        confusion_matrix=confusion_matrix,
        feature_importance=feature_importance,
        predictions=predictions,
    )


def render_supervised_model_report(context: SupervisedReportContext) -> str:
    metrics = context.metrics
    best_row = context.comparison.iloc[0]
    report_lines = [
        "# รายงานโมเดล Supervised Learning",
        "",
        "## สรุปผู้บริหาร",
        f"- จำนวนตัวอย่างที่ใช้ฝึก: {metrics['row_count']} ราย",
        f"- จำนวนฟีเจอร์หลังเข้ารหัส: {metrics['feature_count']} ฟีเจอร์",
        f"- โมเดลที่ดีที่สุด: {metrics['best_model_name']}",
        f"- Accuracy: {metrics['best_model_accuracy']:.3f}",
        f"- Macro F1: {metrics['best_model_macro_f1']:.3f}",
        f"- Weighted F1: {metrics['best_model_weighted_f1']:.3f}",
        f"- ROC AUC: {metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "- ROC AUC: n/a",
        "",
        "## เปรียบเทียบโมเดล",
        _markdown_table(
            ["model", "accuracy", "macro_f1", "weighted_f1"],
            [
                [
                    row["model_name"],
                    f"{row['accuracy']:.3f}",
                    f"{row['macro_f1']:.3f}",
                    f"{row['weighted_f1']:.3f}",
                ]
                for _, row in context.comparison.iterrows()
            ],
        ),
        "",
        "## Confusion Matrix",
        _markdown_table(
            ["", *context.confusion_matrix.columns.tolist()],
            [
                [index, *[int(value) for value in row.tolist()]]
                for index, row in context.confusion_matrix.iterrows()
            ],
        ),
        "",
        "## Feature Importance",
        _markdown_table(
            ["feature", "importance_mean", "importance_std"],
            [
                [
                    row["feature"],
                    f"{row['importance_mean']:.4f}",
                    f"{row['importance_std']:.4f}",
                ]
                for _, row in context.feature_importance.head(10).iterrows()
            ],
        ),
        "",
        "## คำแนะนำเชิงธุรกิจ",
        f"- กลุ่มที่สะท้อนด้วย segment มากที่สุดควรอธิบายด้วยโมเดล `{metrics['best_model_name']}` เพราะให้ผลดีที่สุดบนชุดทดสอบ",
        f"- ควรใช้ feature ที่เด่นที่สุด 3 อันดับแรกจากโมเดลนี้เป็นตัวขับ narrative ใน dashboard และ deck สำหรับลูกค้า",
        f"- ใช้ confusion matrix เพื่อตรวจว่ากลุ่มใดสับสนกันมากที่สุดก่อนนำไปออกแบบ campaign segmentation จริง",
        "",
        "## ไฟล์ผลลัพธ์ที่เกี่ยวข้อง",
        "- `outputs/supervised/metrics_summary.json`",
        "- `outputs/supervised/model_comparison.csv`",
        "- `outputs/supervised/confusion_matrix.csv`",
        "- `outputs/supervised/feature_importance.csv`",
        "- `outputs/supervised/predictions.csv`",
        "- `outputs/supervised/best_model.pkl`",
    ]
    return "\n".join(report_lines).rstrip() + "\n"


def render_supervised_owner_memo(context: SupervisedReportContext) -> str:
    metrics = context.metrics
    return "\n".join(
        [
            "# บันทึกสำหรับทีมธุรกิจ",
            "",
            f"- โมเดลที่ใช้เป็น source of truth: `{metrics['best_model_name']}`",
            f"- Accuracy บน holdout set: {metrics['best_model_accuracy']:.3f}",
            f"- Macro F1 บน holdout set: {metrics['best_model_macro_f1']:.3f}",
            "- ใช้ dashboard แท็บ Supervised เพื่อดูการเทียบโมเดล, confusion matrix, และ feature importance",
            "- ถ้าต้องการ export ไปใช้งานภายนอก ให้หยิบ `best_model.pkl` และ `metrics_summary.json` เป็นหลัก",
            "",
            "## วิธีตีความผล",
            "- ถ้า confusion matrix มีค่าผิดพลาดสูงในบางกลุ่ม แปลว่า segment อาจยังซ้อนทับกัน และควรทบทวน feature engineering",
            "- ถ้า feature importance กระจุกที่คอลัมน์เดิมจำนวนมาก แปลว่าข้อมูล survey มี signal ค่อนข้างแคบ ควรพิจารณาเพิ่มคำถามในรอบถัดไป",
        ]
    ).rstrip() + "\n"


def write_reports_from_output_dir(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    context = load_report_context(output_dir)
    base = Path(report_dir)
    base.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_report": base / REPORT_FILENAMES["model_report"],
        "owner_memo": base / REPORT_FILENAMES["owner_memo"],
    }
    outputs["model_report"].write_text(
        render_supervised_model_report(context), encoding="utf-8"
    )
    outputs["owner_memo"].write_text(
        render_supervised_owner_memo(context), encoding="utf-8"
    )
    return outputs


def run_supervised_workflow(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    from catfood_unsupervised.supervised.pipeline import run_supervised_pipeline

    pipeline_result = run_supervised_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        report_dir=report_dir,
        **pipeline_kwargs,
    )
    return {
        "pipeline": pipeline_result,
        "report_paths": pipeline_result["report_paths"],
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])

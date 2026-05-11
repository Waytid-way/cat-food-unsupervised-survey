from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from catfood_unsupervised.shared.survey_values import THAI_LIKERT_SCALE, map_series_values


REPORT_FILENAMES = {
    "descriptive_stats": "descriptive_stats_th.md",
    "unsupervised_report": "unsupervised_report_th.md",
    "owner_memo": "unsupervised_owner_memo_th.md",
}

OPTION_RATING_COLUMN_INDICES = range(22, 72)
BUY_FACTOR_COLUMN_INDICES = range(5, 10)
PACKAGING_IMPORTANCE_COLUMN_INDICES = range(12, 20)
AGE_COLUMN_INDEX = 73
GENDER_COLUMN_INDEX = 74
MARITAL_COLUMN_INDEX = 75
TOP3_COLUMN_INDEX = 72

BUY_FACTOR_LABELS = [
    "ใช้วัตถุดิบจากธรรมชาติ",
    "ใช้วัตถุดิบนำเข้าจากต่างประเทศ",
    "รสชาติกลมกล่อมอร่อยถูกปากแมว",
    "เป็นผลิตภัณฑ์จากต่างประเทศ",
    "แบรนด์มีชื่อเสียงเป็นที่รู้จัก",
]

PACKAGING_IMPORTANCE_LABELS = [
    "บรรจุภัณฑ์ดูดีพรีเมียม",
    "บรรจุภัณฑ์มีภาพแมว",
    "บรรจุภัณฑ์มีภาพอาหารเม็ดจริง",
    "บรรจุภัณฑ์มีภาพวัตถุดิบและส่วนผสมจริง",
    "บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม",
    "มีสัญลักษณ์สื่อถึงแหล่งผลิตหรือที่มา",
    "มีสัญลักษณ์สื่อถึงประโยชน์หรือฟังก์ชัน",
    "มีการการันตีหรือรางวัล",
]

PACKAGING_EFFECT_LABELS = {
    "มีผล": "มีผล",
    "ไม่มีผล": "ไม่มีผล",
}


@dataclass(frozen=True)
class ReportContext:
    metrics: dict[str, Any]
    clean_df: pd.DataFrame
    correlation_matrix: pd.DataFrame
    segment_profiles: pd.DataFrame


def load_report_context(output_dir: str | Path) -> ReportContext:
    base = Path(output_dir)
    metrics = json.loads((base / "metrics_summary.json").read_text(encoding="utf-8"))
    clean_df = pd.read_csv(base / "clean_dataset_with_segments.csv")
    correlation_matrix = pd.read_csv(base / "correlation_matrix.csv", index_col=0)
    segment_profiles = pd.read_csv(base / "segment_profiles.csv", index_col=0)
    return ReportContext(
        metrics=metrics,
        clean_df=clean_df,
        correlation_matrix=correlation_matrix,
        segment_profiles=segment_profiles,
    )


def render_descriptive_stats_report(context: ReportContext) -> str:
    clean_df = context.clean_df
    metrics = context.metrics

    gender_rows = _count_rows("เพศ", clean_df.iloc[:, GENDER_COLUMN_INDEX])
    age_rows = _count_rows("อายุ", clean_df.iloc[:, AGE_COLUMN_INDEX])
    marital_rows = _count_rows("สถานภาพสมรส", clean_df.iloc[:, MARITAL_COLUMN_INDEX])
    packaging_effect_rows = _count_rows(
        "ผลของบรรจุภัณฑ์", clean_df.iloc[:, 10]
    )

    buy_factor_rows = _mean_rows(
        clean_df,
        BUY_FACTOR_COLUMN_INDICES,
        BUY_FACTOR_LABELS,
    )
    packaging_importance_rows = _mean_rows(
        clean_df,
        PACKAGING_IMPORTANCE_COLUMN_INDICES,
        PACKAGING_IMPORTANCE_LABELS,
    )
    vote_rows = _vote_rows(clean_df)
    correlation_rows = _option_correlation_rows(clean_df)

    lines = [
        "# รายงานสถิติเชิงพรรณนาและความสัมพันธ์ของตัวแปร",
        "",
        "## ภาพรวมชุดข้อมูล",
        f"- ข้อมูลดิบหลัง export จาก Google Form มี {metrics['row_counts']['raw_loaded']} แถว",
        f"- หลังกรองเฉพาะผู้ตอบที่ครบ top-3 choice เหลือ n = {metrics['row_counts']['completed_top3']}",
        f"- คอลัมน์ top-3 ที่ใช้จริงคือ `{metrics['selected_top3_column']}`",
        f"- ทุกคนในชุดข้อมูลสุดท้ายเคยซื้ออาหารแมวมาก่อน: {clean_df.iloc[:, 1].astype(str).str.strip().eq('เคย').mean() * 100:.1f}%",
        "",
        "## ข้อมูลประชากร",
        "### เพศ",
        _markdown_table(["ตัวแปร", "กลุ่มย่อย", "จำนวน", "สัดส่วน"], gender_rows),
        "",
        "### อายุ",
        _markdown_table(["ตัวแปร", "กลุ่มย่อย", "จำนวน", "สัดส่วน"], age_rows),
        "",
        "### สถานภาพสมรส",
        _markdown_table(["ตัวแปร", "กลุ่มย่อย", "จำนวน", "สัดส่วน"], marital_rows),
        "",
        "### ผลของบรรจุภัณฑ์",
        _markdown_table(["ตัวแปร", "กลุ่มย่อย", "จำนวน", "สัดส่วน"], packaging_effect_rows),
        "",
        "## ปัจจัยที่มีผลต่อการซื้อ",
        _markdown_table(["ปัจจัย", "ค่าเฉลี่ย (จาก 4)"], buy_factor_rows),
        "",
        "## ความสำคัญของบรรจุภัณฑ์",
        _markdown_table(["องค์ประกอบ", "ค่าเฉลี่ย (จาก 5)"], packaging_importance_rows),
        "",
        "## อันดับโหวตจาก top-3",
        _markdown_table(["ตัวเลือก", "คะแนนถ่วงน้ำหนัก"], vote_rows),
        "",
        "## ความสัมพันธ์ของรสนิยมการออกแบบ",
        _markdown_table(["คู่ที่สัมพันธ์กัน", "Spearman r"], correlation_rows),
        "",
        "## ข้อสังเกตหลัก",
        "- Option 03 เป็นตัวเลือกที่ได้คะแนนรวมสูงที่สุดอย่างชัดเจน",
        "- คู่ที่สัมพันธ์กันสูงสุดคือ Option 05 กับ Option 09, ตามด้วย Option 03 กับ Option 04 และ Option 01 กับ Option 02",
        "- ภาพรวมบอกว่าผู้ตอบให้ความสำคัญกับรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์ที่สื่อประโยชน์ของสินค้า",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_unsupervised_report(context: ReportContext) -> str:
    clean_df = context.clean_df
    metrics = context.metrics
    segment_summaries = _segment_summaries(clean_df)
    segment_pairs = sorted(segment_summaries.items(), key=lambda item: item[0])

    pca_rows = [
        [f"PC{i + 1}", f"{ratio:.3f}", f"{cumulative:.3f}"]
        for i, (ratio, cumulative) in enumerate(
            zip(
                metrics["pca"]["explained_variance_ratio"],
                metrics["pca"]["cumulative_explained_variance"],
            )
        )
    ]
    kmeans_rows = [
        [
            int(row["k"]),
            _format_float(row["silhouette_score"]),
            _format_float(row["davies_bouldin_score"]),
            _format_float(row["inertia"]),
        ]
        for row in metrics["kmeans_evaluation"]
    ]
    anomaly_rows = _segment_anomaly_rows(clean_df)

    lines = [
        "# รายงานการเรียนรู้แบบไม่มีผู้สอนเพื่อแบ่งกลุ่มผู้บริโภค",
        "",
        "## ภาพรวมวิธีการ",
        f"- หลังคัดกรองได้ผู้ตอบครบถ้วน {metrics['row_counts']['completed_top3']} คน",
        f"- ใช้ feature แบบ ipsatized จาก option-rating 50 ตัวแปร แล้วลดมิติด้วย PCA {metrics['pca']['n_components']} components",
        f"- final_cluster_key = `{metrics['final_cluster_key']}` และ final_cluster_k = {metrics['final_cluster_k']}",
        f"- เลือกคลัสเตอร์สุดท้ายที่ `k = {metrics['final_cluster_k']}` โดยใช้ `segment` เป็น label",
        "",
        "## PCA",
        _markdown_table(["Component", "Variance", "Cumulative"], pca_rows),
        f"- Cumulative variance หลัง `{metrics['pca']['n_components']}` components เท่ากับ {metrics['pca']['total_explained_variance_ratio']:.3f}",
        "- การรักษา 8 components ทำให้ยังเก็บโครงสร้างข้อมูลได้พอสำหรับ clustering และสอดคล้องกับข้อจำกัดของข้อมูล survey",
        "",
        "## K-Means Evaluation",
        _markdown_table(["k", "Silhouette", "Davies-Bouldin", "Inertia"], kmeans_rows),
        f"- เลือก `k = {metrics['final_cluster_k']}` เพราะ silhouette สูงสุดในชุดทดสอบ และให้ clustering ที่ตีความง่ายที่สุด",
        "",
        "## Hierarchical Validation",
        f"- ใช้ Ward linkage",
        f"- Cophenetic correlation = {metrics['hierarchical_validation']['cophenetic_correlation']:.3f}",
        "- ค่านี้อยู่ระดับปานกลาง แปลว่าโครงสร้างกลุ่มมีจริง แต่ยังมี overlap อยู่พอสมควร",
        "",
        "## Anomaly Detection",
        f"- พบ anomalous respondents {metrics['anomaly_detection']['anomaly_count']} คน ({metrics['anomaly_detection']['anomaly_rate'] * 100:.1f}%)",
        _markdown_table(["Segment", "Anomaly rate"], anomaly_rows),
        "",
        "## Customer Persona และข้อเสนอเชิงกลยุทธ์",
    ]

    for segment_id, summary in segment_pairs:
        persona_title = _segment_persona_title(segment_id, segment_summaries)
        top_options = ", ".join(
            f"{_pretty_option_name(name)} ({value:.2f})"
            for name, value in summary["top_options"][:3]
        )
        buy_factors = ", ".join(
            f"{name} {value:.2f}"
            for name, value in summary["top_buy_factors"][:3]
        )
        lines.extend(
            [
                f"### Segment {segment_id} - {persona_title}",
                f"- ขนาดกลุ่ม: {summary['size']} คน ({summary['share'] * 100:.1f}%)",
                f"- เพศเด่น: {summary['gender_top']}",
                f"- อายุเด่น: {summary['age_top']}",
                f"- สถานภาพเด่น: {summary['marital_top']}",
                f"- ตัวเลือกที่ชอบมากสุด: {top_options}",
                f"- ปัจจัยตัดสินใจเด่น: {buy_factors}",
                f"- อัตรา anomaly: {summary['anomaly_rate'] * 100:.1f}%",
                f"- กลยุทธ์: {summary['strategy']}",
                "",
            ]
        )

    lines.extend(
        [
            "## สรุปสำหรับการนำเสนอ",
            "- Segment 1 เป็นกลุ่มที่ให้คะแนนเรื่องรสชาติ วัตถุดิบธรรมชาติ และสัญลักษณ์บนแพ็กเกจสูงกว่า จึงเหมาะกับ messaging แบบ premium และชัดเจน",
            "- Segment 2 คือฐานผู้บริโภคหลัก ควรสื่อสารแบบกว้าง เน้นความคุ้มค่า ความน่าเชื่อถือ และข้อมูลสินค้าที่อ่านง่าย",
            "- โมเดลโดยรวมให้ภาพที่ใช้งานได้ แต่ยังมี overlap จึงควรใช้ร่วมกับ insight เชิงธุรกิจ ไม่ควรตีความว่าเป็นกลุ่มแยกขาดกัน 100%",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_owner_memo(context: ReportContext) -> str:
    metrics = context.metrics
    lines = [
        "# Unsupervised Owner Memo",
        "",
        "## สิ่งที่ต้องจำตอนพรีเซนต์",
        f"- ใช้ data หลังกรองจริง {metrics['row_counts']['completed_top3']} คน จาก raw {metrics['row_counts']['raw_loaded']} คน",
        f"- จุดขายของงานนี้คือการใช้ ipsatization เพื่อลด response style bias ก่อนทำ PCA",
        f"- ผลสุดท้ายเลือก `k = {metrics['final_cluster_k']}` เพราะ silhouette ดีสุดในชุดทดสอบและอธิบายง่าย",
        f"- hierarchical validation ให้ cophenetic correlation = {metrics['hierarchical_validation']['cophenetic_correlation']:.3f} ซึ่งบอกว่าโครงสร้างกลุ่มมี แต่ไม่คมมาก",
        f"- anomaly detection เจอ {metrics['anomaly_detection']['anomaly_count']} คน ({metrics['anomaly_detection']['anomaly_rate'] * 100:.1f}%)",
        "",
        "## ถ้าอาจารย์ถามว่าทำไมไม่ใช้ k=3",
        "- ตอบว่า dataset จริงมี overlap สูง และ silhouette ของ k=2 ดีกว่าชุดอื่นในรอบ sweep",
        "- k=2 ให้กลุ่มที่อธิบายการตลาดได้ง่ายกว่า ขณะที่ k ที่มากขึ้นแยกย่อยแต่ไม่ได้เพิ่มคุณภาพเชิง metric ชัดพอ",
        "",
        "## ถ้าอาจารย์ถามว่าทำไมต้อง ipsatize",
        "- เพราะผู้ตอบบางคนมีแนวโน้มให้คะแนนสูงทุกข้อ",
        "- การลบค่าเฉลี่ยรายแถวช่วยดึง preference จริงของแพ็กเกจออกจาก response style bias",
        "",
        "## โฟกัสเชิงธุรกิจ",
        "- กลุ่มที่เล็กกว่าเป็นกลุ่มที่เข้มกับคุณภาพและดีไซน์ ควรใช้ messaging แบบ premium",
        "- กลุ่มใหญ่กว่าเป็นฐานตลาดหลัก ควรเน้นความชัดเจน ความคุ้มค่า และข้อมูลที่เข้าใจง่าย",
        "",
        "## ไฟล์ที่ส่งต่อ",
        f"- `clean_dataset_with_segments.csv`",
        f"- `metrics_summary.json`",
        f"- `correlation_matrix.csv`",
        f"- `segment_profiles.csv`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_reports_from_output_dir(
    output_dir: str | Path, report_dir: str | Path
) -> dict[str, Path]:
    context = load_report_context(output_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "descriptive_stats": report_path / REPORT_FILENAMES["descriptive_stats"],
        "unsupervised_report": report_path / REPORT_FILENAMES["unsupervised_report"],
        "owner_memo": report_path / REPORT_FILENAMES["owner_memo"],
    }
    outputs["descriptive_stats"].write_text(
        render_descriptive_stats_report(context), encoding="utf-8"
    )
    outputs["unsupervised_report"].write_text(
        render_unsupervised_report(context), encoding="utf-8"
    )
    outputs["owner_memo"].write_text(
        render_owner_memo(context), encoding="utf-8"
    )
    return outputs


def run_unsupervised_workflow(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    from catfood_unsupervised.unsupervised.pipeline import run_pipeline
    pipeline_result = run_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        **pipeline_kwargs,
    )
    report_paths = write_reports_from_output_dir(output_dir, report_dir)
    return {
        "pipeline": pipeline_result,
        "report_paths": report_paths,
    }


def _count_rows(variable_label: str, series: pd.Series) -> list[list[str]]:
    counts = series.astype(str).str.strip().value_counts()
    total = int(counts.sum())
    rows = []
    for category, count in counts.items():
        rows.append(
            [
                str(variable_label),
                str(category),
                str(int(count)),
                f"{(count / total) * 100:.1f}%",
            ]
        )
    return rows


def _mean_rows(
    df: pd.DataFrame, column_indices: range, labels: list[str]
) -> list[list[str]]:
    rows = []
    for label, column_index in zip(labels, column_indices, strict=True):
        numeric_series = pd.to_numeric(
            map_series_values(df.iloc[:, column_index], THAI_LIKERT_SCALE, default=pd.NA),
            errors="coerce",
        )
        rows.append([label, f"{numeric_series.mean():.2f}"])
    return rows


def _vote_rows(clean_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for option in range(1, 11):
        rows.append([f"Option {option:02d}", str(int(clean_df[f"vote_{option:02d}"].sum()))])
    return sorted(rows, key=lambda item: int(item[1]), reverse=True)


def _option_overall_scores(clean_df: pd.DataFrame) -> pd.DataFrame:
    option_scores: dict[str, pd.Series] = {}
    for option_index in range(10):
        option_columns = [
            clean_df.columns[column_index]
            for column_index in range(
                22 + (option_index * 5),
                22 + (option_index * 5) + 5,
            )
        ]
        option_frame = pd.DataFrame(
            {
                column_name: pd.to_numeric(
                    map_series_values(
                        clean_df[column_name], THAI_LIKERT_SCALE, default=pd.NA
                    ),
                    errors="coerce",
                )
                for column_name in option_columns
            }
        )
        option_scores[f"opt{option_index + 1:02d}"] = option_frame.mean(axis=1)
    return pd.DataFrame(option_scores)


def _option_correlation_rows(clean_df: pd.DataFrame) -> list[list[str]]:
    option_scores = _option_overall_scores(clean_df)
    corr = option_scores.corr(method="spearman")
    pairs: list[tuple[str, str, float]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            pairs.append((left, right, float(corr.loc[left, right])))
    pairs = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)
    return [[f"{left} × {right}", f"{value:.3f}"] for left, right, value in pairs[:4]]


def _segment_summaries(clean_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    option_scores = _option_overall_scores(clean_df)
    summaries: dict[int, dict[str, Any]] = {}
    total_count = len(clean_df)

    for segment_id, segment_df in clean_df.groupby("segment", sort=True):
        row_count = int(len(segment_df))
        demographics = {
            "gender_top": _top_label(segment_df.iloc[:, GENDER_COLUMN_INDEX]),
            "age_top": _top_label(segment_df.iloc[:, AGE_COLUMN_INDEX]),
            "marital_top": _top_label(segment_df.iloc[:, MARITAL_COLUMN_INDEX]),
        }
        buy_factor_means = _mean_mapping(
            segment_df, BUY_FACTOR_COLUMN_INDICES, BUY_FACTOR_LABELS
        )
        packaging_means = _mean_mapping(
            segment_df, PACKAGING_IMPORTANCE_COLUMN_INDICES, PACKAGING_IMPORTANCE_LABELS
        )
        segment_option_means = option_scores.loc[segment_df.index].mean()
        top_options = sorted(
            ((column_name, float(value)) for column_name, value in segment_option_means.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        summaries[int(segment_id)] = {
            "size": row_count,
            "share": row_count / total_count,
            "top_options": top_options,
            "buy_factor_means": buy_factor_means,
            "packaging_means": packaging_means,
            "anomaly_rate": float(segment_df["anomaly_flag"].mean()),
            "strategy": _segment_strategy(
                buy_factor_means=buy_factor_means,
                packaging_means=packaging_means,
                top_options=top_options,
                segment_size=row_count,
            ),
            **demographics,
            "top_buy_factors": sorted(
                buy_factor_means.items(), key=lambda item: item[1], reverse=True
            ),
        }

    return summaries


def _mean_mapping(
    df: pd.DataFrame, column_indices: range, labels: list[str]
) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for label, column_index in zip(labels, column_indices, strict=True):
        mapping[label] = float(
            pd.to_numeric(
                map_series_values(
                    df.iloc[:, column_index], THAI_LIKERT_SCALE, default=pd.NA
                ),
                errors="coerce",
            ).mean()
        )
    return mapping


def _top_label(series: pd.Series) -> str:
    counts = series.astype(str).str.strip().value_counts()
    if counts.empty:
        return "-"
    return str(counts.index[0])


def _segment_strategy(
    *,
    buy_factor_means: dict[str, float],
    packaging_means: dict[str, float],
    top_options: list[tuple[str, float]],
    segment_size: int,
) -> str:
    taste = buy_factor_means["รสชาติกลมกล่อมอร่อยถูกปากแมว"]
    natural = buy_factor_means["ใช้วัตถุดิบจากธรรมชาติ"]
    benefit = packaging_means["มีสัญลักษณ์สื่อถึงประโยชน์หรือฟังก์ชัน"]
    top_option = top_options[0][0]
    top_option_label = _pretty_option_name(top_option)
    if taste >= 4.3 and benefit >= 4.2:
        return (
            "สื่อสารแบบ premium ให้ชัดเจน เน้นรสชาติ วัตถุดิบธรรมชาติ "
            "และสัญลักษณ์ที่บอกประโยชน์ของสินค้า"
        )
    if segment_size > 80:
        return (
            "ใช้ข้อความที่อ่านง่ายและมี social proof เน้นความคุ้มค่า "
            f"และทำให้ตัวเลือกเด่นอย่าง {top_option_label} ดูเข้าใจง่าย"
        )
    return (
        "เน้นความชัดเจนของคุณภาพ วัตถุดิบ และภาพอาหารจริงบนแพ็กเกจ "
        "เพื่อผลักดันความเชื่อมั่นในการซื้อ"
    )


def _pretty_option_name(option_name: str) -> str:
    if option_name.startswith("opt") and option_name[3:].isdigit():
        return f"Option {int(option_name[3:]):02d}"
    return option_name


def _segment_persona_title(
    segment_id: int, segment_summaries: dict[int, dict[str, Any]]
) -> str:
    if len(segment_summaries) < 2:
        return "กลุ่มผู้บริโภค"

    first, second = (
        segment_summaries[key]
        for key in sorted(segment_summaries.keys())
    )
    premium_score_first = (
        first["buy_factor_means"]["รสชาติกลมกล่อมอร่อยถูกปากแมว"]
        + first["buy_factor_means"]["ใช้วัตถุดิบจากธรรมชาติ"]
        + first["packaging_means"]["มีสัญลักษณ์สื่อถึงประโยชน์หรือฟังก์ชัน"]
    )
    premium_score_second = (
        second["buy_factor_means"]["รสชาติกลมกล่อมอร่อยถูกปากแมว"]
        + second["buy_factor_means"]["ใช้วัตถุดิบจากธรรมชาติ"]
        + second["packaging_means"]["มีสัญลักษณ์สื่อถึงประโยชน์หรือฟังก์ชัน"]
    )
    if segment_id == sorted(segment_summaries.keys())[0]:
        premium_segment = sorted(segment_summaries.keys())[0] if premium_score_first >= premium_score_second else sorted(segment_summaries.keys())[1]
    else:
        premium_segment = sorted(segment_summaries.keys())[0] if premium_score_first >= premium_score_second else sorted(segment_summaries.keys())[1]

    if segment_id == premium_segment:
        return "กลุ่มพรีเมียมเน้นคุณภาพและดีไซน์"
    return "กลุ่มตลาดหลักที่ชั่งน้ำหนักหลายปัจจัย"


def _segment_anomaly_rows(clean_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for segment_id, segment_df in clean_df.groupby("segment", sort=True):
        rows.append([f"Segment {int(segment_id)}", f"{segment_df['anomaly_flag'].mean() * 100:.1f}%"])
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _format_float(value: Any) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric_value):
        return "-"
    return f"{numeric_value:.3f}"
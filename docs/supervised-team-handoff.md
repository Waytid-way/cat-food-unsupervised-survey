# สรุนงาน Unsupervised Pipeline — สำหรับ Supervised Learning Team

## ข้อมูลหลัง Pipeline

| รายการ | ค่า |
|---|---|
| ข้อมูลดิบ (raw) | 167 คน |
| หลังกรอง (completed top-3) | **148 คน** |
| Features สำหรับ modeling | 50 ตัว (ipsatized + standardized) |
| มิติหลัง PCA | 8 components (68.6% variance) |
| Segment labels | `segment` column (1 หรือ 2) |
| Anomaly flags | `anomaly_flag` column (0 หรือ 1) |

## Segmentation Output

| Segment | ขนาด | คำอธิบาย |
|---|---|---|
| Segment 1 | 36 คน (24.3%) | Premium quality & design focused |
| Segment 2 | 112 คน (75.7%) | Mainstream market, multi-factor权衡 |

## ฟีเจอร์ที่อยู่ใน clean_dataset_with_segments.csv

ไฟล์ `outputs/clean_dataset_with_segments.csv` มีคอลัมน์หลักดังนี้:

**Metadata**
- `Timestamp`, `จากตัวเลือกทั้งหมด คุณชอบการออกแบบบรรจุภัณฑ์...` (top-3 choice)
- `เพศ`, `อายุ`, `สถานภาพสมรส`, `ผลของบรรจุภัณฑ์`

**Vote-weighted scores** (อันดับจาก top-3)
- `vote_01` – `vote_10`

**Buy factors** (Likert 1–5)
- `buy_factor_01` – `buy_factor_05`

**Packaging importance** (Likert 1–5)
- `packaging_importance_01` – `packaging_importance_08`

**Ipsatized option × attribute scores**
- `option_01_attribute_01_ips` – `option_10_attribute_05_ips` (50 features สำหรับ PCA)

**Demographic dummies**
- Age: `age_20-29ปี`, `age_30-39ปี`, `age_40-49ปี`, `age_50ปี ขึ้นไป`
- Gender: `gender_ชาย`, `gender_หญิง`, `gender_อื่นๆ`
- Marital: `marital_มีแฟนแต่ยังไม่แต่งงาน`, `marital_หย่าร้าง/เป็นม่าย`, `marital_แต่งงานแล้ว`, `marital_โสด ไม่มีแฟน`

**Clustering output**
- `segment` — 1 หรือ 2 (K-Means, k=2)
- `anomaly_flag` — 0 = normal, 1 = anomalous (Isolation Forest)

## ฟีเจอร์ที่แนะนำสำหรับ Supervised Learning

### แนะนำใช้ตรงๆ
- `vote_01` – `vote_10` (rank-weighted, พร้อมใช้เลย)
- `buy_factor_01` – `buy_factor_05`
- `packaging_importance_01` – `packaging_importance_08`
- `segment` (target สำหรับ classification)
- `anomaly_flag` (target สำหรับ binary classification หรือ filter)

### พร้อมสำหรับ one-hot encoding
- `เพศ`, `อายุ`, `สถานภาพสมรส`, `ผลของบรรจุภัณฑ์`

### ข้อมูล Ipsatized (สำหรับ advanced modeling)
- `option_*_attribute_*_ips` — 50 features หลัง ipsatization และ PCA reduction แล้ว

## Model Quality Metrics

| Metric | ค่า | ความหมาย |
|---|---|---|
| PCA cumulative variance | 68.6% (8 components) | เก็บโครงสร้างได้พอสมควร |
| Silhouette (k=2) | 0.193 | แยกกลุ่มได้บ้าง, มี overlap |
| Cophenetic correlation | 0.473 | Hierarchical structure มี แต่ไม่คม |
| Anomaly rate | 17.6% (26/148) | ตัวแปรผิดปกติ 1 ใน 6 |

## ข้อควรระวัง

1. **Segment overlap** — Silhouette 0.193 หมายถึง 2 กลุ่มยัง overlap กันพอสมควร อย่าคาดหวังว่า segment เป็นกลุ่มแยกขาดกัน 100%
2. **Anomaly detection** — 26 คนที่ flag ไว้ อาจเป็น noise หรือ genuinely anomalous; ควร review ก่อนใช้เป็น target
3. **Response style bias** — ข้อมูล Likert ผ่าน ipsatization แล้ว ความหมายเปลี่ยนจาก "absolute rating" เป็น "relative preference ภายในแต่ละคน"

## ถ้าต้องการใช้ Segment เป็น Target

```python
import pandas as pd
df = pd.read_csv("outputs/clean_dataset_with_segments.csv")
X = df[["vote_01","vote_02",...,"buy_factor_01",...]]
y = df["segment"]  # 1 หรือ 2
```

หรือถ้าต้องการ predict anomaly:

```python
y_anomaly = df["anomaly_flag"]  # 0 หรือ 1
```

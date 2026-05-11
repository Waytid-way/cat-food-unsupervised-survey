# GOAL.md — AI-Driven Marketing Campaign System
## Project AIE322 / AIE323 / AIE324 / AIE325 | BU

---

## บริบทธุรกิจ

บริษัท marketing agency กำลังออกแบบ packaging อาหารแมว 10 แบบ (opt01–opt10)
และต้องการตัดสินใจเชิงธุรกิจว่าควรลงทุนกับ design ไหน เพื่อเจาะกลุ่มลูกค้าไหน
ผ่าน channel ไหน ด้วย messaging แบบใด

ระบบที่ทีมกำลังสร้างคือ AI Decision Support System ที่แปลง raw survey data
→ actionable marketing intelligence โดยมี Web Dashboard เป็น interface ให้ทีม
business ใช้งานจริง

---

## Dataset

| รายการ | ค่า |
|---|---|
| ไฟล์ | BU-Data-from-Survey-Cases_final-5-2.csv |
| Raw data rows | 167 rows (ไม่นับ empty rows จาก Excel export) |
| Final valid respondents | **n = 148** (กรองเฉพาะผู้ตอบ top3_choice ครบ) |
| ทุกคนใน dataset | cat_owner = 'เคย' (เคยซื้ออาหารแมวมาก่อน) |
| Output file | clean_dataset_with_segments.csv (148 × 147 cols) |

**หมายเหตุ CSV:** ไฟล์มี 1,001 rows แต่มี data จริงเพียง 167 rows
ที่เหลือคือ empty rows ที่ Excel เติมมาอัตโนมัติ — ignore ทั้งหมด
CSV มี 2 header rows (row 0 = metadata "Brief", row 1 = column names)
data เริ่ม row 2

---

## 3 คำถามธุรกิจที่ระบบต้องตอบได้

1. ลูกค้าของเราแบ่งเป็นกี่กลุ่ม แต่ละกลุ่มมีหน้าตาอย่างไร และชอบ packaging แบบไหน
   → ตอบด้วย Unsupervised Learning

2. ถ้ามีลูกค้าใหม่เข้ามา เราสามารถ predict ได้ไหมว่าเขาอยู่กลุ่มไหน
   → ตอบด้วย Supervised Learning

3. ข้อมูลทั้งหมดนี้แปลเป็นกลยุทธ์การตลาดที่ลงมือทำได้จริงอย่างไร
   → ตอบด้วย Business Insight Dashboard

---

## System Pipeline

```
Raw Survey Data (167 rows)
        │
        ▼
[Data Preparation]
กรอง + Encode + Ipsatize + KNN Impute
→ n=148, clean_dataset_with_segments.csv
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
[UNSUPERVISED]                   [SUPERVISED]
PCA → K-Means                    รับ segment labels
→ Hierarchical                   จาก Unsupervised
→ Isolation Forest               → train classifier
        │                                 │
        ▼                                 ▼
Segment Labels (0/1)          Best Model + Reports
Customer Personas               Feature Importance
Anomaly Flags                   Confusion Matrix
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
          [WEB DASHBOARD — Dash/Python]
          Home | Unsupervised | Supervised | Business Insight
                       │
                       ▼
          Strategic Recommendations per Segment
```

---

## ผลลัพธ์ที่ได้จาก Unsupervised (เสร็จแล้ว ✅)

### Key Numbers

| ตัวชี้วัด | ค่า |
|---|---|
| n final | 148 |
| PC1–PC5 cumulative variance | 52.2% (หลัง ipsatize) |
| PC1 ก่อน ipsatize | 31.7% (response style bias) |
| PC1 หลัง ipsatize | 15.1% (genuine preference) |
| K-Means k เลือก | **2** (Silhouette=0.193, DB=2.032) |
| Cophenetic r (Hierarchical) | 0.473 (moderate) |
| Anomalies (Isolation Forest) | **26 คน (17.6%)** |
| Total variance (8 PCs) | 68.6% |

### 2 Customer Segments

| Segment | Persona | n | สัดส่วน | Anomaly rate |
|---|---|---|---|---|
| 0 | The Premium Quality Seeker | **36** | **24.3%** | **19.4%** |
| 1 | The Main Market | **112** | **75.7%** | **17.0%** |

**Segment 0 — The Premium Quality Seeker (n=36)**
- Profile: หญิง 30–39 ปี โสดไม่มีแฟน
- Buy driver: รสชาติ (4.61) + แบรนด์ดัง (4.50) + ธรรมชาติ (4.31)
- Top choice: opt03 (4.31) > opt04 (3.96) > opt02 (3.64)
- Brand sensitivity: สูง
- Strategy: Premium messaging — รสชาติ ธรรมชาติ แบรนด์ดัง สัญลักษณ์บอกประโยชน์

**Segment 1 — The Main Market (n=112)**
- Profile: หญิง 30–39 ปี แต่งงานแล้ว
- Buy driver: รสชาติ (4.38) + ธรรมชาติ (4.25) + แบรนด์ (3.92)
- Top choice: opt03 (3.62) > opt06 (3.43) > opt07 (3.39)
- Brand sensitivity: ปานกลาง
- Strategy: Value messaging — คุ้มค่า น่าเชื่อถือ ข้อมูลชัดเจน

### Option Vote Ranking (Weighted: rank1=3, rank2=2, rank3=1)

| Option | Score | อันดับ |
|---|---|---|
| opt03 | 251 | 1 (ห่างอันดับ 2 เกือบ 2×) |
| opt02 | 138 | 2 |
| opt01 | 114 | 3 |
| opt06 | 104 | 4 |
| opt04 | 68 | 5 |
| opt07 | 66 | 6 |
| opt05 | 55 | 7 |
| opt08 | 46 | 8 |
| opt09 | 27 | 9 |
| opt10 | 19 | 10 |

**⚠ ข้อควรระวัง:** opt03 อาจมี order/anchoring effect ต้องนำ Likert analysis
ประกอบก่อนสรุปว่าเป็น genuine preference อย่างแท้จริง

### Buy Factors (mean /4, n=148)

| Factor | Mean |
|---|---|
| รสชาติ (bf_taste) | **4.43** |
| วัตถุดิบธรรมชาติ (bf_natural) | **4.26** |
| แบรนด์ดัง (bf_famous_brand) | 4.06 |
| วัตถุดิบนำเข้า (bf_imported) | 3.24 |
| สินค้านำเข้า (bf_foreign_brand) | 3.16 |

---

## งานของ Supervised Learning (ยังไม่เสร็จ ⏳)

### Input จาก Unsupervised
- ไฟล์: clean_dataset_with_segments.csv
- Target variable: column `segment` (0, 1)
- กรอง `anomaly_flag == 1` ออกก่อน train ทุกครั้ง
- ใช้ stratified split เพราะ class imbalance (36:112)

### Features สำหรับ train

```python
feature_cols = [
    'age_enc', 'gender_enc', 'marital_enc',
    'bf_natural', 'bf_imported', 'bf_taste',
    'bf_foreign_brand', 'bf_famous_brand',
    'pkg_premium_look', 'pkg_cat_image', 'pkg_food_image',
    'pkg_ingredient_image', 'pkg_eco_friendly',
    'pkg_source_symbol', 'pkg_benefit_symbol', 'pkg_certified',
    'pkg_style_enc', 'pkg_inf_enc'
]
```

### Models ที่ต้องเปรียบเทียบ

| Model | เหตุผล |
|---|---|
| Logistic Regression | Baseline, ตีความ coefficient ได้ตรง |
| Random Forest | Non-linear + feature importance |
| Gradient Boosting | มักให้ accuracy สูงสุดบน tabular data |
| SVM (RBF kernel) | เหมาะกับ high-dimensional space |

### Metrics ที่ต้อง report
- Accuracy, Precision, Recall, F1 per class
- Confusion Matrix
- ROC Curve (One-vs-Rest)
- Feature Importance (RF/GBM)

### Output ที่ส่งให้ Dashboard
- best_model.pkl (saved model สำหรับ prediction)
- model_comparison.csv (ผล metrics ทุก model)
- feature_importance.csv
- confusion_matrix.png

---

## งานของ Dashboard (ยังไม่เสร็จ ⏳)

### 4 หน้าที่ต้องมี

**Home:**
Project overview, business problem, key stats
(n=148, 2 segments, top option=opt03, best model accuracy)

**Unsupervised Learning:**
PCA scree plot, 2D PCA scatter (สี = segment),
Elbow curve + silhouette, Dendrogram,
Anomaly distribution, Segment heatmap

**Supervised Learning:**
Model comparison bar chart, Confusion matrix,
Precision/Recall/F1 per class, Feature importance,
ROC curve

**Business Insight:**
3 Persona cards (visual), Option scorecard,
Packaging attribute importance per segment,
Strategic recommendation ต่อ segment

### Tech Stack
- Python + Dash + dash-bootstrap-components
- Plotly สำหรับ charts ทุกตัว
- SQLite หรือ Firebase สำหรับ data storage
- Pandas + Sklearn + Scipy

---

## Column Naming Convention (ห้ามเปลี่ยน)

```python
opt_attr_names = ['want_to_buy', 'stand_out', 'premium_quality',
                  'taste_feel', 'fit_for_me']

opt_cols  = [f'opt{i:02d}_{a}' for i in range(1,11)
             for a in opt_attr_names]  # 50 cols

bf_cols   = ['bf_natural', 'bf_imported', 'bf_taste',
             'bf_foreign_brand', 'bf_famous_brand']

pkg_cols  = ['pkg_premium_look', 'pkg_cat_image', 'pkg_food_image',
             'pkg_ingredient_image', 'pkg_eco_friendly',
             'pkg_source_symbol', 'pkg_benefit_symbol', 'pkg_certified']

demo_cols = ['age_enc', 'gender_enc', 'marital_enc',
             'pkg_style_enc', 'pkg_inf_enc']

# Ipsatized columns
ips_cols  = [f'ips_{c}' for c in opt_cols]  # 50 cols

# PCA output
pca_cols  = [f'PC{i}' for i in range(1, 9)]  # PC1–PC8

# Vote columns (weighted: rank1=3, rank2=2, rank3=1)
vote_cols = [f'vote_{i:02d}' for i in range(1, 11)]
```

---

## Encoding Reference

```python
opt_map  = {'ไม่เห็นด้วยเลย':1, 'ไม่เห็นด้วย':2,
            'เฉยๆ':3, 'เห็นด้วย':4, 'เห็นด้วยที่สุด':5}

pkg_map  = {'น้อยที่สุด':1, 'น้อย':2, 'ปานกลาง':3,
            'มาก':4, 'มากที่สุด':5}

bf_map   = {'น้อย':1, 'ปานกลาง':2, 'มาก':3, 'มากที่สุด':4}

age_map  = {'20-29ปี':1, '30-39ปี':2,
            '40-49ปี':3, '50ปี ขึ้นไป':4}

gender_map  = {'หญิง':0, 'ชาย':1, 'อื่นๆ':2}

marital_map = {'โสด ไม่มีแฟน':1,
               'มีแฟนแต่ยังไม่แต่งงาน':2,
               'แต่งงานแล้ว':3,
               'หย่าร้าง/เป็นม่าย':4}
```

---

## Open Questions / Risks

1. **opt03 order effect:** ยังไม่ได้ verify ว่า dominance ของ opt03 (251 pts)
   มาจาก genuine preference หรือ anchoring/order bias จากตำแหน่งในแบบสอบถาม

2. **Segment 1 anomaly 23.1%:** สูงผิดปกติ — ยังไม่ชัดว่าเป็น heterogeneous
   taste จริง หรือ response quality ต่ำในกลุ่มนี้

3. **Cophenetic r = 0.473:** Hierarchical ไม่ validate K-Means ได้เต็มที่
   ต้อง acknowledge เป็น limitation ใน report อย่างชัดเจน
   อย่าเขียนว่า "hierarchical confirms k=2" โดยไม่ mention ค่านี้

4. **Sensitivity analysis:** ยังไม่ได้รัน K-Means include vs exclude anomalies
   เพื่อ confirm cluster stability

5. **Class imbalance (36:112):** Supervised ต้องใช้ stratified split
   และ report per-class metrics ไม่ใช่แค่ overall accuracy

---

## กำหนดส่งงาน

**Deadline: 12 พฤษภาคม 2026 เวลา 23:59 น.**
ช่องทาง: Google Form (link ประกาศใน LINE กลุ่ม + MS Teams)

### 3 ชิ้นงานที่ต้องส่ง

**1. Presentation (PDF + Canva link)**
แยกหัวข้อ Supervised / Unsupervised อย่างชัดเจน

**2. รายงานสรุป Source Code**
โครงสร้างโปรแกรม, Software Diagram/Workflow,
คำอธิบายแต่ละไฟล์ Code, ขั้นตอนการทำงาน

**3. Source Code ทั้งหมด**
จัดโฟลเดอร์เป็นระเบียบ, มี requirements.txt,
มี README.md อธิบาย setup + run steps,
ใช้ relative path ทุกที่ (ไม่ hardcode path ของเครื่องตัวเอง)

### Priority 2 วันที่เหลือ

| วัน | งาน |
|---|---|
| 10 พ.ค. (วันนี้) | Unsupervised: finalize reports ทั้ง 4 ชิ้น + clean code → ส่งทีม |
| 11 พ.ค. | Supervised: train + evaluate / Dashboard: build 4 pages (ทำคู่กัน) |
| 12 พ.ค. ก่อนเที่ยง | ประกอบ slides + รวม source code report + test run end-to-end |
| 12 พ.ค. ก่อน 23:59 | Submit Google Form |

---

## Folder Structure แนะนำ

```
project/
├── data/
│   ├── raw/
│   │   └── BU-Data-from-Survey-Cases_final-5-2.csv
│   └── processed/
│       └── clean_dataset_with_segments.csv
├── unsupervised/
│   ├── preprocessing.py
│   ├── pca_clustering.py
│   └── anomaly_detection.py
├── supervised/
│   ├── train_classifier.py
│   └── model_evaluation.py
├── dashboard/
│   ├── app.py
│   ├── pages/
│   │   ├── home.py
│   │   ├── unsupervised.py
│   │   ├── supervised.py
│   │   └── business_insight.py
│   └── components/
├── outputs/
│   ├── models/
│   └── figures/
├── requirements.txt
└── README.md
```

---

*อัปเดตล่าสุด: 10 พฤษภาคม 2026*
*จัดทำโดย: ML Engineer (Unsupervised Learning)*

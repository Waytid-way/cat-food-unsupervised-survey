# รายงาน Data Preprocessing สำหรับโครงการ AI-Driven Marketing Campaign System

---

## ส่วนหน้า

### ปก

**โครงการ AI-Driven Marketing Campaign System**

**รายงาน Data Preprocessing**

---

### คำนำ

รายงานฉบับนี้จัดทำขึ้นเพื่อบันทึกและอธิบายกระบวนการเตรียมข้อมูล (Data Preprocessing) ที่ใช้ในโครงการ AI-Driven Marketing Campaign System ซึ่งเป็นระบบวิเคราะห์ข้อมูลการตลาดสำหรับผลิตภัณฑ์อาหารแมวสำเร็จรูป โดยกระบวนการเตรียมข้อมูลเป็นขั้นตอนที่มีความสำคัญอย่างยิ่งในการเตรียมข้อมูลดิบจากแบบสำรวจให้พร้อมสำหรับการวิเคราะห์และสร้างแบบจำลอง Machine Learning ทั้งในส่วน Unsupervised Learning (Clustering, Anomaly Detection, PCA) และ Supervised Learning (Classification)

วัตถุประสงค์หลักของการเตรียมข้อมูลในโครงการนี้ครอบคลุมการจัดการค่าที่หายไป (Missing Values) การลบข้อมูลซ้ำ (Duplicate Removal) การแปลงประเภทข้อมูล (Data Type Conversion) การเข้ารหัสตัวแปรเชิงหมวดหมู่ (Categorical Encoding) การสร้างตัวแปรใหม่ (Feature Engineering) และการปรับขนาดตัวแปร (Feature Scaling) เพื่อให้ข้อมูลอยู่ในรูปแบบที่เหมาะสมสำหรับการนำไปใช้งานในขั้นตอนถัดไป

---

### สารบัญ

- บทที่ 1 วัตถุประสงค์ของการเตรียมข้อมูล
- บทที่ 2 ข้อมูลที่ใช้ในโครงการ
- บทที่ 3 Data Cleaning
  - 3.1 ตรวจสอบข้อมูลที่หายไป
  - 3.2 การลบข้อมูลซ้ำ
  - 3.3 การปรับรูปแบบข้อมูล
- บทที่ 4 Encoding Categorical Variables
- บทที่ 5 Feature Engineering
- บทที่ 6 Feature Scaling
- บทที่ 7 ผลลัพธ์หลังการเตรียมข้อมูล
- บทที่ 8 สรุป

---

## เนื้อหาหลัก

## บทที่ 1 วัตถุประสงค์ของการเตรียมข้อมูล

การเตรียมข้อมูล (Data Preprocessing) เป็นกระบวนการที่มีความสำคัญอย่างยิ่งในโครงการ AI-Driven Marketing Campaign System เนื่องจากข้อมูลดิบที่ได้จากแบบสำรวจออนไลน์มักมีความไม่สมบูรณ์ มีค่าที่หายไป และอยู่ในรูปแบบที่ไม่เหมาะสมสำหรับการนำไปวิเคราะห์โดยตรง การเตรียมข้อมูลที่ถูกต้องจึงเป็นรากฐานสำคัญที่ส่งผลต่อคุณภาพของผลลัพธ์ที่ได้จากแบบจำลอง Machine Learning ทั้งในส่วน Unsupervised Learning (Clustering, Anomaly Detection, PCA) และ Supervised Learning (Classification)

วัตถุประสงค์หลักของการเตรียมข้อมูลในโครงการนี้ประกอบด้วย 6 ประเด็นหลักดังนี้

**ประการที่ 1** การจัดการค่าที่หายไป (Missing Value Handling) — แบบสำรวจออนไลน์มักมีคำถามที่ผู้ตอบไม่ได้ตอบ หรือตอบคำถามบางส่วนแล้วหยุดไป ระบบจำเป็นต้องมีกลยุทธ์ในการจัดการกับค่าที่หายไปเหล่านี้อย่างเหมาะสม เช่น การใช้ KNN Imputation สำหรับข้อมูลเชิงปริมาณ เพื่อไม่ให้ข้อมูลสูญเสียปริมาณมากเกินไป

**ประการที่ 2** การกรองข้อมูลที่ไม่สมบูรณ์ (Filtering Incomplete Responses) — ระบบกรองเฉพาะการตอบที่สมบูรณ์ โดยเฉพาะคำถาม Top-3 Preferences ที่เป็นข้อมูลหลักในการวิเคราะห์ ข้อมูลที่ไม่สมบูรณ์จะถูกคัดออกก่อนเข้าสู่กระบวนการวิเคราะห์

**ประการที่ 3** การแปลงข้อมูลเชิงคุณภาพเป็นเชิงปริมาณ (Categorical to Numerical Conversion) — คำตอบในแบบสำรวจส่วนใหญ่อยู่ในรูปแบบข้อความภาษาไทย เช่น "มากที่สุด" "มาก" "ปานกลาง" "น้อย" "น้อยที่สุด" ระบบจำเป็นต้องแปลงข้อมูลเหล่านี้เป็นตัวเลข (1-5) ก่อนนำไปวิเคราะห์

**ประการที่ 4** การสร้างตัวแปรใหม่ (Feature Engineering) — การสร้างตัวแปรใหม่จากข้อมูลดิบ เช่น Vote Features ที่คำนวณน้ำหนักจากการจัดอันดับ Top-3 และ Ipsatized Ratings ที่ปรับค่าให้เป็นอิสระต่อกันแต่ละแถว เพื่อให้ได้ตัวแปรที่มีความหมายทางธุรกิจมากขึ้น

**ประการที่ 5** การปรับขนาดตัวแปร (Feature Scaling) — ตัวแปรต่างๆ มีมาตราส่วนที่แตกต่างกัน ระบบใช้ StandardScaler เพื่อปรับให้อยู่ในมาตราส่วนเดียวกันก่อนนำไปทำ PCA ซึ่งเป็นขั้นตอนสำคัญในการลดมิติข้อมูล

**ประการที่ 6** การเตรียมข้อมูลสำหรับ Supervised Learning — ในส่วน Supervised Learning ระบบต้องเตรียมข้อมูลให้อยู่ในรูปแบบที่พร้อมสำหรับการสร้าง Classification Model โดยใช้ OneHotEncoder และ Pipeline ของ sklearn

---

## บทที่ 2 ข้อมูลที่ใช้ในโครงการ

### 2.1 ชุดข้อมูลที่พบในโปรเจกต์

โครงการ AI-Driven Marketing Campaign System ประมวลผลข้อมูลจากแบบสำรวจออนไลน์เกี่ยวกับพฤติกรรมและความชอบของผู้บริโภคที่มีประสบการณ์เลี้ยงแมวในการเลือกซื้ออาหารแมวสำเร็จรูปชนิดเม็ด โดยมีชุดข้อมูลหลักและชุดข้อมูลผลลัพธ์ดังนี้

| ชื่อไฟล์ | ประเภท | คำอธิบาย |
|---|---|---|
| `BU Data from Survey Cases_final(5).csv` | Raw Data | ไฟล์ข้อมูลดิบจากแบบสำรวจออนไลน์ |
| `outputs/clean_dataset_with_segments.csv` | Processed Data | ข้อมูลที่ผ่านการทำความสะอาดแล้ว พร้อมคอลัมน์ segment |
| `outputs/correlation_matrix.csv` | Analysis Output | เมทริกซ์สหสัมพันธ์ระหว่างตัวแปร |
| `outputs/segment_profiles.csv` | Analysis Output | โปรไฟล์ของแต่ละ Segment |
| `outputs/supervised/model_comparison.csv` | Model Output | ผลเปรียบเทียบระหว่าง Classification Models |
| `outputs/supervised/predictions.csv` | Model Output | ผลการ predict ของ best model |

### 2.2 โครงสร้างข้อมูลดิบ

ไฟล์ข้อมูลดิบ `BU Data from Survey Cases_final(5).csv` มีโครงสร้างที่ประกอบด้วย 76 คอลัมน์ โดยสามารถจำแนกประเภทข้อมูลได้ดังนี้

**คอลัมน์ข้อมูลประชากรศาสตร์ (Demographics)**

- อายุ (Age): คอลัมน์ที่ 73
- เพศ (Gender): คอลัมน์ที่ 74
- สถานภาพสมรส (Marital Status): คอลัมน์ที่ 75

**คอลัมน์ปัจจัยการตัดสินใจซื้อ (Buy Factors)**

- คอลัมน์ที่ 5-9: ข้อมูลปัจจัยที่มีผลต่อการตัดสินใจซื้อ 5 ข้อ เช่น วัตถุดิบจากธรรมชาติ เนื้อปลาทูน่านำเข้าจากญี่ปุ่น รสชาติ แบรนด์ และประเทศต้นทาง

**คอลัมน์บรรจุภัณฑ์ (Packaging)**

- คอลัมน์ที่ 10: คำถามว่าบรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อหรือไม่
- คอลัมน์ที่ 12-20: ระดับความสำคัญของคุณลักษณะบรรจุภัณฑ์ 8 ข้อ

**คอลัมน์การจัดอันดับตัวเลือก (Option Ratings)**

- คอลัมน์ที่ 22-71: การประเมินตัวเลือกบรรจุภัณฑ์ 10 ตัวเลือก ต่อ 5 แอตทริบิวต์ รวม 50 คอลัมน์

**คอลัมน์ Top-3 Preferences**

- คอลัมน์ที่ 72: คำถามให้เลือกบรรจุภัณฑ์ที่ชอบที่สุด 3 อันดับแรก

### 2.3 ข้อมูลที่ได้จากการประมวลผล

ไฟล์ `clean_dataset_with_segments.csv` ที่เป็นผลลัพธ์หลังการประมวลผล ประกอบด้วยคอลัมน์ที่สร้างขึ้นใหม่ดังนี้

- คอลัมน์ `top3_rank_1`, `top3_rank_2`, `top3_rank_3`: การแปลงข้อมูล Top-3 จากข้อความเป็นตัวเลข
- คอลัมน์ `vote_01` ถึง `vote_10`: คะแนนน้ำหนักสะสมจากการจัดอันดับ
- คอลัมน์ `PC1` ถึง `PC8`: Principal Component Scores จาก PCA
- คอลัมน์ `segment`: หมายเลข Cluster ที่ KMeans จัดให้
- คอลัมน์ `anomaly_flag` และ `anomaly_score`: ผลจาก Isolation Forest

---

## บทที่ 3 Data Cleaning

### 3.1 ตรวจสอบข้อมูลที่หายไป

#### 3.1.1 การกรองคำตอบที่ไม่สมบูรณ์ (Non-Empty Filtering)

ก่อนเข้าสู่กระบวนการวิเคราะห์ ระบบจะกรองเฉพาะคำตอบที่สมบูรณ์ โดยเฉพาะคำถาม Top-3 Preferences ซึ่งเป็นคำถามหลักในการวิเคราะห์ ฟังก์ชัน `filter_completed_responses` ใน `data_loading.py:37-40` จะทำการตรวจสอบว่าคอลัมน์ Top-3 มีค่าไม่ว่างเปล่าและไม่ใช่สตริงว่าง คำตอบที่ไม่ผ่านการตรวจสอบจะถูกคัดออก

```python
def filter_completed_responses(
    df: pd.DataFrame, top3_column: str
) -> pd.DataFrame:
    return df.loc[_non_empty_mask(df[top3_column])].reset_index(drop=True)

def _non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")
```

#### 3.1.2 การ Impute ค่าที่หายไปสำหรับ Buy Factors ด้วย KNN Imputer

สำหรับคอลัมน์ปัจจัยการตัดสินใจซื้อ (Buy Factors) ที่มีค่าที่หายไป ระบบใช้กลยุทธ์ KNN Imputation โดยใช้ `KNNImputer` จาก sklearn กับพารามิเตอร์ `n_neighbors=5` ซึ่งเป็นวิธีที่เหมาะสมสำหรับข้อมูลเชิงปริมาณที่มีความสัมพันธ์ระหว่างตัวแปร ตามฟังก์ชัน `impute_buy_factors` ใน `preprocessing.py:55-77`

```python
def impute_buy_factors(
    df: pd.DataFrame, columns: Sequence[str], n_neighbors: int = 5
) -> pd.DataFrame:
    selected_columns = list(columns)
    if not selected_columns:
        return pd.DataFrame(index=df.index)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    buy_factor_frame = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    if buy_factor_frame.empty:
        return pd.DataFrame(index=df.index, columns=selected_columns, dtype=float)

    output = buy_factor_frame.astype(float).copy()
    imputable_columns = output.columns[output.notna().any(axis=0)].tolist()
    if not imputable_columns:
        return output

    effective_neighbors = min(n_neighbors, len(buy_factor_frame))
    imputer = KNNImputer(n_neighbors=effective_neighbors)
    imputed_values = imputer.fit_transform(output.loc[:, imputable_columns])
    output.loc[:, imputable_columns] = imputed_values
    return output
```

#### 3.1.3 การ Impute ค่าที่หายไปสำหรับ Supervised Features

ในส่วน Supervised Learning ระบบใช้การเติมค่าเริ่มต้นด้วยสตริง `"__missing__"` สำหรับค่าที่หายไป แล้วแปลงเป็นประเภท string ทั้งหมด ก่อนนำไปผ่าน OneHotEncoder ตามฟังก์ชัน `build_supervised_feature_frame` ใน `src/catfood_unsupervised/supervised/features.py:22-31`

```python
def build_supervised_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")
    if ANOMALY_COLUMN not in df.columns:
        raise ValueError(f"Missing required anomaly column: {ANOMALY_COLUMN}")

    feature_columns = get_supervised_feature_columns(df)
    X = df.loc[:, feature_columns].copy().fillna("__missing__").astype(str)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int).rename(TARGET_COLUMN)
    return X, y
```

#### 3.1.4 การจัดการค่าที่หายไปสำหรับ Ipsatized Ratings

หลังจากการคำนวณ Ipsatized Option Ratings โดยฟังก์ชัน `ipsatize_rows` ค่าที่หายไปจะถูกเติมด้วย `0.0` เนื่องจากการ Ipsatization คำนวณจากค่าเฉลี่ยต่อแถว ค่าที่หายไปจะส่งผลต่อการคำนวณค่าเฉลี่ย แต่ในท้ายที่สุดจะถูกแทนที่ด้วย 0.0 เพื่อให้อยู่ในรูปแบบที่พร้อมสำหรับ PCA ตามโค้ดใน `pipeline.py:184-188`

```python
ipsatized_option_ratings = pd.DataFrame(
    ipsatize_rows(option_ratings.to_numpy()),
    index=completed_df.index,
    columns=[f"{column}_ips" for column in option_ratings.columns],
).fillna(0.0)
```

### 3.2 การลบข้อมูลซ้ำ

ระบบไม่มีขั้นตอนการลบข้อมูลซ้ำ (Dedup) โดยตรง เนื่องจากแบบสำรวจออนไลน์ไม่อนุญาตให้ผู้ตอบตอบซ้ำได้ อย่างไรก็ตาม ระบบมีการกรองข้อมูลที่ไม่สมบูรณ์ออก 2 ระดับ ดังนี้

**ระดับที่ 1: กรองคำตอบที่ไม่สมบูรณ์** — ฟังก์ชัน `filter_completed_responses` กรองเฉพาะแถวที่มีคำตอบในคอลัมน์ Top-3 Preferences แถวที่ว่างเปล่าหรือมีแค่ช่องว่างจะถูกคัดออก

**ระดับที่ 2: กรองคำตอบที่มี Anomaly (Supervised Pipeline)** — ใน Supervised Pipeline ฟังก์ชัน `load_supervised_dataset` จะกระดับแถวที่มี `anomaly_flag != 0` ออก เนื่องจากแถวเหล่านี้ถูกระบุว่าเป็น Outlier จากกระบวนการ Anomaly Detection

```python
def load_supervised_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, ANOMALY_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    anomaly_flags = pd.to_numeric(df[ANOMALY_COLUMN], errors="coerce")
    filtered = df.loc[anomaly_flags.eq(0)].copy()
    filtered[TARGET_COLUMN] = pd.to_numeric(filtered[TARGET_COLUMN], errors="raise").astype(int)
    return filtered.reset_index(drop=True)
```

### 3.3 การปรับรูปแบบข้อมูล

#### 3.3.1 การแปลง Likert Scale จากข้อความเป็นตัวเลข

แบบสำรวจใช้ Likert Scale เป็นข้อความภาษาไทย ระบบแปลงเป็นตัวเลข 1-5 ตามพจนานุกรม `THAI_LIKERT_SCALE` ใน `preprocessing.py:10-21`

```python
THAI_LIKERT_SCALE = {
    "น้อยที่สุด": 1,
    "น้อย": 2,
    "ปานกลาง": 3,
    "มาก": 4,
    "มากที่สุด": 5,
    "ไม่เห็นด้วยเลย": 1,
    "ไม่เห็นด้วย": 2,
    "เฉยๆ": 3,
    "เห็นด้วย": 4,
    "เห็นด้วยที่สุด": 5,
}
```

#### 3.3.2 การแปลง Demographic Data

ข้อมูลประชากรศาสตร์ (อายุ เพศ สถานภาพสมรส) ถูกแปลงจากข้อความเป็น One-Hot Encoded columns โดยฟังก์ชัน `_encode_demographics` ใน `pipeline.py:264-278`

```python
def _encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    demographic_frame = pd.DataFrame(
        {
            "age": df.iloc[:, AGE_COLUMN_INDEX].astype(str).str.strip(),
            "gender": df.iloc[:, GENDER_COLUMN_INDEX].astype(str).str.strip(),
            "marital": df.iloc[:, MARITAL_COLUMN_INDEX].astype(str).str.strip(),
        },
        index=df.index,
    )
    return pd.get_dummies(
        demographic_frame,
        columns=["age", "gender", "marital"],
        prefix=["age", "gender", "marital"],
        dtype=int,
    )
```

#### 3.3.3 การแปลง Top-3 Rankings

ข้อมูล Top-3 Preferences ที่อยู่ในรูปข้อความ เช่น `"Option 3, Option 6, Option 8"` ถูกแปลงเป็น DataFrame ที่มี 3 คอลัมน์ (`rank1`, `rank2`, `rank3`) โดยฟังก์ชัน `_extract_top3_rankings` ใน `pipeline.py:217-232`

```python
TOP3_PATTERN = re.compile(r"Option\s*(\d+)", flags=re.IGNORECASE)

def _extract_top3_rankings(top3_series: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for row_number, value in enumerate(top3_series.tolist()):
        options = TOP3_PATTERN.findall("" if pd.isna(value) else str(value))
        if len(options) < 3:
            raise ValueError(
                f"Expected at least 3 option references in top-3 response row {row_number}."
            )
        rows.append(
            {
                "rank1": int(options[0]),
                "rank2": int(options[1]),
                "rank3": int(options[2]),
            }
        )
    return pd.DataFrame(rows, index=top3_series.index)
```

---

## บทที่ 4 Encoding Categorical Variables

โครงการนี้ใช้วิธี Encoding หลายรูปแบบตามประเภทของข้อมูล ซึ่งสามารถจำแนกได้ดังนี้

### 4.1 Label Encoding สำหรับ Likert Scale

สำหรับคำตอบแบบ Likert Scale ที่เป็นข้อความภาษาไทย ระบบใช้ Label Encoding โดยการ Map ข้อความเป็นตัวเลข 1-5 ผ่านพจนานุกรม `THAI_LIKERT_SCALE` ที่ประกาศไว้ใน `preprocessing.py:10-21` และ `IMPORTANCE_MAPPING` ที่ประกาศไว้ใน `pipeline.py:41-58`

| ข้อความไทย | ค่าตัวเลข |
|---|---|
| น้อยที่สุด / ไม่เห็นด้วยเลย | 1 |
| น้อย / ไม่เห็นด้วย | 2 |
| ปานกลาง / เฉยๆ | 3 |
| มาก / เห็นด้วย | 4 |
| มากที่สุด / เห็นด้วยที่สุด | 5 |

การแปลงนี้ทำผ่านฟังก์ชัน `map_series_values` ที่ใช้ `.map()` บน pandas Series

### 4.2 One-Hot Encoding สำหรับ Demographics

ข้อมูลประชากรศาสตร์ (อายุ เพศ สถานภาพสมรส) ถูกแปลงเป็น One-Hot Encoded columns โดยใช้ `pd.get_dummies()` ผ่านฟังก์ชัน `_encode_demographics` ตัวอย่างผลลัพธ์ที่ได้ เช่น คอลัมน์ `age_20-29ปี`, `age_30-39ปี`, `gender_ชาย`, `gender_หญิง`, `marital_โสด`, `marital_แต่งงานแล้ว` เป็นต้น

### 4.3 One-Hot Encoding สำหรับ Supervised Features

ใน Supervised Learning Pipeline ระบบใช้ `OneHotEncoder` จาก sklearn ผ่านฟังก์ชัน `build_supervised_preprocessor` ใน `src/catfood_unsupervised/supervised/features.py:34-45`

```python
def build_supervised_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(feature_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
```

ตัวแปรที่ถูก Encode ในส่วนนี้คือ 16 Feature Columns ที่ประกอบด้วย ปัจจัยการตัดสินใจซื้อ 5 ข้อ บรรจุภัณฑ์ 1 ข้อ ความชอบบรรจุภัณฑ์ 8 ข้อ และข้อมูลประชากรศาสตร์ 3 ข้อ ตามที่กำหนดใน `schema.py:7-26`

### 4.4 Binary Encoding สำหรับ Packaging Effect

คอลัมน์ "บรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อหรือไม่" ถูกแปลงเป็น Binary (0/1) ด้วย `PACKAGING_EFFECT_MAPPING`

```python
PACKAGING_EFFECT_MAPPING = {
    "ไม่มีผล": 0,
    "มีผล": 1,
}
```

---

## บทที่ 5 Feature Engineering

### 5.1 Vote Features

Vote Features เป็นตัวแปรที่สร้างขึ้นจากข้อมูล Top-3 Rankings โดยคำนวณคะแนนน้ำหนักสะสม (Weighted Score) ให้กับแต่ละตัวเลือกบรรจุภัณฑ์ 10 ตัวเลือก ตามสูตร

- อันดับ 1 (Rank 1): น้ำหนัก = 3 คะแนน
- อันดับ 2 (Rank 2): น้ำหนัก = 2 คะแนน
- อันดับ 3 (Rank 3): น้ำหนัก = 1 คะแนน

สำหรับแต่ละแถว ถ้าผู้ตอบเลือก Option 3 เป็นอันดับ 1 คะแนน 3 จะถูกบวกเข้ากับ `vote_03` ฟังก์ชัน `build_vote_features` ใน `features.py:14-41` รัน Validation เพื่อให้แน่ใจว่าแต่ละแถวมีตัวเลือกที่ไม่ซ้ำกันใน 3 อันดับ

```python
def build_vote_features(
    df: pd.DataFrame,
    rank_columns: Sequence[str],
    option_count: int,
    weights: Sequence[int] = (3, 2, 1),
    prefix: str = "vote_",
) -> pd.DataFrame:
    rank_column_names = list(rank_columns)
    if len(rank_column_names) != len(weights):
        raise ValueError("rank_columns and weights must have the same length.")

    parsed_rankings = pd.DataFrame(index=df.index)
    for rank_column in rank_column_names:
        parsed_rankings[rank_column] = df[rank_column].map(
            lambda value: _parse_option_number(value, rank_column=rank_column)
        )

    _validate_rankings(parsed_rankings, option_count=option_count)

    feature_columns = [f"{prefix}{option:02d}" for option in range(1, option_count + 1)]
    vote_features = pd.DataFrame(0, index=df.index, columns=feature_columns, dtype=int)

    for rank_column, weight in zip(rank_column_names, weights, strict=True):
        ranked_options = parsed_rankings[rank_column]
        for option in range(1, option_count + 1):
            vote_features.loc[ranked_options.eq(option), f"{prefix}{option:02d}"] += weight

    return vote_features
```

**ตัวอย่างผลลัพธ์:**

| คอลัมน์ | คำอธิบาย |
|---|---|
| vote_01 | คะแนนน้ำหนักรวมสำหรับ Option 1 |
| vote_02 | คะแนนน้ำหนักรวมสำหรับ Option 2 |
| ... | ... |
| vote_10 | คะแนนน้ำหนักรวมสำหรับ Option 10 |

### 5.2 Ipsatized Option Ratings

Ipsatization เป็นเทคนิคที่ปรับค่าให้เป็นอิสระต่อ Response Style ของผู้ตอบแต่ละคน โดยนำค่าเฉลี่ยต่อแถวไปลบออกจากแต่ละค่า ทำให้ผู้ที่ตอบ "มากที่สุด" ทุกข้อจะไม่ได้คะแนนสูงกว่าผู้ที่ตอบ "ปานกลาง" ทุกข้อ

สูตรการคำนวณ Ipsatization:

```
ipsatized_value = original_value - row_mean
```

ฟังก์ชัน `ipsatize_rows` ใน `features.py:44-58` จัดการกับค่าที่หายไปโดยไม่นับรวมในการคำนวณค่าเฉลี่ย

```python
def ipsatize_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("ipsatize_rows expects a 2D array-like input.")

    observed_mask = ~np.isnan(array)
    counts = observed_mask.sum(axis=1, keepdims=True)
    sums = np.where(observed_mask, array, 0.0).sum(axis=1, keepdims=True)
    means = np.divide(
        sums,
        counts,
        out=np.full((array.shape[0], 1), np.nan, dtype=float),
        where=counts > 0,
    )
    return array - means
```

ผลลัพธ์จาก Ipsatization จะถูกนำไปทำ StandardScaler แล้วจึงทำ PCA

### 5.3 Principal Component Scores (PCA)

ระบบใช้ PCA (Principal Component Analysis) เพื่อลดมิติของ Ipsatized Option Ratings จาก 50 คอลัมน์เหลือจำนวนที่เลือกตามเกณฑ์ Cumulative Explained Variance >= 80% โดยใช้ฟังก์ชัน `run_pca` จาก `models.py` และ `_run_pca_pipeline` ใน `pipeline.py:281-329`

ขั้นตอนการทำ PCA:

1. **Standardization**: ปรับขนาด Ipsatized Ratings ด้วย `StandardScaler`
2. **Exploratory PCA**: ทดลองหาจำนวน Components ที่เหมาะสม (สูงสุด 8 components)
3. **Component Selection**: เลือกจำนวน Components ที่ทำให้ Cumulative Explained Variance >= 80%
4. **Final PCA**: สร้าง Final PCA Model ด้วยจำนวน Components ที่เลือก

### 5.4 Anomaly Detection Features

ระบบใช้ Isolation Forest เพื่อตรวจจับ Outliers ในข้อมูล ผลลัพธ์ประกอบด้วย

- `anomaly_flag`: ค่า 0 = ปกติ, 1 = Anomaly
- `anomaly_score`: คะแนนความผิดปกติ (ยิ่งสูงยิ่งผิดปกติมาก)

Anomaly Features ที่ใช้ในการตรวจจับประกอบด้วย Vote Features, Buy Factors (หลัง Impute), Packaging Effect, Packaging Importance, Ipsatized Option Ratings และ Demographics

---

## บทที่ 6 Feature Scaling

### 6.1 StandardScaler สำหรับ Ipsatized Option Ratings

ก่อนนำ Ipsatized Option Ratings ไปทำ PCA ระบบจะปรับขนาดด้วย `StandardScaler` เพื่อให้ทุก Features อยู่ในมาตราส่วนเดียวกัน ป้องกันไม่ให้ Feature ใดมีอิทธิพลมากเกินไปเพียงเพราะมีค่าสูงกว่า ตามฟังก์ชัน `_run_pca_pipeline` ใน `pipeline.py:288-292`

```python
standardized = pd.DataFrame(
    StandardScaler().fit_transform(ipsatized_option_ratings),
    index=ipsatized_option_ratings.index,
    columns=ipsatized_option_ratings.columns,
)
```

### 6.2 ไม่มีการ Scaling สำหรับ Demographics

ข้อมูล Demographics ที่ผ่าน One-Hot Encoding แล้ว อยู่ในรูปแบบ Binary (0/1) ซึ่งไม่จำเป็นต้องปรับขนาดเนื่องจากอยู่ในช่วง [0, 1] อยู่แล้ว

### 6.3 ไม่มีการ Scaling สำหรับ Vote Features

Vote Features อยู่ในรูปของคะแนนน้ำหนักรวม ไม่มีการปรับขนาดด้วย Scaler ใดๆ เนื่องจากถูกใช้โดยตรงใน Anomaly Detection และสำหรับการสร้าง Segment Profiles

---

## บทที่ 7 ผลลัพธ์หลังการเตรียมข้อมูล

### 7.1 สถานะของ Processed Data

ไฟล์ `outputs/clean_dataset_with_segments.csv` เป็นผลลัพธ์หลังการเตรียมข้อมูลทั้งหมด ประกอบด้วย 102 แถว (จากข้อมูลดิบจำนวนหนึ่งที่ถูกกรองออก) และมีโครงสร้างคอลัมน์ดังนี้

| กลุ่มคอลัมน์ | จำนวนคอลัมน์ | สถานะ |
|---|---|---|
| ข้อมูลดิบจากแบบสำรวจ | 76 | สมบูรณ์ |
| Top-3 Rankings (rank1, rank2, rank3) | 3 | แปลงจากข้อความเป็นตัวเลขแล้ว |
| Vote Features (vote_01 ถึง vote_10) | 10 | สร้างจาก Top-3 Rankings แล้ว |
| PCA Scores (PC1 ถึง PC8) | 8 | ปรับขนาดด้วย StandardScaler แล้ว |
| Segment Labels | 1 | ผลจาก KMeans Clustering |
| Anomaly Detection | 2 | ผลจาก Isolation Forest |

### 7.2 ตัวอย่างข้อมูลที่ประมวลผลแล้ว

```
Timestamp,อายุ,เพศ,สถานภาพสมรส,top3_rank_1,top3_rank_2,top3_rank_3,
 vote_01,vote_02,...,vote_10,PC1,PC2,...,PC8,segment,anomaly_flag,anomaly_score
6/26/2024 14:15:25,30-39ปี,ชาย,มีแฟนแต่ยังไม่แต่งงาน,3,6,8,
 0,0,3,0,0,2,0,1,0,0,-0.2819,1.0319,...,2,0,0.3994
```

### 7.3 การจัดการข้อมูลที่หายไปในผลลัพธ์

- Buy Factors: ค่าที่หายไปถูก Impute ด้วย KNNImputer (k=5) แล้ว
- Ipsatized Option Ratings: ค่าที่หายไปถูกเติมด้วย 0.0 แล้ว
- Supervised Features: ค่าที่หายไปถูกเติมด้วย `"__missing__"` ก่อนเข้า OneHotEncoder
- Demographic: ไม่มีค่าที่หายไปเนื่องจากใช้ get_dummies

### 7.4 Segment Labels

ข้อมูลที่ประมวลผลแล้วมีการสร้าง Segment Labels ผ่าน KMeans Clustering โดยใช้ PCA Scores เป็น Input จำนวน Cluster (k) ถูกเลือกอัตโนมัติจากการประเมิน Silhouette Score และ Davies-Bouldin Score

---

## บทที่ 8 สรุป

กระบวนการ Data Preprocessing ในโครงการ AI-Driven Marketing Campaign System ประกอบด้วยขั้นตอนที่ครบถ้วนตั้งแต่การนำเข้าข้อมูลดิบจากแบบสำรวจออนไลน์ การกรองคำตอบที่ไม่สมบูรณ์ การจัดการค่าที่หายไปด้วย KNN Imputation การแปลง Likert Scale จากข้อความเป็นตัวเลข การสร้าง Vote Features จาก Top-3 Rankings การ Ipsatize Option Ratings การลดมิติด้วย PCA การสร้าง Segment Labels ด้วย KMeans และการตรวจจับ Anomaly ด้วย Isolation Forest

การเตรียมข้อมูลที่เหมาะสมเป็นปัจจัยสำคัญที่ทำให้ผลลัพธ์จาก Unsupervised Learning (Clustering, Anomaly Detection) และ Supervised Learning (Classification) มีความถูกต้องและน่าเชื่อถือ ระบบนี้ใช้ Best Practices ในการเตรียมข้อมูล เช่น การใช้ KNN Imputation แทนการลบแถวทิ้ง การใช้ StandardScaler ก่อน PCA การใช้ Ipsatization เพื่อลด Response Bias และการใช้ OneHotEncoder สำหรับ Categorical Features ใน Supervised Pipeline ซึ่งทำให้ข้อมูลพร้อมสำหรับการนำไปวิเคราะห์ในขั้นตอนถัดไป

---

*รายงานฉบับนี้จัดทำจากการวิเคราะห์โค้ดใน repository ณ วันที่ 11 พฤษภาคม 2569*

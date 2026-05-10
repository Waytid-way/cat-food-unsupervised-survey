# Enhanced Dashboard Design Spec

## Goal
อัปเกรด Dash dashboard ให้แสดงข้อมูลครบถ้วนขึ้น ด้วย structured redesign ที่มี hierarchy ชัดเจน โดยเพิ่ม 4 new visualizations:

1. **Segment Profiles** — demographic + buy factor breakdown แยกตาม segment
2. **K-Means Evaluation Table** — table แสดง silhouette/DB/inertia ของทุก k
3. **Option Rating Distribution** — heatmap/radar แสดง rating ของทั้ง 10 options ตาม 5 attributes
4. **Top-3 Ranking Breakdown** — % แยก rank-1, rank-2, rank-3 ของแต่ละ option

## Architecture

### File Structure
```
src/catfood_unsupervised/dashboard/
├── app.py                      # Main Dash app (keep)
├── config.py                   # KPI configs, PALETTE, TAB_ITEMS (keep)
├── data_loader.py              # Data loading layer (keep)
├── components/
│   ├── kpi_banner.py           # KPI cards (keep)
│   ├── tab_eda.py              # Modify: เพิ่ม Top-3 Breakdown + Option Rating
│   ├── tab_correlation.py      # Keep as-is
│   ├── tab_clustering.py       # Modify: เพิ่ม K-Means Evaluation Table
│   ├── tab_persona.py          # Modify: เพิ่ม Segment Profiles breakdown
│   └── shared.py               # Create: shared chart helpers, color utilities
└── styles/
    ├── custom.css              # Create: custom CSS สำหรับ enhanced styling
    └── theme.py                # Create: theme constants (dark/light mode support)
```

### New Components

| Component | Responsibility |
|-----------|---------------|
| `shared.py` | `render_summary_stats()` — response count + completion rate. `segment_color_map()` — helper for consistent segment colors |
| `styles/custom.css` | Enhanced card styling, better chart backgrounds, typography improvements |
| `styles/theme.py` | `DARK_THEME`, `LIGHT_THEME` constants — toggle-ready |

### Tab Changes

**Tab 1: EDA**
- เพิ่ม: Top-3 Ranking Breakdown (stacked bar chart แสดง rank-1, rank-2, rank-3 %)
- เพิ่ม: Option Rating Distribution (radar chart หรือ heatmap แสดง rating profile ของ 10 options)
- Refactor: แบ่ง sections ชัดเจน — Demographics | Buy Factors | Voting | Ratings

**Tab 3: Clustering**
- เพิ่ม: K-Means Evaluation Table (DataTable แสดง k, silhouette, DB, inertia)
- Refactor: แบ่ง sections — PCA Results | Clustering Metrics | Validation

**Tab 4: Persona**
- เพิ่ม: Segment Profile Cards — demographic breakdown (gender %, age %, marital %) + buy factor preferences แยกตาม segment
- Refactor: แบ่ง sections — Scatter Plot | Segment Overview | Segment Deep-Dive

### Data Requirements

```python
# New metrics/tables needed (จาก pipeline output ที่มีอยู่แล้ว):
# - clean_df: top3_rank_1, top3_rank_2, top3_rank_3, segment, buy_factor_*, demographics
# - segment_profiles: mean ของทุก feature แยกตาม segment
# - metrics: kmeans_evaluation (list of dicts มี k, silhouette_score, davies_bouldin_score, inertia)
# - metrics: row_counts (raw_loaded, completed_top3)
```

## New Visualizations Detail

### 1. Top-3 Ranking Breakdown (Tab 1)
- Stacked horizontal bar chart
- 10 options (y-axis) × 3 segments (stacked: rank-1=3pts, rank-2=2pts, rank-3=1pt normalized)
- Colors: segment palette (segment1, segment2, anomaly)

### 2. Option Rating Distribution (Tab 1)
- Radar chart หรือ heatmap
- 10 options × 5 attributes (agreement scale 1-5)
- แสดง mean rating per option

### 3. K-Means Evaluation Table (Tab 3)
- Dash DataTable ใต้ elbow/silhouette/DB charts
- Columns: k, Silhouette Score, Davies-Bouldin Index, Inertia
- Sortable by any column, best value highlighted

### 4. Segment Profile Deep-Dive (Tab 4)
- แทนที่ persona cards ที่มี top values เป็น full breakdown
- 2-column layout (Segment 1 | Segment 2)
- แถว: gender stacked bar, age stacked bar, marital stacked bar, buy factor bars
- แต่ละ segment แสดง % distribution ไม่ใช่แค่ top value

## Style Guide

- **Color Palette:** ใช้ PALETTE ที่มีอยู่แล้วใน `config.py`
- **Card Style:** white background, subtle shadow, 8px border-radius, border-left 4px accent color per section
- **Typography:** Font stack: Inter, -apple-system, sans-serif
- **Charts:** white plot background, grid lines subtle (#eee), better axis labels
- **Responsive:** Bootstrap grid — 1 col mobile, 2 cols tablet, maintain desktop

## Scope (YAGNI)
- NOT building dark/light mode toggle yet — just set up theme constants for future
- NOT building real-time data refresh — data reloads on tab switch (existing behavior)
- NOT building export functionality — focus on visualization first

## Success Criteria
- [ ] Tab 1: Top-3 breakdown + Option Rating distribution แสดงผลถูกต้อง
- [ ] Tab 3: K-Means table แสดงทุก k พร้อม sort/click ที่ chart
- [ ] Tab 4: Segment profile แสดง % breakdown ไม่ใช่แค่ top value
- [ ] Summary stats (response count, completion rate) แสดงใน KPI banner หรือ tab header
- [ ] CSS enhancements ทำให้ dashboard ดู professional ขึ้นโดยไม่ breaking existing functionality
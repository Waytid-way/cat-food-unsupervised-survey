# Business Insight Compact Boardroom Snapshot Design Spec

## Goal
Redesign the `/business` page into a compact executive summary. The page should answer the business question quickly:

> What is the segment story, what matters most, and what should we do next?

The current page is useful but still reads like a technical dashboard. It shows persona cards and a wide chart, but it does not immediately surface a leadership-friendly headline or a short list of actions. This redesign will keep the data-driven foundation and make the page easier to scan in one pass.

## Design Principles
- Lead with the conclusion, not the chart.
- Use compact cards and short labels instead of raw feature dumps.
- Keep the page responsive and readable on desktop and mobile.
- Reuse the existing dashboard shell, colors, and card language so the redesign feels native.

## Proposed Layout

### 1. Hero Summary Strip
At the top of the page, show a short executive headline and a one-sentence interpretation of the segment data. This is the first thing a manager should read.

The hero block should include:
- A page eyebrow such as `Business Insight`
- A headline derived from the data, for example:
  - `Segment 2 dominates the market story, while Segment 1 holds the premium signal`
- A short supporting sentence with the main takeaway

### 2. KPI Row
Directly below the hero, show 3 to 4 small KPI cards. These should summarize the most important facts without requiring the chart:
- Total completed responses
- Largest segment share and count
- Final cluster count
- Overall anomaly rate or strongest packaging signal

The KPI row should be visually compact and should fit in one line on desktop, then stack on smaller screens.

### 3. Main Content Grid
Use a two-column layout for the body:

#### Left Column: Segment Snapshot Cards
Render one compact card per segment from `segment_profiles`.

Each card should show:
- Segment ID
- Segment size and share
- Top voted option
- Top buy factor
- Top packaging importance
- One short human-readable summary sentence

The labels for buy factors and packaging importance should use the existing label lists from `catfood_unsupervised.unsupervised.reporting` so the page shows readable business language instead of raw field names.

#### Right Column: Segment Mix + Recommendations
Use the right column for the most executive-friendly visual and the action summary:
- One compact `Segment Size Distribution` chart
  - Prefer a donut or compact horizontal bar chart
  - The chart should answer "how big is each segment?"
- One `What to do next` card with 2 to 4 bullets
  - These bullets should be derived from the strongest patterns in the data
  - The copy should be short enough for a boardroom view

### 4. Optional Footer
If space permits, add a subtle footer line with:
- Data source note
- Refresh context
- Link back to the unsupervised page for deeper analysis

## Content Strategy

### What stays
- The page still uses the same pipeline outputs from `outputs/unsupervised`
- The page still reads `metrics_summary.json`, `segment_profiles.csv`, and `clean_dataset_with_segments.csv`
- The page still shows segment-level insight, not raw per-response data

### What changes
- Replace the current `Persona Cards` section as the primary story
- Remove the large, dense grouped bar chart from the main view
- Replace raw labels such as `Factor 1` or `Pkg 3` with human-readable labels
- Convert the page from a technical report view into a boardroom snapshot

## Data Flow
1. `load_all_data(UNSUPERVISED_OUTPUT_DIR)` loads metrics, cleaned data, correlation matrix, and segment profiles.
2. The page derives a lightweight view model:
   - Headline
   - KPI values
   - Segment card copy
   - Segment mix figure
   - Recommendation bullets
3. The layout renders those sections with existing Dash and Bootstrap components.

No pipeline schema changes are required for this redesign.

## Error Handling
- If `outputs/unsupervised` is missing, show a friendly empty state that tells the user to run the unsupervised pipeline first.
- If one derived field is missing, skip only that card or bullet instead of failing the whole page.
- Never render the raw exception to the user.

## Styling Notes
- Keep the page compact with tighter vertical spacing than the current `Business Insight` view.
- Use a neutral background and subtle card shadows so the page feels like a presentation board rather than an analytics table.
- Use segment colors consistently across the cards and the chart.
- Make the headline large and the supporting text small.
- Ensure the layout collapses cleanly on mobile into a single column.

## Files To Update
- `src/catfood_unsupervised/dashboard/pages/insight.py`
- `src/catfood_unsupervised/dashboard/styles/custom.css`
- `tests/dashboard/test_insight_page.py`

## Testing Plan
- Verify the page renders with unsupervised outputs present.
- Verify the page shows the compact executive layout and does not rely on the old `Persona Cards` heading.
- Verify the page still shows the empty state when the output directory is missing.
- Verify the new labels are human-readable and no raw feature names leak into the UI.

## Acceptance Criteria
- The first screen clearly communicates the business story.
- The page fits the `Compact Boardroom Snapshot` direction chosen by the user.
- The layout is responsive and does not feel crowded.
- The page uses existing data only, with no pipeline changes.
- The page is easier to present to a non-technical audience than the current layout.

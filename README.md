# Linear Regression Lab

**Live demo:** https://shishiv30.github.io/regression/

To try it on the published page, pick **San Jose sold listings (training)** from **Load test dataset…**, set `zipCode` to **Enum (x)**, and press **Train** (`closePrice` is the outcome). Then pick **San Jose for-sale listings (verify)** under **Predict many records** to see the gap chart. The source files are [`public/test-data/test-san-jose-training-data.csv`](public/test-data/test-san-jose-training-data.csv) (4,711 sold listings, each with a `closePrice`) and [`public/test-data/test-san-jose-verify-data.csv`](public/test-data/test-san-jose-verify-data.csv) (1,655 for-sale listings, 1,647 with a `listPrice`). Both have `lat`/`lng`, so the location adjustment, comparables and the gap map are on. Edit them in place, then run `npm run test-data` to refresh the row counts.

Every push to `main` runs the tests, builds the site and publishes it to the `gh-pages` branch (`.github/workflows/deploy.yml`).

Client-side linear regression on your own CSV records. Nothing leaves the browser.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # static site in dist/
npm test         # calculation-layer tests
npm run test-data  # regenerate public/test-data/
```

To check a build the way GitHub Pages serves it (under `/regression/`):

```
BASE_PATH=/regression/ npm run build && BASE_PATH=/regression/ npm run preview
```

## Structure

- `src/calc/`: the calculation layer (no DOM)
  - `csv.js` parses the CSV (PapaParse)
  - `schema.js` detects number vs. enum, finds numeric min–max ranges, builds deduplicated enum dictionaries
  - `model.js` encodes the data (min-max scaling, one-hot against a baseline) and handles weights in original units, prediction and metrics
  - `trainer.js` runs gradient descent with TensorFlow.js (CPU backend) for N steps
  - `confidence.js` turns left-out errors into likely ranges and confidence levels
  - `outliers.js` finds records the formula misses far more than the rest, so training can leave them out
  - `location.js` adds the location adjustment from nearby training records (latitude/longitude)
  - `batch.js` estimates every row of a second CSV and measures the gap to its actual value
  - `storage.js` saves the dataset, field setup, model and batch file in localStorage
- `src/ui/`: the presentation layer
  - top bar: CSV import, field cards (one dropdown each: Number (x), Enum (x), Outcome (y), Latitude, Longitude, Label or Ignore), training controls (`fields.js`)
  - left panel: training report, loss curve, weights (`report.js`)
  - main area, three tabs: **Weights** (`graph.js`, each input points to the outcome), **Gap chart** (`gapChart.js`) and **Data table** (`dataTable.js`, every predicted record with `est_<outcome>` and the actual value in the first two columns, then the rest of the fields, gap and gap %; click a header to sort)
  - right panel: predict one record (`report.js`) and predict many records from a CSV (`batch.js`)

### Field dropdown

Each field card has one dropdown that sets both what the field is for and how it is read:

| Choice | Trained on | Meaning |
| --- | --- | --- |
| **Number (x)** | yes | numeric input, one weight |
| **Enum (x)** | yes | categorical input, one weight per value (e.g. `zipCode`) |
| **Outcome (y)** | — | the value to estimate (numeric) |
| **Latitude** / **Longitude** | location adjustment | coordinates for the location adjustment, comparables and the gap map |
| **Label** | no | names a record: shown in the gap chart tooltip when the batch file has the same column |
| **Ignore** | no | not used |

Starting choices come from the values only, never from column names, so they work for any subject:

- a pair of numeric columns with 4+ decimal places, almost no whole numbers, and values within ±90 / ±180 becomes **Latitude** / **Longitude**;
- the last other numeric column becomes the **Outcome**;
- a row counter (whole numbers, all different, no gaps, like 1…N) and text with more than 30 different values (ids, names, dates) become **Label**;
- other numbers and enums with 2–30 values become inputs.

### Missing values

**Missing values** (next to Train) decides what happens to a record with a blank input:

- **Drop record** (default): the record is left out of training.
- **Fill with median**: the blank takes the field's training median (numbers) or most common value (enums), so the record is kept. A single prediction or batch row with a blank takes the same value, with a note.

A record without an outcome is always dropped. The report shows how many records had blanks filled.

### How weights and estimates are read

- A number field's weight is in original units: the outcome change per 1 unit of that field.
- An enum field gets one weight per value, relative to its most common value (the baseline, weight 0).
- A single prediction is broken down from an **average record**: each field shows how its value moves the result away from that average.
- In batch prediction, columns the formula does not use are ignored. A formula field with no matching column or a blank cell uses the training average (mean for numbers, baseline for enums), and a ⚠ after the estimate names the missing fields on hover.

## Group by: one formula per enum value

Tick **Group by** on an enum field card (e.g. `listingType.name`) to train a separate formula for each of its values, each using only that value's records.

- Each group gets its own ranges, averages and enum dictionaries. The group-by field itself is not an input.
- Inputs with nothing to learn in a group are left out of that group's formula: blank in more than half of the group's records (e.g. bedrooms for LAND), empty, or the same value everywhere. The group's card lists them with the reason, e.g. "bedrooms (92% blank in this group)".
- Groups with fewer than 10 complete records are skipped and listed with the reason.
- ⚠ marks a formula with fewer than 2 records per weight. It can memorise its training records, so its R² looks better than it really is. Enum inputs with many values (like `zipCode`) add one weight per value.
- The **Weights** tab shows one weight graph per group, each drawn from that group's own formula (`groupGraphs.js`). Click a graph's header, a row in the left panel's group table, or **Showing formula for** to select a group; the left panel and the predict form follow the selected group.
- A single prediction uses only the formula of the chosen group, and its breakdown is drawn on that group's graph.
- Single and batch predictions use the formula of the record's group. In batch prediction, map the group column under **Column matching**. Rows whose group has no formula get no estimate and are marked **no formula**.
- The **Gap chart**, **Data table** and the batch summary show only the rows of the selected group, so each group's estimates can be checked on their own. Select another group to see its rows; **Download results CSV** always has every row.

## Outliers: leave bad records out of training

A few bad records (a price with an extra zero, a size typed in the wrong unit) can drag every weight towards themselves. **Outliers** in the training bar (default **Drop > 3.5σ**) leaves them out before the formula is trained:

1. A quick exact least-squares fit on the same inputs gives every record an error (actual − fit).
2. A record's error is measured in robust standard deviations: its distance from the median error, divided by 1.4826 × the median absolute deviation. A few wild records barely move a median, so they can't hide themselves the way they would with a mean and standard deviation.
3. Records past the cutoff are left out and the fit is redone, until the set stops changing (at most 20% of the records are left out).

Errors, not raw values, are compared. A 9,000 sqft house with a high price fits the formula and is kept; a 2,500 sqft house at a mansion price does not fit and is left out.
- **Group by**: each group is screened on its own records, against its own formula.
- **Location**: a record's error is compared with the median error of its nearest neighbours first, so a pricey house in a pricey area is not flagged.
- The report lists every record left out (row, group, actual value, the screening fit, and how many σ off). Outliers are also kept out of the location adjustment's stored residuals.

On an earlier set of 500 Austin for-sale listings (no longer in the repo; 5-fold, every held-out record counted, outliers included), dropping > 3.5σ cut the median error from 40% to 23%.

## Log scale

**Scale** in the training bar (default **Auto**) picks what the formula estimates:
- **Log** estimates log(outcome), so each weight is a percentage change: "+6% per bathroom" instead of "+$40k per bathroom". Prices grow by percentages (a bathroom adds more to a $2M house than to a $300k one), so this fits them better, and the outlier screen compares percentage errors, so pricey homes are no longer flagged for having large dollar errors.
- **Auto** uses log for an outcome that is always above 0 and spans at least 10×; **Linear** keeps the outcome as it is.
- On a log scale the weights table and graph show effects as percentages, the equation reads `log(outcome) = …`, and the prediction breakdown still adds up in outcome units: each part is taken in order, with its own percentage alongside.

## Comparables, confidence and likely range

With a location, **Comparables by** (default **Auto**) adds a second estimate: the median value per size unit (e.g. price per sqft) of the nearest training records of similar size, times the record's size. Auto picks the number input whose log tracks the outcome best (e.g. sqft for houses and condos, lot size for land).
- The estimate blends formula + location and comparables 50/50 (geometric mean). The breakdown shows the blend as a **comparables** row, and a single prediction lists the comparables with their distance, size, value and value per size.
- **Confidence** (High / Medium / Low) comes from two signals known before the real value: how far the formula and the comparables disagree, and how much the comparables' value per size varies. Training records are split into thirds by the combined score.
- **Likely range**: every estimate gets the range that held 8 in 10 actual values. It comes from training records the estimate had not seen: each record's formula estimate is fitted without it (5-fold, exact least squares), its location and comparables leave it out, and outliers are included. With comparables, each confidence level has its own range. Without a location, the range comes from the formula's held-out errors.
- The report's **How far off estimates are** table shows, per level, the record count, the typical miss and the 80% range. Batch prediction shows the share of actual values inside their range (honest ranges hold about 80%) and the median gap per level. The data table and results CSV add `confidence`, `est_low` and `est_high` (and `comparables_adj`).
- The left-out R² shown with the location is lower than the training R²: it is measured on records the formula did not see. In grouped models, thin formulas (⚠) are left out of the combined left-out figures.

On the same earlier Austin listings (5-fold, every held-out record counted):

| | Median error | Within ±10% | In 80% range |
|---|---|---|---|
| Linear, outliers, location | 22.7% | 24% | 79% |
| + log scale | 19.1% | 27% | 81% |
| + comparables | 18.4% | 28% | 79% |
| &nbsp;&nbsp;High confidence | 15.1% | 31% | 79% |
| &nbsp;&nbsp;Medium | 19.7% | 29% | 78% |
| &nbsp;&nbsp;Low | 22.1% | 23% | 78% |

## Location: adjust by nearby records

A linear formula can't express "this neighbourhood is pricier". Mark a field **Latitude** and another **Longitude** on their field cards to add a location adjustment. A pair of columns that look like coordinates by their values (4+ decimal places, almost no whole numbers, within ±90 / ±180) is picked automatically, whatever the columns are called. The coordinates are not formula inputs and get no weight.

1. Training fits the formula as usual. It then stores every training record's coordinates and **residual**: the actual value minus the formula's estimate.
2. For a new record, the **K nearest** training records (**Neighbours**, default 10) give a distance-weighted average of their residuals. That average is added to the formula's estimate:
   - A Gaussian kernel weights the neighbours by distance. Its distance scale (0.25 to 32 km) is picked in training: the one that best predicts each record's residual from its neighbours, with the record itself left out.
   - The average is shrunk towards 0 as if one more neighbour with residual 0 sat on the spot, so records far from any training data get almost no adjustment.
3. The report shows R² and RMSE for the formula alone and for formula + location, from that same left-out check. The weight graph gets a purple **location** node. A single prediction lists its nearest training records with their distance, value and how much the formula missed on each.
4. **Group by**: each group's adjustment uses only that group's records.
5. **Missing coordinates**: a record without usable lat/lng (blank, unparsable or out of range) gets the formula's estimate with no adjustment, and the estimate gets a ⚠. In batch prediction, map the coordinate columns under **Column matching**; the results CSV adds a `location_adj` column.

Try it with **Neighbourhood prices (lat/lng, training)**, then **Neighbourhood prices (lat/lng, verify)** as the batch file.

## Check estimates against real outcomes

1. Train on one file, e.g. **San Jose sold listings (training)** with `closePrice` as the outcome.
2. Under **Predict many records** (right panel), import a second CSV, e.g. **San Jose for-sale listings (verify)**.
3. Check **Column matching**:
   - Inputs read the same-named column. Inputs with no matching column (e.g. `soldXdaysago`) and cells that are blank use the training average; those estimates get a ⚠ (hover for which fields). Columns the formula does not use (e.g. `listXdaysago`) are ignored.
   - **Actual value** (e.g. `listPrice`) is what each estimate is compared with. It defaults to the outcome's column, else the unused numeric column whose values fall inside the training outcome range most often.
   - **Row label** (e.g. `mlsNumber`) names each row in the chart and table. It defaults to a field marked **Label** in training, else a text column with a different value on nearly every row.
4. The **Gap chart** tab draws every record as a node. Blue means the actual value is above the estimate, orange-red means below, and a bigger node means a bigger |gap %| (capped at 50%). A hollow ring has no actual value. It has three layouts:
   - **Map** (when the batch file has latitude/longitude): each record sits at its coordinates. Records with the same **Link by** value (defaults to the enum input with the most values that still average 2+ training records each, e.g. `zipCode` set to Enum, else none) are linked to a ◆ hub at their centre. Hover a hub for that group's median gap.
   - **Estimate vs actual**: actual across, estimate up, log scale when values span 10× or more. On the dashed diagonal the estimate is exact; each node's stem is its gap. This is the default without coordinates.
   - **List**: one row per record, sorted by gap %.

   Scroll to zoom, drag to pan and double-click to reset. Hovering a node shows its **Label** fields, the link value, the estimate with its confidence and likely range, and the actual value and gap; it also highlights the records linked with it. The right panel shows the summary (median gap, share within ±10%, average gap), the per-row table and **Download results CSV**, which adds `est_<outcome>`, `gap`, `gap_pct` and `imputed_fields` columns.

Gap = actual − estimate; gap % = gap ÷ actual. Rows with no actual value get an estimate but no gap.

## Test data collection

`npm run test-data` regenerates `public/test-data/`. Each dataset is built from a known formula plus noise, and the formula is recorded in `index.json` as `truth`. All of them can be loaded from the "Load test dataset…" menu and from the batch **Test file…** menu. The two San Jose files are real data kept in `public/test-data/` and only counted, so they have no `truth`. Train on the sold file (`closePrice`), then verify on the for-sale file (`listPrice`).

| File | What it checks |
|---|---|
| `salary.csv` | 3 enums, a free-text column, age that tracks experience, `"85,000"` outcomes, missing cells |
| `fuel-efficiency.csv` | 8 inputs with strongly correlated engine specs |
| `ad-spend.csv` | 30 rows only; one input has no real effect |
| `messy.csv` | CRLF line endings, padding, quoted commas, `N/A`, constant/empty columns, blank rows |
| `company-value.csv` | inputs ~0.005 to ~1M, outcome ~1e9 |
| `sensor-large.csv` | 120k rows, bigger than the localStorage quota |
| `location-train.csv` / `location-verify.csv` | price with a neighbourhood effect (two pricey spots, one cheap) from latitude/longitude; 10 verify rows lack coordinates |
| `test-san-jose-training-data.csv` | Real San Jose sold listings (training); outcome is `closePrice` |
| `test-san-jose-verify-data.csv` | Real San Jose for-sale listings (verify); compare estimates with `listPrice` |

Tests:
- `test/collection.test.js` trains on each synthetic dataset. It checks the result against the exact least-squares solution (within 1%) and against the generating formula (within 10%).
- `test/batch.test.js` checks batch prediction: column matching, training-average fill-in, gap maths, the results CSV, data-based picks for the actual and row-label columns, and the San Jose train-on-sold then verify-on-for-sale flow (every listing estimated, median gap under 25%).
- `test/calc.test.js` covers CSV parsing, type detection and training on the housing sample.
- `test/location.test.js` checks the location adjustment: role detection, the left-out gain, shifts near the pricey and cheap spots, fade-out far away, missing coordinates, batch and group-by.
- `test/confidence.test.js` checks the log scale (breakdown adds up, weights as percentages), comparables, confidence levels and that held-out actual values mostly fall in their range.
- `test/outliers.test.js` poisons a clean dataset with a few bad records and checks they are left out, the weights come back, and each group is screened on its own.
- `test/missing.test.js` checks dropping vs. filling blank inputs, and leaving mostly-blank fields out of a group's formula.
- `test/group.test.js` checks group-by training: one formula per group, dropped inputs, skipped groups, and routing in single and batch prediction.

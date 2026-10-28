# Linear Regression Lab

**Live demo:** https://shishiv30.github.io/regression/

To try it on the published page, pick **Austin for-sale listings (training)** from **Load test dataset…**, set `zipCode` to **Enum**, and press **Train** (`listPrice` is the outcome). Then pick **Austin sold listings (verify)** under **Predict many records** to see the gap chart. The source files are [`test/test-austin-training-data.csv`](test/test-austin-training-data.csv) (500 for-sale listings) and [`test/test-austin-verify-data.csv`](test/test-austin-verify-data.csv) (19 sold listings, 3 with a `closePrice`).

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
  - `outliers.js` finds records the formula misses far more than the rest, so training can leave them out
  - `location.js` adds the location adjustment from nearby training records (latitude/longitude)
  - `batch.js` estimates every row of a second CSV and measures the gap to its actual value
  - `storage.js` saves the dataset, field setup, model and batch file in localStorage
- `src/ui/`: the presentation layer
  - top bar: CSV import, field cards (role and type), training controls (`fields.js`)
  - left panel: training report, loss curve, weights (`report.js`)
  - main area, three tabs: **Weights** (`graph.js`, each input points to the outcome), **Gap chart** (`gapChart.js`) and **Data table** (`dataTable.js`, every predicted record with `est_<outcome>` and the actual value in the first two columns, then the rest of the fields, gap and gap %; click a header to sort)
  - right panel: predict one record (`report.js`) and predict many records from a CSV (`batch.js`)

### How weights and estimates are read

- A number field's weight is in original units: the outcome change per 1 unit of that field.
- An enum field gets one weight per value, relative to its most common value (the baseline, weight 0).
- A single prediction is broken down from an **average record**: each field shows how its value moves the result away from that average.
- In batch prediction, columns the formula does not use are ignored. A formula field with no matching column or a blank cell uses the training average (mean for numbers, baseline for enums), and a ⚠ after the estimate names the missing fields on hover.

## Group by: one formula per enum value

Tick **Group by** on an enum field card (e.g. `listingType.name`) to train a separate formula for each of its values, each using only that value's records.

- Each group gets its own ranges, averages and enum dictionaries. The group-by field itself is not an input.
- Inputs with nothing to learn in a group are left out of that group's formula. For example, LAND has no bedrooms or sqft.
- Groups with fewer than 10 complete records are skipped and listed with the reason.
- ⚠ marks a formula with fewer than 2 records per weight. It can memorise its training records, so its R² looks better than it really is. Enum inputs with many values (like `zipCode`) add one weight per value.
- The **Weights** tab shows one weight graph per group, each drawn from that group's own formula (`groupGraphs.js`). Click a graph's header, a row in the left panel's group table, or **Showing formula for** to select a group; the left panel and the predict form follow the selected group.
- A single prediction uses only the formula of the chosen group, and its breakdown is drawn on that group's graph.
- Single and batch predictions use the formula of the record's group. In batch prediction, map the group column under **Column matching**. Rows whose group has no formula get no estimate and are marked **no formula**.

## Outliers: leave bad records out of training

A few bad records (a price with an extra zero, a size typed in the wrong unit) can drag every weight towards themselves. **Outliers** in the training bar (default **Drop > 3.5σ**) leaves them out before the formula is trained:

1. A quick exact least-squares fit on the same inputs gives every record an error (actual − fit).
2. A record's error is measured in robust standard deviations: its distance from the median error, divided by 1.4826 × the median absolute deviation. A few wild records barely move a median, so they can't hide themselves the way they would with a mean and standard deviation.
3. Records past the cutoff are left out and the fit is redone, until the set stops changing (at most 20% of the records are left out).

Errors, not raw values, are compared. A 9,000 sqft house with a high price fits the formula and is kept; a 2,500 sqft house at a mansion price does not fit and is left out.
- **Group by**: each group is screened on its own records, against its own formula.
- **Location**: a record's error is compared with the median error of its nearest neighbours first, so a pricey house in a pricey area is not flagged.
- The report lists every record left out (row, group, actual value, the screening fit, and how many σ off). Outliers are also kept out of the location adjustment's stored residuals.

On the Austin listings (5-fold, every held-out record counted, outliers included), dropping > 3.5σ cut the median error from 40% to 23%.

## Location: adjust by nearby records

A linear formula can't express "this neighbourhood is pricier". Mark a field **Latitude** and another **Longitude** on their field cards to add a location adjustment. Columns named `lat`/`latitude` and `lng`/`lon`/`longitude` are picked automatically. The coordinates are not formula inputs and get no weight.

1. Training fits the formula as usual. It then stores every training record's coordinates and **residual**: the actual value minus the formula's estimate.
2. For a new record, the **K nearest** training records (**Neighbours**, default 10) give a distance-weighted average of their residuals. That average is added to the formula's estimate:
   - A Gaussian kernel weights the neighbours by distance. Its distance scale (0.25 to 32 km) is picked in training: the one that best predicts each record's residual from its neighbours, with the record itself left out.
   - The average is shrunk towards 0 as if one more neighbour with residual 0 sat on the spot, so records far from any training data get almost no adjustment.
3. The report shows R² and RMSE for the formula alone and for formula + location, from that same left-out check. The weight graph gets a purple **location** node. A single prediction lists its nearest training records with their distance, value and how much the formula missed on each.
4. **Group by**: each group's adjustment uses only that group's records.
5. **Missing coordinates**: a record without usable lat/lng (blank, unparsable or out of range) gets the formula's estimate with no adjustment, and the estimate gets a ⚠. In batch prediction, map the coordinate columns under **Column matching**; the results CSV adds a `location_adj` column.

Try it with **Neighbourhood prices (lat/lng, training)**, then **Neighbourhood prices (lat/lng, verify)** as the batch file.

## Check estimates against real outcomes

1. Train on one file, e.g. **Austin for-sale listings (training)** with `listPrice` as the outcome.
2. Under **Predict many records** (right panel), import a second CSV, e.g. **Austin sold listings (verify)**.
3. Check **Column matching**:
   - Inputs read the same-named column. Inputs with no matching column (e.g. `listXdaysago`) and cells that are blank use the training average; those estimates get a ⚠ (hover for which fields). Columns the formula does not use (e.g. `soldXdaysago`) are ignored.
   - **Actual value** (e.g. `closePrice`) is what each estimate is compared with.
   - **Row label** (e.g. `mlsNumber`) names each row in the chart and table.
4. The **Gap chart** tab shows each record's estimate (hollow dot) and actual value (solid dot), sorted by gap %. Blue means the actual value is above the estimate; orange-red means below. The right panel shows the summary (median gap, share within ±10%, average gap), the per-row table and **Download results CSV**, which adds `est_<outcome>`, `gap`, `gap_pct` and `imputed_fields` columns.

Gap = actual − estimate; gap % = gap ÷ actual. Rows with no actual value get an estimate but no gap.

## Test data collection

`npm run test-data` regenerates `public/test-data/`. Each dataset is built from a known formula plus noise, and the formula is recorded in `index.json` as `truth`. All of them can be loaded from the "Load test dataset…" menu and from the batch **Test file…** menu. The two Austin files are copied in from `test/`; they are real data, so they have no `truth`.

| File | What it checks |
|---|---|
| `salary.csv` | 3 enums, a free-text column, age that tracks experience, `"85,000"` outcomes, missing cells |
| `fuel-efficiency.csv` | 8 inputs with strongly correlated engine specs |
| `ad-spend.csv` | 30 rows only; one input has no real effect |
| `messy.csv` | CRLF line endings, padding, quoted commas, `N/A`, constant/empty columns, blank rows |
| `company-value.csv` | inputs ~0.005 to ~1M, outcome ~1e9 |
| `sensor-large.csv` | 120k rows, bigger than the localStorage quota |
| `location-train.csv` / `location-verify.csv` | price with a neighbourhood effect (two pricey spots, one cheap) from latitude/longitude; 10 verify rows lack coordinates |

| `test-austin-training-data.csv` | Real Austin for-sale listings (training) |
| `test-austin-verify-data.csv` | Real Austin sold listings (verify) |

Tests:
- `test/collection.test.js` trains on each synthetic dataset. It checks the result against the exact least-squares solution (within 1%) and against the generating formula (within 10%).
- `test/batch.test.js` checks batch prediction: column matching, training-average fill-in, gap maths, the results CSV, and the Austin train-then-verify flow.
- `test/calc.test.js` covers CSV parsing, type detection and training on the housing sample.
- `test/location.test.js` checks the location adjustment: role detection, the left-out gain, shifts near the pricey and cheap spots, fade-out far away, missing coordinates, batch and group-by.
- `test/outliers.test.js` poisons a clean dataset with a few bad records and checks they are left out, the weights come back, and each group is screened on its own.
- `test/group.test.js` checks group-by training: one formula per group, dropped inputs, skipped groups, and routing in single and batch prediction.

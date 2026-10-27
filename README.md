# Linear Regression Lab

**Live demo:** https://shishiv30.github.io/regression/

Every push to `main` runs the tests, builds the site and publishes it to the `gh-pages` branch (`.github/workflows/deploy.yml`).

Client-side linear regression on your own CSV records. Nothing leaves the browser.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # static site in dist/
npm test         # calculation-layer tests
```

## Structure

- `src/calc/`: the calculation layer (no DOM)
  - `csv.js` parses the CSV (PapaParse)
  - `schema.js` detects number vs. enum, finds numeric min–max ranges, builds deduplicated enum dictionaries
  - `model.js` encodes the data (min-max scaling, one-hot against a baseline) and handles weights in original units, prediction and metrics
  - `trainer.js` runs gradient descent with TensorFlow.js (CPU backend) for N steps
  - `storage.js` saves the dataset, field setup and model in localStorage
- `src/ui/`: the presentation layer: top bar (import and fields), left panel (report and prediction), main canvas (weight graph)

## Test data collection

`npm run test-data` regenerates `public/test-data/`. Each dataset is built from a known formula plus noise, and the formula is recorded in `index.json` as `truth`. All of them can be loaded from the "Load test dataset…" menu.

| File | What it checks |
|---|---|
| `salary.csv` | 3 enums, a free-text column, age that tracks experience, `"85,000"` outcomes, missing cells |
| `fuel-efficiency.csv` | 8 inputs with strongly correlated engine specs |
| `ad-spend.csv` | 30 rows only; one input has no real effect |
| `messy.csv` | CRLF line endings, padding, quoted commas, `N/A`, constant/empty columns, blank rows |
| `company-value.csv` | inputs ~0.005 to ~1M, outcome ~1e9 |
| `sensor-large.csv` | 120k rows, bigger than the localStorage quota |

`test/collection.test.js` trains on each one. It checks the result against the exact least-squares solution (within 1%) and against the generating formula (within 10%).

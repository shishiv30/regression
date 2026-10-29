# Regression Core: Estimating a Property Price

Example data: `public/test-data/test-san-jose-training-data.csv` (sold) and `test-san-jose-verify-data.csv` (for sale).

---

## 1. The Goal

Estimate a home's price from its facts (beds, baths, sqft, zip...), using homes that already sold.

---

## 2. Input

A table of records. One row = one property.

| bedrooms | bathroomsTotal | sqftTotal | listingType.name | zipCode | closePrice |
|---|---|---|---|---|---|
| 3 | 2 | 1703 | SINGLE_FAMILY_HOUSE | 95118 | 1,600,000 |
| 4 | 2 | 1322 | SINGLE_FAMILY_HOUSE | 95130 | 2,120,000 |

---

## 3. Field Roles

Each column gets one role:

- **Outcome (y)**: what we predict, `closePrice`
- **Metric (x)**: a quantity
- **Dimension (x)**: a category
- **Label / Ignore**: not used, e.g. `address`, `mlsNumber`

---

## 4. Metric

A number you can add, compare and scale. Bigger means more.

Examples: `bedrooms`, `bathroomsTotal`, `sqftTotal`, `lotSizeSqft`, `yearBuilt`

Weight meaning: **price change per +1 unit** (e.g. $600 per sqft).

---

## 5. Dimension

A category you can only name, not rank. "95118" is not "bigger" than "95112".

Examples: `zipCode`, `listingType.name`

Weight meaning: **price shift vs. a baseline value** (e.g. zip 95130 is +$180k vs. the most common zip).

---

## 6. Output

1. **Weights**: one per metric, one per dimension value, plus an intercept (the formula)
2. **Estimate**: the formula applied to any new property
3. **Gap**: estimate vs. actual, to judge how good the formula is

---

## 7. The Formula

A weighted sum:

```
estimate = intercept
         + w_bed  × bedrooms
         + w_bath × bathrooms
         + w_sqft × sqft
         + w_zip[zipCode]
```

We don't pick the weights. We learn them from data.

---

## 8. Training: Learn the Weights

Use sold homes, where facts **and** true price are both known.

1. Start with all weights = 0
2. Estimate every training row
3. Measure error: average of (estimate − actual)²
4. Nudge each weight to shrink the error
5. Repeat for N steps (default 500)

This is **gradient descent** (`src/calc/trainer.js`). The loss curve should fall and flatten.

---

## 9. Tiny Example

| sqft | closePrice |
|---|---|
| 1,000 | $900k |
| 1,500 | $1,150k |
| 2,000 | $1,400k |
| 2,500 | $1,650k |

Training finds: `price = 400,000 + 500 × sqft`

- intercept = 400,000 (the line's starting point)
- w_sqft = 500 (each sqft adds $500)

Real data is never a perfect line, so training picks the weights with the smallest overall error.

---

## 10. Encoding Before Training

- **Metrics**: scaled to 0–1, so sqft (thousands) and beds (single digits) are on equal footing. Weights are converted back to real units afterward.
- **Dimensions**: one-hot against a baseline. Each non-baseline value gets a 0/1 column and its own weight.

```
zipCode      is_95130   is_95112
95118 (base)     0          0
95130            1          0
95112            0          1
```

---

## 11. Log Scale (optional)

Prices move by percentages, not flat dollars. With log scale the formula estimates `log(price)`, so a weight reads as **≈ % change** per unit.

The app turns it on automatically when the outcome is always positive and spans 10× or more.

---

## 12. Estimate a New Property

Training is done once. Estimating is just arithmetic.

Weights: intercept 150k, bed 40k, bath 60k, sqft 600, zip 95130 +180k

New listing: 3 bed, 2 bath, 1,800 sqft, zip 95130

```
150,000 + 3×40,000 + 2×60,000 + 1,800×600 + 180,000
= 150,000 + 120,000 + 120,000 + 1,080,000 + 180,000
= $1,650,000
```

---

## 13. Extras on Top of the Formula

- **Location adjustment / comparables**: nearby sold homes (lat/lng) correct for neighborhood effects
- **Range and confidence**: errors on held-out records become a low–high range
- **Missing values**: dropped, or filled with the median (numbers) or most common value (enums)

---

## 14. Compare with the Actual Value

Estimate homes the model **never trained on**, then compare to the real price.

```
estimate = 1,650,000
actual   = 1,720,000
gap      = actual − estimate = +70,000   (under-estimated)
gap %    = 70,000 / 1,720,000 = +4.1 %
```

In the app: **Predict many records** → gap chart (`src/calc/batch.js`).

---

## 15. Score the Whole Verify Set

| Metric | Meaning |
|---|---|
| **MAE** | Average \|gap\| in dollars |
| **RMSE** | Like MAE, punishes big misses harder |
| **Median \|gap %\|** | Typical % off |
| **R²** | Share of price variation explained (1 = perfect, 0 = no better than the average) |

---

## 16. Reading the Gaps

- Good on training, bad on verify → **overfitting**
- Gaps lean one way within a group → a **missing input**
- A few huge gaps → **outliers** (`outliers.js` can leave them out)
- For-sale listing vs. `listPrice` → an over/under-priced signal, not an error

---

## 17. The Whole Flow

```
sold listings (facts + closePrice)
   → assign roles (metric / dimension / outcome)
   → encode (scale, one-hot)
   → TRAIN (gradient descent) → weights
   → new property: Σ weight × value → ESTIMATE
   → verify set: GAP = actual − estimate → MAE / R²
```

---

## 18. Where to Look in Code

| Concept | File |
|---|---|
| CSV, number vs. enum | `src/calc/csv.js`, `schema.js` |
| Encode, predict, evaluate | `src/calc/model.js` |
| Training | `src/calc/trainer.js` |
| Batch estimate and gaps | `src/calc/batch.js` |
| Confidence, outliers, location | `confidence.js`, `outliers.js`, `location.js` |

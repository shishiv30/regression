// Outlier screening before training. A record is an outlier when the formula
// misses it by far more than it misses a typical record: its residual sits more
// than `cutoff` robust standard deviations (1.4826 × median absolute deviation)
// from the median residual. Residuals, not raw values, are compared, so a big
// pricey house is fine while a small house listed at a mansion price is not.
//
// The screening fit is exact least squares (fast, no gradient descent). It is
// repeated on the kept records until the flagged set stops changing, so one
// wild record cannot hide others by dragging the first fit towards itself.

export const DEFAULT_OUTLIER_CUTOFF = 3.5;
// Never drop more than this share: past it the data is just noisy, not poisoned.
const MAX_SHARE = 0.2;
const MAX_ROUNDS = 10;

const median = (xs) => {
  const s = Float64Array.from(xs).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// Solve min mean((X·w + b − y)²) + ridge·‖w‖² over the rows in `keep`. A
// tiny ridge is always added so unused enum levels or duplicate columns don't
// make the system singular.
export function leastSquares(X, y, keep = X.map((_, i) => i), ridge = 0) {
  return solve(normalEquations(X, y, keep), keep.length, ridge);
}

// Xᵀ·X | Xᵀ·y (upper triangle) with an intercept column, for solve().
function normalEquations(X, y, keep) {
  const d = X[0].length + 1; // last column is the intercept
  const A = Array.from({ length: d }, () => new Float64Array(d + 1));
  for (const i of keep) {
    const x = [...X[i], 1];
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) A[a][b] += x[a] * x[b];
      A[a][d] += x[a] * y[i];
    }
  }
  return A;
}

function solve(normal, n, ridge) {
  const d = normal.length;
  const A = normal.map((r) => Float64Array.from(r));
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    if (a < d - 1) A[a][a] += (ridge + 1e-9) * n + 1e-12;
  }
  // Gaussian elimination with partial pivoting.
  for (let c = 0; c < d; c++) {
    let p = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    if (A[c][c] === 0) continue;
    for (let r = c + 1; r < d; r++) {
      const f = A[r][c] / A[c][c];
      if (f) for (let k = c; k <= d; k++) A[r][k] -= f * A[c][k];
    }
  }
  const w = new Float64Array(d);
  for (let c = d - 1; c >= 0; c--) {
    let s = A[c][d];
    for (let k = c + 1; k < d; k++) s -= A[c][k] * w[k];
    w[c] = A[c][c] ? s / A[c][c] : 0;
  }
  return { weights: Array.from(w.subarray(0, d - 1)), bias: w[d - 1] };
}

// Each row's estimate from a fit that did not see it (k-fold). Used to
// measure honest errors for the likely ranges.
export function crossFitted(X, y, folds = 5, ridge = 0) {
  const out = new Array(X.length).fill(NaN);
  for (let f = 0; f < folds; f++) {
    const keep = [];
    for (let i = 0; i < X.length; i++) if (i % folds !== f) keep.push(i);
    if (keep.length < 2) continue;
    const { weights, bias } = leastSquares(X, y, keep, ridge);
    for (let i = f; i < X.length; i += folds) out[i] = X[i].reduce((a, v, j) => a + v * weights[j], bias);
  }
  return out;
}

// Ridge strengths tried, on inputs scaled to [0, 1]; 0 is plain least squares.
export const RIDGE_GRID = [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1];

// The ridge strength whose fits miss left-out records least (5-fold). With
// plenty of records per weight this is 0 or tiny; with few (a small group, an
// enum with many values) it pulls the weights towards 0, so the formula
// stops memorising its records.
export function chooseRidge(X, y, folds = 5) {
  if (X.length < 2 * folds) return 0;
  const parts = [];
  for (let f = 0; f < folds; f++) {
    const keep = [];
    for (let i = 0; i < X.length; i++) if (i % folds !== f) keep.push(i);
    parts.push({ f, normal: normalEquations(X, y, keep), n: keep.length });
  }
  let best = { ridge: 0, err: Infinity };
  for (const ridge of RIDGE_GRID) {
    let err = 0;
    for (const { f, normal, n } of parts) {
      const { weights, bias } = solve(normal, n, ridge);
      for (let i = f; i < X.length; i += folds) err += (X[i].reduce((a, v, j) => a + v * weights[j], bias) - y[i]) ** 2;
    }
    if (err < best.err * (1 - 1e-6)) best = { ridge, err };
  }
  return best.ridge;
}

// X, y: the encoded training data. Returns the indexes of outlier rows, each
// with its screening-fit estimate and robust z-score (most extreme first).
// local(residuals, keptIndexes), when given, is the residual each row's
// neighbourhood leads us to expect (see location.js); it is taken off first,
// so a pricey house in a pricey area is not flagged.
export function findOutliers(X, y, { cutoff = DEFAULT_OUTLIER_CUTOFF, local } = {}) {
  const n = X.length;
  if (!(cutoff > 0) || n < X[0].length + 4) return [];
  let flagged = new Set();
  let found = [];
  for (let round = 0; round < MAX_ROUNDS; round++) {
    const keep = [];
    for (let i = 0; i < n; i++) if (!flagged.has(i)) keep.push(i);
    const { weights, bias } = leastSquares(X, y, keep);
    const fit = X.map((x) => x.reduce((a, v, j) => a + v * weights[j], bias));
    let res = fit.map((f, i) => y[i] - f);
    if (local) {
      const expected = local(res, keep);
      res = res.map((r, i) => r - expected[i]);
    }
    const mid = median(keep.map((i) => res[i]));
    const scale = 1.4826 * median(keep.map((i) => Math.abs(res[i] - mid)));
    if (!(scale > 0)) return found; // a near-perfect fit: nothing stands out
    found = res
      .map((r, i) => ({ i, fit: fit[i], z: (r - mid) / scale }))
      .filter((o) => Math.abs(o.z) > cutoff)
      .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
      .slice(0, Math.floor(n * MAX_SHARE));
    const next = new Set(found.map((o) => o.i));
    if (next.size === flagged.size && [...next].every((i) => flagged.has(i))) break;
    flagged = next;
  }
  return found;
}

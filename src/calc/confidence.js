// How far off an estimate is likely to be. Every range comes from errors on
// records the estimate had not seen (left out of their own neighbours), except
// for formulas with no location, where only the training errors exist.
//
// With comparables, each estimate also gets a confidence level from two
// signals known before the real value: how far the formula and the
// comparables disagree, and how much the comparables' value per size unit
// varies. The training records' combined scores split into thirds give the
// high, medium and low levels, and each level gets its own range.

// An 80% range: the 10th to 90th percentile of the errors.
const Q_LO = 0.1, Q_HI = 0.9;
// A level needs this many records for its own range; else the overall one is used.
export const MIN_LEVEL_RECORDS = 10;
export const LEVELS = ['high', 'medium', 'low'];

const quantile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
export const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// Errors are ratios (actual ÷ estimate, as a log) when the outcome is always
// positive, so a range is a percentage of the estimate; otherwise differences.
export const rangeMode = (outcome) => (outcome.min > 0 ? 'ratio' : 'diff');

// pairs: [actual, estimate]. Returns the 80% error range, or null when too few.
export function errorRange(pairs, mode, source) {
  const errs = [], apes = [];
  for (const [a, e] of pairs) {
    const err = mode === 'ratio' ? (a > 0 && e > 0 ? Math.log(a / e) : NaN) : a - e;
    if (!Number.isFinite(err)) continue;
    errs.push(err);
    if (a !== 0) apes.push(Math.abs(a - e) / Math.abs(a));
  }
  if (errs.length < MIN_LEVEL_RECORDS) return null;
  errs.sort((x, y) => x - y);
  return { mode, source, n: errs.length, lo: quantile(errs, Q_LO), hi: quantile(errs, Q_HI), medianApe: apes.length ? median(apes) : null };
}

// The likely low and high values around an estimate.
export function applyRange(range, value) {
  if (!range) return null;
  if (range.mode === 'ratio') return value > 0 ? { low: value * Math.exp(range.lo), high: value * Math.exp(range.hi) } : null;
  return { low: value + range.lo, high: value + range.hi };
}

// Each signal is divided by its training median and the two are added; the
// training records' scores split into thirds give the level cut-offs.
export const scoreOf = (thresholds, disagree, spread) => disagree / (thresholds.disagree || 1e-12) + spread / (thresholds.spread || 1e-12);
export function levelOf(thresholds, disagree, spread) {
  const s = scoreOf(thresholds, disagree, spread);
  return s <= thresholds.high ? 'high' : s > thresholds.low ? 'low' : 'medium';
}

// Blend of the formula (+ location) and the comparables: a geometric mean
// when both are positive, so neither side's scale dominates.
export const blendEstimates = (a, b) => (a > 0 && b > 0 ? Math.sqrt(a * b) : (a + b) / 2);
export const disagreement = (a, b) => (a > 0 && b > 0 ? Math.abs(Math.log(a / b)) : Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-12));

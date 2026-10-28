// Location adjustment. After the formula is trained, each training record's
// residual (actual − formula estimate) is kept with its latitude/longitude.
// A new record's estimate is then shifted by the distance-weighted average
// residual of its K nearest training records: if the formula under-prices the
// houses around it, it probably under-prices this one too.
import { isMissing, toNumber } from './schema.js';
import { toModel, toValue } from './outcome.js';
import { median, errorRange, rangeMode, levelOf, scoreOf, LEVELS, MIN_LEVEL_RECORDS, blendEstimates, disagreement } from './confidence.js';

export const DEFAULT_NEIGHBOURS = 10;
// Candidate distance scales; training keeps the one that best predicts each
// record's residual from its neighbours (with the record itself left out).
const SCALES_KM = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
// Shrinks the adjustment towards 0 when the neighbours are few or far away:
// it acts like one extra neighbour at distance 0 whose residual is 0.
const PRIOR_WEIGHT = 1;
// Above this many records, the left-out check runs on an even sample.
const CHECK_SAMPLE = 2000;
const MIN_POINTS = 5;

// The two fields marked as latitude and longitude, or null when not both are.
export function locationFields(config) {
  const cols = Object.keys(config.roles ?? {});
  const lat = cols.find((c) => config.roles[c] === 'lat');
  const lng = cols.find((c) => config.roles[c] === 'lng');
  return lat && lng ? { lat, lng } : null;
}

export function readCoords(record, lat, lng) {
  if (isMissing(record[lat]) || isMissing(record[lng])) return null;
  const a = toNumber(record[lat]), b = toNumber(record[lng]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) > 90 || Math.abs(b) > 180) return null;
  return { lat: a, lng: b };
}

// Flat-earth distance in km; accurate to well under 1% at city scale.
export function distanceKm(lat1, lng1, lat2, lng2) {
  const dy = (lat2 - lat1) * 110.57;
  const dx = (lng2 - lng1) * 111.32 * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  return Math.hypot(dx, dy);
}

// Indexes and distances of the k points nearest to (lat, lng), closest first.
function nearest(points, lat, lng, k, skip = -1) {
  const best = []; // sorted by distance, at most k long
  for (let i = 0; i < points.length; i++) {
    if (i === skip) continue;
    const d = distanceKm(lat, lng, points[i][0], points[i][1]);
    if (best.length === k && d >= best[k - 1].d) continue;
    let j = best.length < k ? best.length : k - 1;
    while (j > 0 && best[j - 1].d > d) {
      best[j] = best[j - 1];
      j--;
    }
    best[j] = { i, d };
  }
  return best;
}

const kernel = (d, scaleKm) => Math.exp(-0.5 * (d / scaleKm) ** 2);

function blend(points, neighbours, scaleKm) {
  let sw = 0, swr = 0;
  for (const { i, d } of neighbours) {
    const w = kernel(d, scaleKm);
    sw += w;
    swr += w * points[i][2];
  }
  return swr / (sw + PRIOR_WEIGHT);
}

const round = (v, digits) => Math.round(v * 10 ** digits) / 10 ** digits;

// Comparables: the k training records nearest to a point that are also
// similar in size. Distance in km plus SIZE_KM per unit of |log size ratio|,
// so a record half the size counts like one ~0.35 km further away.
const SIZE_KM = 0.5;
function comparables(points, lat, lng, size, k, skip = -1) {
  const best = [];
  const ls = Math.log(size);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (i === skip || !(p[5] > 0) || !(p[3] > 0)) continue;
    const d = distanceKm(lat, lng, p[0], p[1]);
    const s = d + SIZE_KM * Math.abs(Math.log(p[5]) - ls);
    if (best.length === k && s >= best[k - 1].s) continue;
    let j = best.length < k ? best.length : k - 1;
    while (j > 0 && best[j - 1].s > s) {
      best[j] = best[j - 1];
      j--;
    }
    best[j] = { i, d, s };
  }
  if (!best.length) return null;
  // Value per size unit (as a log), its median and how much it varies.
  const per = best.map(({ i }) => Math.log(points[i][3] / points[i][5]));
  const mid = median(per);
  return { value: Math.exp(mid) * size, spread: median(per.map((v) => Math.abs(v - mid))), near: best };
}

// rows: the records the formula was trained on; estimate(row) is the
// formula's value for a row in model space (log(outcome) on a log scale).
// outcome: the model's outcome ({ name, log, min }). size: the field
// comparables are matched and priced by, or null for none.
// holdout: records kept out of training (outliers); they only widen the ranges.
// Returns { location } or { skipped: reason }.
// unseen(row), when given, is the row's formula estimate from a fit that did
// not see it; the left-out check uses it instead of estimate(row).
export function buildLocation(rows, estimate, { lat, lng }, outcome, { neighbours = DEFAULT_NEIGHBOURS, rowIndex, size = null, holdout = [], unseen = estimate } = {}) {
  const points = []; // [lat, lng, residual (model space), actual, training row index, size]
  const source = []; // the row behind each point
  for (const row of rows) {
    const c = readCoords(row, lat, lng);
    if (!c) continue;
    source.push(row);
    const actual = toNumber(row[outcome.name]);
    const p = [round(c.lat, 6), round(c.lng, 6), toModel(outcome, actual) - estimate(row), actual, rowIndex?.get(row) ?? -1];
    if (size) p.push(toNumber(row[size]));
    points.push(p);
  }
  if (points.length < MIN_POINTS) {
    return { skipped: `only ${points.length} records have a usable ${lat}/${lng} (needs ${MIN_POINTS})` };
  }
  const k = Math.max(1, Math.min(Math.round(neighbours), points.length - 1));

  // Left-out check: predict each record's residual from its neighbours only.
  const step = Math.max(1, points.length / CHECK_SAMPLE);
  const sample = [];
  for (let s = 0; s < points.length; s += step) sample.push(Math.floor(s));
  const near = sample.map((i) => nearest(points, points[i][0], points[i][1], k, i));

  let best = null;
  for (const scaleKm of SCALES_KM) {
    const adj = sample.map((i, s) => blend(points, near[s], scaleKm));
    const sse = sample.reduce((a, i, s) => a + (points[i][2] - adj[s]) ** 2, 0);
    if (!best || sse < best.sse) best = { scaleKm, sse, adj };
  }

  // Fit figures in outcome units: formula alone, + location, + comparables,
  // every record left out of its own neighbours.
  const actual = sample.map((i) => points[i][3]);
  const formula = sample.map((i) => unseen(source[i]));
  const before = formula.map((f) => toValue(outcome, f));
  const after = formula.map((f, s) => toValue(outcome, f + best.adj[s]));
  const comps = size ? sample.map((i) => comparables(points, points[i][0], points[i][1], points[i][5], k, i)) : null;
  const usable = (s) => comps?.[s] && points[sample[s]][5] > 0;
  const final = after.map((v, s) => (usable(s) ? blendEstimates(v, comps[s].value) : v));
  const mean = actual.reduce((a, b) => a + b, 0) / sample.length;
  const sst = actual.reduce((a, v) => a + (v - mean) ** 2, 0);
  const sse = (est) => est.reduce((a, e, s) => a + (actual[s] - e) ** 2, 0);
  const sorted = [...best.adj].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

  // How far off the final estimates were: overall, and per confidence level
  // when there are comparables. Records left out as outliers count too: new
  // records will include ones like them, and a range that ignores them is
  // too narrow.
  const mode = rangeMode(outcome);
  const pairs = actual.map((a, s) => [a, final[s]]);
  const signals = sample.map((_, s) => (usable(s) ? { disagree: disagreement(after[s], comps[s].value), spread: comps[s].spread } : null));
  for (const row of holdout) {
    const c = readCoords(row, lat, lng);
    const a = toNumber(row[outcome.name]);
    if (!c || !Number.isFinite(a)) continue;
    const at = toValue(outcome, estimate(row) + blend(points, nearest(points, c.lat, c.lng, k), best.scaleKm));
    const sz = size ? toNumber(row[size]) : NaN;
    const cp = sz > 0 ? comparables(points, c.lat, c.lng, sz, k) : null;
    pairs.push([a, cp ? blendEstimates(at, cp.value) : at]);
    signals.push(cp ? { disagree: disagreement(at, cp.value), spread: cp.spread } : null);
  }
  const range = errorRange(pairs, mode, 'left-out');
  let confidence = null;
  if (comps) {
    const known = signals.filter(Boolean);
    if (known.length >= 3 * MIN_LEVEL_RECORDS) {
      const thresholds = { disagree: median(known.map((x) => x.disagree)), spread: median(known.map((x) => x.spread)) };
      const scores = known.map((x) => scoreOf(thresholds, x.disagree, x.spread)).sort((a, b) => a - b);
      thresholds.high = scores[Math.floor(scores.length / 3)];
      thresholds.low = scores[Math.floor((2 * scores.length) / 3)];
      const levels = {};
      for (const level of LEVELS) {
        levels[level] = errorRange(pairs.filter((_, s) => signals[s] && levelOf(thresholds, signals[s].disagree, signals[s].spread) === level), mode, 'left-out');
      }
      confidence = { thresholds, levels };
    }
  }

  for (const p of points) {
    p[2] = round(p[2], 4);
    p[3] = round(p[3], 4);
  }
  return {
    location: {
      lat,
      lng,
      neighbours: k,
      scaleKm: best.scaleKm,
      points,
      // How far the adjustment typically moves an estimate in model space (10th to 90th percentile).
      spread: q(0.9) - q(0.1),
      size,
      range,
      confidence,
      metrics: locationMetrics({ n: sample.length, mean, sst, sseBefore: sse(before), sseAfter: sse(after), ...(comps ? { sseComps: sse(final) } : {}) }),
    },
  };
}

// R² and RMSE of the formula alone vs. formula + location, from sums that can
// be combined across groups.
export function locationMetrics({ n, mean, sst, sseBefore, sseAfter, sseComps }) {
  return {
    n, mean, sst, sseBefore, sseAfter,
    r2Before: sst > 0 ? 1 - sseBefore / sst : NaN,
    r2After: sst > 0 ? 1 - sseAfter / sst : NaN,
    rmseBefore: Math.sqrt(sseBefore / n),
    rmseAfter: Math.sqrt(sseAfter / n),
    ...(sseComps === undefined ? {} : { sseComps, r2Comps: sst > 0 ? 1 - sseComps / sst : NaN, rmseComps: Math.sqrt(sseComps / n) }),
  };
}

// Pool per-group sums into one overall figure (total SS around the overall mean).
export function combineLocationMetrics(list) {
  const n = list.reduce((a, m) => a + m.n, 0);
  if (!n) return null;
  const mean = list.reduce((a, m) => a + m.n * m.mean, 0) / n;
  return locationMetrics({
    n,
    mean,
    sst: list.reduce((a, m) => a + m.sst + m.n * (m.mean - mean) ** 2, 0),
    sseBefore: list.reduce((a, m) => a + m.sseBefore, 0),
    sseAfter: list.reduce((a, m) => a + m.sseAfter, 0),
    // A group without comparables counts with its location-only error.
    ...(list.some((m) => m.sseComps !== undefined) ? { sseComps: list.reduce((a, m) => a + (m.sseComps ?? m.sseAfter), 0) } : {}),
  });
}

// For outlier screening: a function giving each row the median residual of
// its k nearest other rows (among those still kept), so a record is judged
// against its neighbourhood rather than the whole map. Rows without usable
// coordinates get 0. Candidates come from a coarse grid, so this stays fast
// on large files.
export function neighbourResiduals(rows, { lat, lng }, k = DEFAULT_NEIGHBOURS) {
  const pts = rows.map((r) => readCoords(r, lat, lng));
  const near = nearestAll(pts, 2 * k);
  return (res, keep) => {
    const kept = new Uint8Array(rows.length);
    for (const i of keep) kept[i] = 1;
    return near.map((list) => {
      const rs = [];
      for (const j of list) if (kept[j] && rs.push(res[j]) === k) break;
      if (!rs.length) return 0;
      rs.sort((a, b) => a - b);
      const m = rs.length >> 1;
      return rs.length % 2 ? rs[m] : (rs[m - 1] + rs[m]) / 2;
    });
  };
}

// Indexes of the m nearest other points for every point (closest first).
function nearestAll(pts, m) {
  const idx = pts.map((p, i) => (p ? i : -1)).filter((i) => i >= 0);
  const out = pts.map(() => []);
  if (idx.length < 2) return out;
  let [minLat, maxLat, minLng, maxLng] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const i of idx) {
    minLat = Math.min(minLat, pts[i].lat); maxLat = Math.max(maxLat, pts[i].lat);
    minLng = Math.min(minLng, pts[i].lng); maxLng = Math.max(maxLng, pts[i].lng);
  }
  // Square-ish cells holding ~4 points each on average.
  const cell = Math.max(1e-6, Math.sqrt(((maxLat - minLat) * (maxLng - minLng) || 1e-6) / (idx.length / 4)));
  const key = (a, b) => `${a},${b}`;
  const cx = (p) => Math.floor((p.lng - minLng) / cell), cy = (p) => Math.floor((p.lat - minLat) / cell);
  const grid = new Map();
  for (const i of idx) {
    const kk = key(cx(pts[i]), cy(pts[i]));
    if (!grid.has(kk)) grid.set(kk, []);
    grid.get(kk).push(i);
  }
  const maxRing = Math.ceil(Math.max(maxLat - minLat, maxLng - minLng) / cell) + 1;
  // km per grid cell, the smaller of the two directions: a safe stopping bound.
  const kmPerCell = cell * Math.min(110.57, 111.32 * Math.cos((Math.max(Math.abs(minLat), Math.abs(maxLat)) * Math.PI) / 180));
  for (const i of idx) {
    const p = pts[i], x = cx(p), y = cy(p);
    const best = [];
    for (let r = 0; r <= maxRing; r++) {
      for (let dx = -r; dx <= r; dx++) {
        for (let dy = -r; dy <= r; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          for (const j of grid.get(key(x + dx, y + dy)) ?? []) {
            if (j !== i) best.push({ j, d: distanceKm(p.lat, p.lng, pts[j].lat, pts[j].lng) });
          }
        }
      }
      best.sort((a, b) => a.d - b.d);
      best.length = Math.min(best.length, m);
      // Every point beyond ring r is at least r cells away.
      if (best.length === Math.min(m, idx.length - 1) && best.at(-1).d <= r * kmPerCell) break;
    }
    out[i] = best.map((b) => b.j);
  }
  return out;
}

// The comparables estimate for one record: the median value per size unit of
// its nearest similar-sized training records, times its size. Null without
// comparables, usable coordinates or a positive size.
export function comparablesEffect(location, record) {
  if (!location.size) return null;
  const c = readCoords(record, location.lat, location.lng);
  const size = toNumber(record[location.size]);
  if (!c || !(size > 0)) return null;
  const comps = comparables(location.points, c.lat, c.lng, size, location.neighbours);
  if (!comps) return null;
  const { points } = location;
  return {
    value: comps.value,
    spread: comps.spread,
    size,
    neighbours: comps.near.map(({ i, d }) => ({ distanceKm: d, actual: points[i][3], size: points[i][5], row: points[i][4] })),
  };
}

// The adjustment for one record, with the neighbours behind it.
// Returns null when the record has no usable coordinates.
export function locationEffect(location, record) {
  const c = readCoords(record, location.lat, location.lng);
  if (!c) return null;
  const { points, neighbours: k, scaleKm } = location;
  const near = nearest(points, c.lat, c.lng, k);
  const weights = near.map(({ d }) => kernel(d, scaleKm));
  const total = weights.reduce((a, b) => a + b, 0) + PRIOR_WEIGHT;
  return {
    lat: c.lat,
    lng: c.lng,
    adjustment: blend(points, near, scaleKm),
    neighbours: near.map(({ i, d }, j) => ({
      distanceKm: d,
      residual: points[i][2],
      actual: points[i][3],
      row: points[i][4],
      share: weights[j] / total, // this neighbour's share of the adjustment
    })),
  };
}

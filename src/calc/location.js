// Location adjustment. After the formula is trained, each training record's
// residual (actual − formula estimate) is kept with its latitude/longitude.
// A new record's estimate is then shifted by the distance-weighted average
// residual of its K nearest training records: if the formula under-prices the
// houses around it, it probably under-prices this one too.
import { isMissing, toNumber } from './schema.js';

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

// rows: the records the formula was trained on; estimate(row) is the
// formula's value for a row. Returns { location } or { skipped: reason }.
export function buildLocation(rows, estimate, { lat, lng }, outcome, { neighbours = DEFAULT_NEIGHBOURS, rowIndex } = {}) {
  const points = []; // [lat, lng, residual, actual, training row index]
  for (const row of rows) {
    const c = readCoords(row, lat, lng);
    if (!c) continue;
    const actual = toNumber(row[outcome]);
    points.push([round(c.lat, 6), round(c.lng, 6), actual - estimate(row), actual, rowIndex?.get(row) ?? -1]);
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

  const actual = sample.map((i) => points[i][3]);
  const mean = actual.reduce((a, b) => a + b, 0) / sample.length;
  const sst = actual.reduce((a, v) => a + (v - mean) ** 2, 0);
  const sseBefore = sample.reduce((a, i) => a + points[i][2] ** 2, 0);
  const sorted = [...best.adj].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];

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
      // How far the adjustment typically moves an estimate (10th to 90th percentile).
      spread: q(0.9) - q(0.1),
      metrics: locationMetrics({ n: sample.length, mean, sst, sseBefore, sseAfter: best.sse }),
    },
  };
}

// R² and RMSE of the formula alone vs. formula + location, from sums that can
// be combined across groups.
export function locationMetrics({ n, mean, sst, sseBefore, sseAfter }) {
  return {
    n, mean, sst, sseBefore, sseAfter,
    r2Before: sst > 0 ? 1 - sseBefore / sst : NaN,
    r2After: sst > 0 ? 1 - sseAfter / sst : NaN,
    rmseBefore: Math.sqrt(sseBefore / n),
    rmseAfter: Math.sqrt(sseAfter / n),
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
  });
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

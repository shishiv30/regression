// Batch prediction: estimate the outcome for every row of a second CSV
// (e.g. sold listings) and, where the row has a real value, measure the gap.
import Papa from 'papaparse';
import { isMissing, toNumber, analyzeColumns } from './schema.js';
import { predict, isGrouped, resolveModel, noGroupReason, modelInputs } from './model.js';
import { readCoords } from './location.js';
import { LEVELS } from './confidence.js';

// Default mapping, by names the model knows and by the values in the file:
//  · each model input reads the same-named column;
//  · the actual value is the outcome's column, else the unused numeric column
//    with at least half its values inside the training outcome range whose
//    values look most like the outcome: middle and spread closest to the
//    training range's (so a code such as a ZIP, which sits inside a price
//    range but barely varies, is not taken for a price);
//  · the row label is a field marked Label in training, else an unused text
//    column with a different value on (nearly) every row.
export function suggestMapping(model, columns, rows = []) {
  const inputs = {};
  for (const t of modelInputs(model)) inputs[t.name] = columns.includes(t.name) ? t.name : null;
  const taken = new Set(Object.values(inputs));
  const free = columns.filter((c) => !taken.has(c));
  const stats = rows.length ? analyzeColumns(free, rows) : {};
  const actual = columns.includes(model.outcome.name) ? model.outcome.name : likeOutcome(model.outcome, free, stats, rows);
  const label =
    (model.labels ?? []).find((c) => columns.includes(c) && c !== actual) ??
    free.find((c) => c !== actual && stats[c]?.type === 'enum' && stats[c].values.length >= 0.9 * stats[c].count && stats[c].count > 1) ??
    null;
  return { inputs, actual, label };
}

function likeOutcome({ min, max }, columns, stats, rows) {
  const log = min > 0;
  // Positions on the outcome's own scale: log when it is all above 0.
  const at = (v) => (log ? Math.log(Math.max(v, min * 1e-6)) : v);
  const lo = at(min), hi = at(max), span = hi - lo || 1;
  let best = null;
  for (const c of columns) {
    if (stats[c]?.type !== 'number') continue;
    const vals = rows.map((r) => toNumber(r[c])).filter(Number.isFinite).sort((a, b) => a - b);
    if (!vals.length) continue;
    const inside = vals.filter((v) => v >= min && v <= max).length / vals.length;
    if (inside < 0.5) continue;
    const q = (p) => at(vals[Math.min(vals.length - 1, Math.floor(p * vals.length))]);
    const off = Math.abs(q(0.5) - (lo + hi) / 2) / span;
    // Its 5–95% spread against the training range, as a ratio (a column with
    // no spread at all is the least like it).
    const spread = Math.abs(Math.log(Math.max(q(0.95) - q(0.05), span * 1e-6) / span));
    const score = off + spread + (1 - inside);
    if (!best || score < best.score) best = { c, score };
  }
  return best?.c ?? null;
}

// Value used when a row has no usable input: the training fill value when the
// model was trained with blanks filled (median / most common value), else the
// training mean for numbers (range midpoint for models saved before means were
// recorded) and the baseline (most common value) for enums.
export function fallbackValue(t) {
  if (t.fill !== undefined) return t.fill;
  if (t.kind === 'number') return t.mean ?? (t.min + t.max) / 2;
  return t.baseline;
}

// The record a formula reads for one batch row: mapped values, with the
// training fallback for blank or unusable inputs (listed in `imputed`).
function recordOf(model, formula, group, row, mapping) {
  const record = isGrouped(model) ? { [model.groupBy]: group } : {};
  const imputed = [];
  for (const t of formula.terms) {
    const col = mapping.inputs[t.name];
    const raw = col ? row[col] : '';
    const usable = !isMissing(raw) && (t.kind !== 'number' || (t.log ? toNumber(raw) > 0 : Number.isFinite(toNumber(raw))));
    if (usable) record[t.name] = raw;
    else {
      record[t.name] = fallbackValue(t);
      imputed.push(t.name);
    }
  }
  // Coordinates are not averaged in: without them the row just gets no
  // location adjustment.
  const missingLocation = [];
  if (formula.location) {
    const { lat, lng } = formula.location;
    const coords = { [lat]: mapping.inputs[lat] ? row[mapping.inputs[lat]] : '', [lng]: mapping.inputs[lng] ? row[mapping.inputs[lng]] : '' };
    if (readCoords(coords, lat, lng)) Object.assign(record, coords);
    else missingLocation.push(...[[lat, 90], [lng, 180]].filter(([f, max]) => !(Math.abs(toNumber(coords[f])) <= max)).map(([f]) => f));
    // The comparables' size field, when it is not a formula input.
    const size = formula.location.size;
    if (size && !(size in record) && mapping.inputs[size]) record[size] = row[mapping.inputs[size]];
  }
  return { record, imputed, missingLocation };
}

export function predictBatch(model, rows, mapping) {
  const results = rows.map((row, index) => {
    const label = mapping.label && !isMissing(row[mapping.label]) ? String(row[mapping.label]) : `row ${index + 1}`;
    const raw = mapping.actual ? toNumber(row[mapping.actual]) : NaN;
    // 0 or below where every training value is above 0 is a placeholder
    // (e.g. "no price yet"), not a value to measure the estimate against.
    const placeholder = Number.isFinite(raw) && raw <= 0 && model.outcome.min > 0;
    const actual = placeholder ? NaN : raw;
    const has = Number.isFinite(actual);
    const actualNote = placeholder ? `${row[mapping.actual]} read as blank: every training ${model.outcome.name} is above 0` : null;

    // A grouped model uses the formula of the row's group; with no formula
    // for that group the row gets no estimate (it is never guessed).
    let group = null;
    let formula = model;
    if (isGrouped(model)) {
      const col = mapping.inputs[model.groupBy];
      group = col && !isMissing(row[col]) ? String(row[col]).trim() : '';
      formula = resolveModel(model, group);
      if (!formula) {
        return { index, label, group, estimate: null, actual: has ? actual : null, actualNote, gap: null, gapPct: null, imputed: [], missingLocation: [], locationAdj: null, compsAdj: null, low: null, high: null, confidence: null, warnings: [], suspect: [], error: noGroupReason(model, group) };
      }
    }

    const { record, imputed, missingLocation } = recordOf(model, formula, group, row, mapping);
    const p = predict(model, record);
    const suspect = has ? screenRow(formula, row, mapping, actual) : [];
    const amount = (key) => p.contributions.find((c) => c[key])?.contribution ?? null;
    const gap = has ? actual - p.value : null;
    return {
      index,
      label,
      group,
      estimate: p.value,
      actual: has ? actual : null,
      actualNote,
      gap,
      gapPct: has && actual !== 0 ? gap / actual : null,
      imputed,
      missingLocation,
      locationAdj: amount('location'),
      compsAdj: amount('comps'),
      low: p.confidence?.low ?? null,
      high: p.confidence?.high ?? null,
      confidence: p.confidence?.level ?? null,
      warnings: p.warnings,
      suspect,
    };
  });
  return { results, summary: summarizeResults(results) };
}

// How far outside the training range an input may be (as a share of that
// range) before the row is a different kind of record than the formula saw.
export const RANGE_SLACK = 0.25;

// Why a row's actual value can't fairly score the formula, judged without the
// estimate (with the same cutoff as the training outlier screen):
//  · its actual value per size unit is out of line with every training record
//    of its kind (a typo, or a different kind of record);
//  · an input is far outside anything the formula was trained on.
// Such rows still get an estimate; they are left out of the gap figures.
function screenRow(formula, row, mapping, actual) {
  const out = [];
  const sc = formula.screen;
  if (!sc) return out;
  const size = sc.size ? toNumber(mapping.inputs[sc.size] ? row[mapping.inputs[sc.size]] : NaN) : 1;
  if (sc.log ? actual > 0 && size > 0 : true) {
    const z = ((sc.log ? Math.log(actual / size) : actual) - sc.median) / sc.spread;
    if (Math.abs(z) > sc.cutoff) out.push(`actual ${sc.size ? `per ${sc.size} ` : ''}is ${Math.abs(z).toFixed(1)}σ ${z > 0 ? 'above' : 'below'} the training records`);
  }
  for (const t of formula.terms) {
    if (t.kind !== 'number' || !mapping.inputs[t.name]) continue;
    const v = toNumber(row[mapping.inputs[t.name]]);
    const slack = RANGE_SLACK * (t.max - t.min);
    if (Number.isFinite(v) && (v < t.min - slack || v > t.max + slack)) out.push(`${t.name} ${v} is far outside the training range ${t.min} – ${t.max}`);
  }
  return out;
}

// The share of compared rows counted as the biggest gaps (at least BIG_MIN).
const BIG_SHARE = 0.1, BIG_MIN = 5;
// An enum value seen fewer times than this in training is rare.
const RARE = 5;
// The fewest compared rows worth diagnosing.
export const DIAGNOSE_MIN = 20;
// A field only hurts when leaving it out also cuts the median |gap| by at
// least this much (absolute) and doesn't raise the mean |gap|: a tiny but
// consistent change is not worth acting on.
const MIN_GAIN = 0.001;
// A paired test must find the gaps different at about 95% (|z| ≥ 1.96)
// before a field is said to help or hurt.
const Z_95 = 1.96;
// A pattern among the biggest gaps: at least this share of them, and at
// least LIFT × its share among the other rows.
const PATTERN_SHARE = 0.2, LIFT = 2;

// Wilcoxon signed-rank test (normal approximation) on paired differences:
// z > 0 when the differences lean positive. Zeros are dropped; ties share
// their average rank.
export function signedRankZ(diffs) {
  const d = diffs.filter((v) => v !== 0).map((v) => ({ v, a: Math.abs(v) })).sort((x, y) => x.a - y.a);
  const n = d.length;
  if (n < 10) return 0;
  let wPlus = 0, tie = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && d[j + 1].a === d[i].a) j++;
    const rank = (i + j + 2) / 2, t = j - i + 1;
    tie += t ** 3 - t;
    for (let k = i; k <= j; k++) if (d[k].v > 0) wPlus += rank;
    i = j + 1;
  }
  const mean = (n * (n + 1)) / 4;
  const sd = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24 - tie / 48);
  return sd > 0 ? (wPlus - mean) / sd : 0;
}

// What drives the gaps of one formula, from batch rows that have an actual
// value and passed the screen (only `group`'s rows for a grouped model).
// For each input (and the location, as one field):
//  · without: median and mean |gap %| with the field set to its training
//    fallback (mean / most common value; for the location, no location
//    adjustment or comparables), a drop-feature check. A paired signed-rank
//    test on each row's |gap %| with and without it gives the verdict:
//    'hurts' (Ignore is suggested), 'helps' or 'unclear';
//  · driver: among the biggest gaps (top 10% |gap %|) and the other rows, the
//    share where this field moved the estimate furthest in the direction of
//    its error;
//  · unusual: the same shares for an unusual value: blank (fallback used),
//    outside the training range, or an enum value unseen or seen fewer than
//    5 times in training.
// Returns null with fewer than DIAGNOSE_MIN rows.
export function diagnoseGaps(model, rows, mapping, results, group = null) {
  const formula = isGrouped(model) ? resolveModel(model, group) : model;
  if (!formula) return null;
  const rs = results.filter((r) => r.gap !== null && r.gapPct !== null && !r.suspect?.length && (!isGrouped(model) || r.group === group));
  if (rs.length < DIAGNOSE_MIN) return null;
  const base = rs.map((r) => ({ r, ...recordOf(model, formula, r.group, rows[r.index], mapping) }));
  const ape = (est, r) => Math.abs((r.actual - est) / r.actual);
  const stats = (apes) => ({ medianApe: median(apes), mape: apes.reduce((a, v) => a + v, 0) / apes.length });
  const own = rs.map((r) => Math.abs(r.gapPct));
  const before = stats(own);
  // Median/mean |gap %| without a field, and the paired test against `own`.
  const compare = (apes) => {
    const z = signedRankZ(apes.map((a, i) => own[i] - a));
    const without = stats(apes);
    const hurts = z >= Z_95 && before.medianApe - without.medianApe >= MIN_GAIN && without.mape <= before.mape;
    return { without, z, verdict: hurts ? 'hurts' : z <= -Z_95 ? 'helps' : 'unclear' };
  };

  const cut = [...rs].map((r) => Math.abs(r.gapPct)).sort((a, b) => b - a)[Math.max(BIG_MIN, Math.ceil(BIG_SHARE * rs.length)) - 1];
  const big = base.map((b) => Math.abs(b.r.gapPct) >= cut);
  const nBig = big.filter(Boolean).length, nRest = base.length - nBig;
  const share = (flags) => ({
    big: flags.filter((f, i) => f && big[i]).length / nBig,
    rest: nRest ? flags.filter((f, i) => f && !big[i]).length / nRest : 0,
  });

  // The part that pushed each estimate furthest towards its error.
  const drivers = base.map(({ r, record }) => {
    const err = Math.sign(r.estimate - r.actual);
    let best = null;
    for (const c of predict(model, record).contributions) {
      const push = c.contribution * err;
      if (push > 0 && (!best || push > best.push)) best = { push, name: c.location ? 'location' : c.comps ? 'location' : c.name };
    }
    return best?.name ?? null;
  });

  const fields = [];
  for (const t of formula.terms) {
    const test = compare(base.map(({ r, record }) => ape(predict(model, { ...record, [t.name]: fallbackValue(t) }).value, r)));
    const unusual = base.map(({ record, imputed }) => {
      if (imputed.includes(t.name)) return true;
      if (t.kind === 'number') {
        const v = toNumber(record[t.name]);
        return v < t.min || v > t.max;
      }
      const v = String(record[t.name]).trim();
      if (!(v in t.weights)) return true;
      return t.freq && formula.training?.rowsUsed ? t.freq[v] * formula.training.rowsUsed < RARE : false;
    });
    fields.push({ name: t.name, columns: [t.name], kind: t.kind, ...test, driver: share(drivers.map((d) => d === t.name)), unusual: share(unusual) });
  }
  if (formula.location) {
    const { lat, lng } = formula.location;
    const test = compare(base.map(({ r, record }) => ape(predict(model, record, { location: false }).value, r)));
    fields.push({ name: `${lat} / ${lng}`, columns: [lat, lng], kind: 'location', ...test, driver: share(drivers.map((d) => d === 'location')), unusual: share(base.map((b) => b.missingLocation.length > 0)) });
  }
  const pattern = ({ big, rest }) => big >= PATTERN_SHARE && big >= LIFT * rest;
  for (const f of fields) {
    f.gain = before.medianApe - f.without.medianApe;
    f.suggest = f.verdict === 'hurts';
    f.driverPattern = pattern(f.driver);
    f.unusualPattern = pattern(f.unusual);
  }
  // Hurting fields first, then by how much the estimates lean on the field.
  const order = { hurts: 0, unclear: 1, helps: 2 };
  fields.sort((a, b) => order[a.verdict] - order[b.verdict] || (a.verdict === 'helps' ? a.gain - b.gain : b.gain - a.gain));
  return { rows: rs.length, bigRows: nBig, bigCut: cut, before, fields };
}

// Gap statistics for a set of batch results (all rows, or one group's).
// Rows flagged by the screen are counted but left out of the gap figures.
export function summarizeResults(results) {
  // Compared = has both an estimate and an actual value, and passed the screen.
  const withActual = results.filter((r) => r.gap !== null && !r.suspect?.length);
  const n = withActual.length;
  const summary = {
    rows: results.length,
    withActual: n,
    screened: results.filter((r) => r.gap !== null && r.suspect?.length).length,
    imputedRows: results.filter((r) => r.imputed.length || r.missingLocation.length).length,
    noEstimate: results.filter((r) => r.estimate === null).length,
  };
  if (!n) return summary;
  const pcts = withActual.filter((r) => r.gapPct !== null).map((r) => Math.abs(r.gapPct));
  // Did the ranges hold, and is high confidence really better?
  const ranged = withActual.filter((r) => r.low !== null);
  const byLevel = {};
  for (const level of LEVELS) {
    const rs = withActual.filter((r) => r.confidence === level && r.gapPct !== null);
    if (rs.length) byLevel[level] = { n: rs.length, medianApe: median(rs.map((r) => Math.abs(r.gapPct))) };
  }
  return {
    ...summary,
    inRange: ranged.length ? ranged.filter((r) => r.actual >= r.low && r.actual <= r.high).length / ranged.length : null,
    byLevel,
    meanGap: withActual.reduce((a, r) => a + r.gap, 0) / n,
    mae: withActual.reduce((a, r) => a + Math.abs(r.gap), 0) / n,
    rmse: Math.sqrt(withActual.reduce((a, r) => a + r.gap ** 2, 0) / n),
    mape: pcts.length ? pcts.reduce((a, b) => a + b, 0) / pcts.length : null,
    medianApe: pcts.length ? median(pcts) : null,
    within10: pcts.length ? pcts.filter((p) => p <= 0.1).length / pcts.length : null,
  };
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Original columns plus the estimate and gap, for download.
export function resultsToCsv(columns, rows, results, outcomeName) {
  const located = results.some((r) => r.locationAdj !== null);
  const compared = results.some((r) => r.compsAdj !== null);
  const ranged = results.some((r) => r.low !== null);
  const rated = results.some((r) => r.confidence !== null);
  const num = (v) => (v === null ? '' : v.toFixed(2));
  const extra = [
    `est_${outcomeName}`,
    ...(ranged ? [`est_low`, `est_high`] : []),
    ...(rated ? ['confidence'] : []),
    ...(located ? ['location_adj'] : []),
    ...(compared ? ['comparables_adj'] : []),
    'gap', 'gap_pct', 'imputed_fields', 'note',
  ];
  return Papa.unparse({
    fields: [...columns, ...extra],
    data: rows.map((row, i) => {
      const r = results[i];
      return [
        ...columns.map((c) => row[c]),
        num(r.estimate),
        ...(ranged ? [num(r.low), num(r.high)] : []),
        ...(rated ? [r.confidence ?? ''] : []),
        ...(located ? [num(r.locationAdj)] : []),
        ...(compared ? [num(r.compsAdj)] : []),
        r.gap === null ? '' : r.gap.toFixed(2),
        r.gapPct === null ? '' : (r.gapPct * 100).toFixed(2),
        [...r.imputed, ...r.missingLocation].join(' '),
        [r.error, r.actualNote, ...(r.suspect ?? []).map((w) => `left out of gap figures: ${w}`)].filter(Boolean).join('; '),
      ];
    }),
  });
}

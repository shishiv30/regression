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
//    whose values fall inside the training outcome range most often (at
//    least half), the closest to its middle on a tie;
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
  const mid = log ? Math.sqrt(min * max) : (min + max) / 2;
  let best = null;
  for (const c of columns) {
    if (stats[c]?.type !== 'number') continue;
    const vals = rows.map((r) => toNumber(r[c])).filter(Number.isFinite);
    const inside = vals.filter((v) => v >= min && v <= max).length / vals.length;
    const s = stats[c];
    const off = log && s.median > 0 ? Math.abs(Math.log(s.median / mid)) : Math.abs(s.median - mid) / (max - min || 1);
    if (inside >= 0.5 && (!best || inside > best.inside || (inside === best.inside && off < best.off))) best = { c, inside, off };
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

export function predictBatch(model, rows, mapping) {
  const results = rows.map((row, index) => {
    const label = mapping.label && !isMissing(row[mapping.label]) ? String(row[mapping.label]) : `row ${index + 1}`;
    const actual = mapping.actual ? toNumber(row[mapping.actual]) : NaN;
    const has = Number.isFinite(actual);

    // A grouped model uses the formula of the row's group; with no formula
    // for that group the row gets no estimate (it is never guessed).
    let group = null;
    let formula = model;
    if (isGrouped(model)) {
      const col = mapping.inputs[model.groupBy];
      group = col && !isMissing(row[col]) ? String(row[col]).trim() : '';
      formula = resolveModel(model, group);
      if (!formula) {
        return { index, label, group, estimate: null, actual: has ? actual : null, gap: null, gapPct: null, imputed: [], missingLocation: [], locationAdj: null, compsAdj: null, low: null, high: null, confidence: null, warnings: [], error: noGroupReason(model, group) };
      }
    }

    const record = isGrouped(model) ? { [model.groupBy]: group } : {};
    const imputed = [];
    for (const t of formula.terms) {
      const col = mapping.inputs[t.name];
      const raw = col ? row[col] : '';
      const usable = !isMissing(raw) && (t.kind !== 'number' || Number.isFinite(toNumber(raw)));
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
    const p = predict(model, record);
    const amount = (key) => p.contributions.find((c) => c[key])?.contribution ?? null;
    const gap = has ? actual - p.value : null;
    return {
      index,
      label,
      group,
      estimate: p.value,
      actual: has ? actual : null,
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
    };
  });
  return { results, summary: summarizeResults(results) };
}

// Gap statistics for a set of batch results (all rows, or one group's).
export function summarizeResults(results) {
  // Compared = has both an estimate and an actual value.
  const withActual = results.filter((r) => r.gap !== null);
  const n = withActual.length;
  const summary = {
    rows: results.length,
    withActual: n,
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
        r.error ?? '',
      ];
    }),
  });
}

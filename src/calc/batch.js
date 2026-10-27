// Batch prediction: estimate the outcome for every row of a second CSV
// (e.g. sold listings) and, where the row has a real value, measure the gap.
import Papa from 'papaparse';
import { isMissing, toNumber } from './schema.js';
import { predict, isGrouped, resolveModel, noGroupReason, modelInputs } from './model.js';
import { readCoords } from './location.js';

const ACTUAL_RE = /(close|sold|sale|actual|final).*(price|value)|(price|value).*(close|sold|actual)/i;
// Row-label candidates, most specific first.
const LABEL_RES = [/mls/i, /^id$|[_\s.-]id$|Id$/, /address/i, /^(name|title)$/i];

// Default mapping: each model input reads the same-named column; the actual
// value is the outcome's column if present, else a column that looks like it.
export function suggestMapping(model, columns) {
  const inputs = {};
  for (const t of modelInputs(model)) inputs[t.name] = columns.includes(t.name) ? t.name : null;
  const actual = columns.includes(model.outcome.name) ? model.outcome.name : columns.find((c) => ACTUAL_RE.test(c)) ?? null;
  const label = LABEL_RES.map((re) => columns.find((c) => re.test(c))).find(Boolean) ?? null;
  return { inputs, actual, label };
}

// Value used when a row has no usable input: the training mean for numbers
// (range midpoint for models saved before means were recorded), the baseline
// (most common value) for enums.
export function fallbackValue(t) {
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
        return { index, label, group, estimate: null, actual: has ? actual : null, gap: null, gapPct: null, imputed: [], missingLocation: [], locationAdj: null, warnings: [], error: noGroupReason(model, group) };
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
    }
    const p = predict(model, record);
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
      locationAdj: p.location?.adjustment ?? null,
      warnings: p.warnings,
    };
  });
  return { results, summary: summarize(results) };
}

function summarize(results) {
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
  return {
    ...summary,
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
  const extra = [`est_${outcomeName}`, ...(located ? ['location_adj'] : []), 'gap', 'gap_pct', 'imputed_fields', 'note'];
  return Papa.unparse({
    fields: [...columns, ...extra],
    data: rows.map((row, i) => {
      const r = results[i];
      return [
        ...columns.map((c) => row[c]),
        r.estimate === null ? '' : r.estimate.toFixed(2),
        ...(located ? [r.locationAdj === null ? '' : r.locationAdj.toFixed(2)] : []),
        r.gap === null ? '' : r.gap.toFixed(2),
        r.gapPct === null ? '' : (r.gapPct * 100).toFixed(2),
        [...r.imputed, ...r.missingLocation].join(' '),
        r.error ?? '',
      ];
    }),
  });
}

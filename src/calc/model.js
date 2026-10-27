// Encoding records into a numeric design matrix, turning trained (normalised)
// weights back into original units, prediction and evaluation.
//
// Numbers are min-max scaled to [0, 1]; enums are one-hot encoded against a
// baseline (the most frequent value), so each non-baseline value gets a weight
// meaning "outcome shift vs. baseline". The outcome is min-max scaled too.
import { isMissing, toNumber } from './schema.js';
import { locationEffect } from './location.js';

// The group-by field splits the data instead of being an input.
const inputColumns = (stats, config) =>
  Object.keys(config.roles).filter((c) => stats[c] && config.roles[c] === 'input' && c !== config.groupBy);

export function validateConfig(stats, config) {
  const cols = Object.keys(config.roles).filter((c) => stats[c]);
  const outcomes = cols.filter((c) => config.roles[c] === 'outcome');
  const inputs = inputColumns(stats, config);
  if (outcomes.length !== 1) return 'Mark exactly one field as the outcome (y).';
  if (stats[outcomes[0]].type !== 'number') return `Outcome "${outcomes[0]}" must be numeric.`;
  if (config.groupBy) {
    const g = stats[config.groupBy];
    if (!g || g.type !== 'enum') return `Group-by field "${config.groupBy}" must be an enum.`;
    if (config.groupBy === outcomes[0]) return 'The outcome cannot also be the group-by field.';
  }
  if (!inputs.length) return 'Mark at least one field as an input (x).';
  for (const [role, label] of [['lat', 'latitude'], ['lng', 'longitude']]) {
    const marked = cols.filter((c) => config.roles[c] === role);
    if (marked.length > 1) return `Mark only one field as the ${label}.`;
    if (marked.length && stats[marked[0]].type !== 'number') return `${label[0].toUpperCase() + label.slice(1)} "${marked[0]}" must be numeric.`;
  }
  const hasLat = cols.some((c) => config.roles[c] === 'lat'), hasLng = cols.some((c) => config.roles[c] === 'lng');
  if (hasLat !== hasLng) return `Mark a ${hasLat ? 'longitude' : 'latitude'} field too, or neither.`;
  for (const c of inputs) {
    const s = stats[c];
    if (s.type === 'enum' && s.values.length < 2) return `Input "${c}" has only one value; it cannot carry a weight.`;
  }
  return null;
}

export function buildSpec(stats, config) {
  const err = validateConfig(stats, config);
  if (err) throw new Error(err);
  const cols = Object.keys(config.roles).filter((c) => stats[c]);
  const o = stats[cols.find((c) => config.roles[c] === 'outcome')];
  const inputs = inputColumns(stats, config)
    .map((c) => {
      const s = stats[c];
      if (s.type === 'number') {
        return { name: c, kind: 'number', min: s.min, max: s.max, mean: s.mean, scale: s.max - s.min || 1 };
      }
      const [baseline, ...levels] = s.values;
      const freq = Object.fromEntries(s.values.map((v) => [v, s.counts[v] / s.count]));
      return { name: c, kind: 'enum', baseline, levels, values: s.values, freq };
    });
  return { outcome: { name: o.name, min: o.min, max: o.max, scale: o.max - o.min || 1 }, inputs };
}

function encodeInputs(row, spec) {
  const x = [];
  for (const inp of spec.inputs) {
    const raw = row[inp.name];
    if (isMissing(raw)) return null;
    if (inp.kind === 'number') {
      const v = toNumber(raw);
      if (!Number.isFinite(v)) return null;
      x.push((v - inp.min) / inp.scale);
    } else {
      const v = String(raw).trim();
      for (const lvl of inp.levels) x.push(v === lvl ? 1 : 0);
    }
  }
  return x;
}

// Rows with a missing / invalid value in any selected field are dropped.
export function encodeDataset(rows, spec) {
  const X = [], y = [], used = [];
  for (const row of rows) {
    const x = encodeInputs(row, spec);
    const t = toNumber(row[spec.outcome.name]);
    if (!x || !Number.isFinite(t)) continue;
    X.push(x);
    y.push((t - spec.outcome.min) / spec.outcome.scale);
    used.push(row);
  }
  if (used.length < 2) throw new Error('Fewer than 2 complete records for the selected fields.');
  return { X, y, used, dropped: rows.length - used.length };
}

// Convert normalised weights/bias into a plain-JSON model in original units:
//   outcome = intercept + Σ weight·x (numbers) + Σ weights[value] (enums)
export function buildModel(spec, { weights, bias }) {
  const ys = spec.outcome.scale;
  let intercept = spec.outcome.min + ys * bias;
  let k = 0;
  const terms = spec.inputs.map((inp) => {
    if (inp.kind === 'number') {
      const weight = (ys * weights[k++]) / inp.scale;
      intercept -= weight * inp.min;
      return { name: inp.name, kind: 'number', weight, min: inp.min, max: inp.max, mean: inp.mean };
    }
    const w = { [inp.baseline]: 0 };
    for (const lvl of inp.levels) w[lvl] = ys * weights[k++];
    return { name: inp.name, kind: 'enum', baseline: inp.baseline, values: inp.values, weights: w, freq: inp.freq };
  });
  return { outcome: { ...spec.outcome }, intercept, terms };
}

// Effect size in outcome units: how much the term can move the outcome across
// its observed range (numbers) or between its extreme categories (enums).
export function termImportance(term) {
  if (term.kind === 'number') return Math.abs(term.weight) * (term.max - term.min);
  const w = Object.values(term.weights);
  return Math.max(...w) - Math.min(...w);
}

export function termSign(term) {
  return term.kind === 'number' ? Math.sign(term.weight) : 0;
}

// What a term adds for an "average" record: weight·mean for numbers, the
// frequency-weighted category weight for enums.
function averageEffect(t) {
  if (t.kind === 'number') return t.weight * (t.mean ?? 0);
  if (!t.freq) return 0;
  return Object.entries(t.freq).reduce((sum, [v, p]) => sum + p * (t.weights[v] ?? 0), 0);
}

// A grouped model holds one formula per value of its group-by field.
export const isGrouped = (model) => !!model?.groups;

// The formula that applies to a record (null when its group has none).
export function resolveModel(model, groupValue) {
  if (!isGrouped(model)) return model;
  return model.groups[String(groupValue ?? '').trim()] ?? null;
}

export function noGroupReason(model, groupValue) {
  const v = String(groupValue ?? '').trim();
  if (!v) return `${model.groupBy} is required to pick a formula.`;
  const skipped = model.skipped.find((s) => s.value === v);
  return skipped
    ? `No formula for ${model.groupBy} = "${v}": ${skipped.reason}`
    : `No formula for ${model.groupBy} = "${v}" (not in the training data).`;
}

// The latitude/longitude fields of a model (any group's), or null.
export function modelLocation(model) {
  if (!isGrouped(model)) return model.location ?? null;
  return Object.values(model.groups).find((g) => g.location)?.location ?? null;
}

// Every field a model reads: the group-by field first, then the union of all
// groups' inputs (a group may lack some inputs), then latitude and longitude.
export function modelInputs(model) {
  const loc = modelLocation(model);
  const coords = loc ? [{ name: loc.lat, kind: 'number', location: true }, { name: loc.lng, kind: 'number', location: true }] : [];
  if (!isGrouped(model)) return [...model.terms, ...coords];
  const seen = new Map();
  for (const g of Object.values(model.groups)) for (const t of g.terms) if (!seen.has(t.name)) seen.set(t.name, t);
  const values = [...Object.keys(model.groups), ...model.skipped.map((s) => s.value)];
  return [{ name: model.groupBy, kind: 'enum', values, baseline: values[0], group: true }, ...seen.values(), ...coords];
}

// record: { [fieldName]: string | number }
// `reference` is the prediction for an average training record; each
// contribution is how far this record's value moves the result from it, so
// reference + Σ contribution = value.
// With a location adjustment and a record that has coordinates, the last
// contribution is the location's (flagged `location: true`).
// opts.location = false gives the formula alone.
export function predict(model, record, opts = {}) {
  if (isGrouped(model)) {
    const group = String(record[model.groupBy] ?? '').trim();
    const sub = resolveModel(model, group);
    if (!sub) throw new Error(noGroupReason(model, group));
    return { ...predict(sub, record, opts), group };
  }
  let value = model.intercept;
  let reference = model.intercept;
  const contributions = [];
  const warnings = [];
  for (const t of model.terms) {
    const raw = record[t.name];
    if (isMissing(raw)) throw new Error(`"${t.name}" is required.`);
    let c, input;
    if (t.kind === 'number') {
      input = toNumber(raw);
      if (!Number.isFinite(input)) throw new Error(`"${t.name}" must be a number.`);
      if (input < t.min || input > t.max) {
        warnings.push(`${t.name} = ${input} is outside the training range ${t.min} – ${t.max}; the result is an extrapolation.`);
      }
      c = t.weight * input;
    } else {
      input = String(raw).trim();
      if (!(input in t.weights)) {
        warnings.push(`${t.name} = "${input}" was never seen in training; treated as baseline "${t.baseline}".`);
        c = 0;
      } else c = t.weights[input];
    }
    const avg = averageEffect(t);
    value += c;
    reference += avg;
    contributions.push({ name: t.name, input, contribution: c - avg });
  }
  let location = null;
  if (model.location && opts.location !== false) {
    location = locationEffect(model.location, record);
    if (location) {
      value += location.adjustment;
      contributions.push({ name: 'location', input: `${location.lat}, ${location.lng}`, contribution: location.adjustment, location: true });
    } else {
      warnings.push(`No usable ${model.location.lat}/${model.location.lng}, so no location adjustment.`);
    }
  }
  return { value, reference, contributions, warnings, location };
}

export function evaluate(model, rows) {
  const n = rows.length;
  let se = 0, ae = 0, sum = 0, st = 0;
  const actual = rows.map((r) => toNumber(r[model.outcome.name]));
  const mean = actual.reduce((a, b) => a + b, 0) / n;
  rows.forEach((row, i) => {
    // The formula alone: a training record's own residual would leak into
    // its location adjustment. The location gets its own left-out check.
    const err = predict(model, row, { location: false }).value - actual[i];
    se += err * err;
    ae += Math.abs(err);
    st += (actual[i] - mean) ** 2;
    sum += actual[i];
  });
  const mse = se / n;
  return { n, mse, rmse: Math.sqrt(mse), mae: ae / n, r2: st > 0 ? 1 - se / st : NaN, meanOutcome: sum / n };
}

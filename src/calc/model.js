// Encoding records into a numeric design matrix, turning trained (normalised)
// weights back into original units, prediction and evaluation.
//
// Numbers are min-max scaled to [0, 1]; enums are one-hot encoded against a
// baseline (the most frequent value), so each non-baseline value gets a weight
// meaning "outcome shift vs. baseline". The outcome is min-max scaled too.
import { isMissing, toNumber } from './schema.js';

export function validateConfig(stats, config) {
  const cols = Object.keys(config.roles).filter((c) => stats[c]);
  const outcomes = cols.filter((c) => config.roles[c] === 'outcome');
  const inputs = cols.filter((c) => config.roles[c] === 'input');
  if (outcomes.length !== 1) return 'Mark exactly one field as the outcome (y).';
  if (stats[outcomes[0]].type !== 'number') return `Outcome "${outcomes[0]}" must be numeric.`;
  if (!inputs.length) return 'Mark at least one field as an input (x).';
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
  const inputs = cols
    .filter((c) => config.roles[c] === 'input')
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

// record: { [fieldName]: string | number }
// `reference` is the prediction for an average training record; each
// contribution is how far this record's value moves the result from it, so
// reference + Σ contribution = value.
export function predict(model, record) {
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
  return { value, reference, contributions, warnings };
}

export function evaluate(model, rows) {
  const n = rows.length;
  let se = 0, ae = 0, sum = 0, st = 0;
  const actual = rows.map((r) => toNumber(r[model.outcome.name]));
  const mean = actual.reduce((a, b) => a + b, 0) / n;
  rows.forEach((row, i) => {
    const err = predict(model, row).value - actual[i];
    se += err * err;
    ae += Math.abs(err);
    st += (actual[i] - mean) ** 2;
    sum += actual[i];
  });
  const mse = se / n;
  return { n, mse, rmse: Math.sqrt(mse), mae: ae / n, r2: st > 0 ? 1 - se / st : NaN, meanOutcome: sum / n };
}

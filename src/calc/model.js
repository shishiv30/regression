// Encoding records into a numeric design matrix, turning trained (normalised)
// weights back into original units, prediction and evaluation.
//
// Numbers are min-max scaled to [0, 1]; enums are one-hot encoded against a
// baseline (the most frequent value), so each non-baseline value gets a weight
// meaning "outcome shift vs. baseline". The outcome is min-max scaled too.
//
// A blank input drops the record, or with config.missing = 'median' is filled
// with the training median (numbers) or the most common value (enums). A
// record without an outcome is always dropped.
import { isMissing, toNumber } from './schema.js';
import { locationEffect, comparablesEffect } from './location.js';
import { toModel, toValue } from './outcome.js';
import { applyRange, blendEstimates, disagreement, levelOf, MIN_LEVEL_RECORDS } from './confidence.js';

export { toModel, toValue };

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
  if (config.logOutcome === true && !(stats[outcomes[0]].min > 0)) return `Log scale needs every ${outcomes[0]} above 0.`;
  return null;
}

// Log scale: the formula estimates log(outcome), so each weight is a
// percentage change. It suits prices and other values that span a wide range
// and grow by percentages. 'auto' turns it on for an outcome that is always
// positive and spans at least 10×.
export function useLogOutcome(stats, config) {
  const o = stats[Object.keys(config.roles).find((c) => config.roles[c] === 'outcome')];
  if (!o || !(o.min > 0)) return false;
  if (config.logOutcome === 'auto') return o.max / o.min >= 10;
  return config.logOutcome === true;
}

// Log scale for a number input: the formula reads log(x), so doubling x has
// the same effect at any size (diminishing returns per unit: the 1,000 sqft
// between 1,000 and 2,000 add more than those between 5,000 and 6,000).
// Needs every value above 0. config.scales[c] is 'log', 'linear' or unset
// (auto). Auto only acts with the outcome on a log scale, where a plain input
// would make the outcome grow exponentially with it (× the same % for every
// extra unit). It then picks log for an input that spans at least 2× (below
// that log(x) is nearly a straight line, e.g. zip codes) and whose log tracks
// log(outcome) more closely than the input itself does.
export function useLogInput(c, stats, config, rows = []) {
  const s = stats[c];
  if (s?.type !== 'number' || !(s.min > 0)) return false;
  const pick = config.scales?.[c];
  if (pick === 'log') return true;
  if (pick === 'linear' || s.max < 2 * s.min || !rows.length || !useLogOutcome(stats, config)) return false;
  const outcome = Object.keys(config.roles).find((k) => config.roles[k] === 'outcome');
  const xs = [], ys = [];
  for (const r of rows) {
    const x = toNumber(r[c]), y = toNumber(r[outcome]);
    if (x > 0 && y > 0) {
      xs.push(x);
      ys.push(Math.log(y));
    }
  }
  return xs.length > 2 && correlation(xs.map(Math.log), ys) ** 2 > correlation(xs, ys) ** 2;
}

export function correlation(a, b) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

// The value a number term reads: log(x) on a log scale.
const termInput = (t, x) => (t.log ? Math.log(x) : x);

// rows: the training records, for the auto choice of log-scale inputs.
export function buildSpec(stats, config, rows = []) {
  const err = validateConfig(stats, config);
  if (err) throw new Error(err);
  const cols = Object.keys(config.roles).filter((c) => stats[c]);
  const o = stats[cols.find((c) => config.roles[c] === 'outcome')];
  const fill = config.missing === 'median';
  const inputs = inputColumns(stats, config)
    .map((c) => {
      const s = stats[c];
      if (s.type === 'number') {
        // min/max/mean stay in the input's units; lo/scale map what the
        // formula reads (log(x) on a log scale) to [0, 1].
        const log = useLogInput(c, stats, config, rows);
        const lo = log ? Math.log(s.min) : s.min, hi = log ? Math.log(s.max) : s.max;
        const logs = log ? rows.map((r) => toNumber(r[c])).filter((v) => v > 0).map(Math.log) : [];
        return {
          name: c, kind: 'number', min: s.min, max: s.max, mean: s.mean, lo, scale: hi - lo || 1,
          ...(log && { log, logMean: logs.length ? logs.reduce((a, v) => a + v, 0) / logs.length : (lo + hi) / 2 }),
          ...(fill && { fill: s.median }),
        };
      }
      const [baseline, ...levels] = s.values;
      const freq = Object.fromEntries(s.values.map((v) => [v, s.counts[v] / s.count]));
      return { name: c, kind: 'enum', baseline, levels, values: s.values, freq, ...(fill && { fill: baseline }) };
    });
  // min/max stay in outcome units; offset/scale map the training target
  // (log(outcome) on a log scale) to [0, 1].
  const log = useLogOutcome(stats, config);
  const lo = log ? Math.log(o.min) : o.min, hi = log ? Math.log(o.max) : o.max;
  return { outcome: { name: o.name, min: o.min, max: o.max, log, offset: lo, scale: hi - lo || 1 }, inputs };
}

// Null when the record cannot be used; `filled` collects the names of blank
// inputs that took their fill value.
function encodeInputs(row, spec, filled) {
  const x = [];
  for (const inp of spec.inputs) {
    let raw = row[inp.name];
    if (isMissing(raw)) {
      if (inp.fill === undefined) return null;
      raw = inp.fill;
      filled.push(inp.name);
    }
    if (inp.kind === 'number') {
      const v = toNumber(raw);
      if (!Number.isFinite(v) || (inp.log && !(v > 0))) return null;
      x.push((termInput(inp, v) - inp.lo) / inp.scale);
    } else {
      const v = String(raw).trim();
      for (const lvl of inp.levels) x.push(v === lvl ? 1 : 0);
    }
  }
  return x;
}

// Rows with a missing / invalid value in any selected field are dropped,
// unless the spec fills blank inputs. filled = rows that had one filled.
export function encodeDataset(rows, spec) {
  const X = [], y = [], used = [];
  let filled = 0;
  for (const row of rows) {
    const blanks = [];
    const x = encodeInputs(row, spec, blanks);
    const t = toNumber(row[spec.outcome.name]);
    if (!x || !Number.isFinite(t) || (spec.outcome.log && !(t > 0))) continue;
    X.push(x);
    y.push((toModel(spec.outcome, t) - spec.outcome.offset) / spec.outcome.scale);
    used.push(row);
    if (blanks.length) filled++;
  }
  if (used.length < 2) throw new Error('Fewer than 2 complete records for the selected fields.');
  return { X, y, used, dropped: rows.length - used.length, filled };
}

// Convert normalised weights/bias into a plain-JSON model in original units:
//   outcome = intercept + Σ weight·x (numbers) + Σ weights[value] (enums)
// On a log scale the right-hand side is log(outcome), so a weight w means
// "× e^w per unit" (about w·100 % for small w). A log-scale input enters as
// w·log(x): with a log outcome, w is the % change in the outcome per 1 % more
// x (an elasticity).
export function buildModel(spec, { weights, bias }) {
  const ys = spec.outcome.scale;
  let intercept = spec.outcome.offset + ys * bias;
  let k = 0;
  const terms = spec.inputs.map((inp) => {
    if (inp.kind === 'number') {
      const weight = (ys * weights[k++]) / inp.scale;
      intercept -= weight * inp.lo;
      return {
        name: inp.name, kind: 'number', weight, min: inp.min, max: inp.max, mean: inp.mean,
        ...(inp.log && { log: true, logMean: inp.logMean }),
        ...('fill' in inp && { fill: inp.fill }),
      };
    }
    const w = { [inp.baseline]: 0 };
    for (const lvl of inp.levels) w[lvl] = ys * weights[k++];
    return { name: inp.name, kind: 'enum', baseline: inp.baseline, values: inp.values, weights: w, freq: inp.freq, ...('fill' in inp && { fill: inp.fill }) };
  });
  const { name, min, max, log } = spec.outcome;
  return { outcome: { name, min, max, log }, intercept, terms };
}

// Effect size in outcome units: how much the term can move the outcome across
// its observed range (numbers) or between its extreme categories (enums).
export function termImportance(term) {
  if (term.kind === 'number') return Math.abs(term.weight) * (termInput(term, term.max) - termInput(term, term.min));
  const w = Object.values(term.weights);
  return Math.max(...w) - Math.min(...w);
}

export function termSign(term) {
  return term.kind === 'number' ? Math.sign(term.weight) : 0;
}

// What a term adds for an "average" record: weight·mean for numbers, the
// frequency-weighted category weight for enums.
function averageEffect(t) {
  if (t.kind === 'number') return t.weight * (t.log ? t.logMean : t.mean ?? 0);
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
  // A comparables size field that is not also a formula input.
  const sizes = (terms, formulas) => {
    const names = new Set(terms.map((t) => t.name));
    return [...new Set(formulas.map((f) => f.location?.size).filter((s) => s && !names.has(s)))].map((name) => ({ name, kind: 'number', comps: true }));
  };
  if (!isGrouped(model)) return [...model.terms, ...coords, ...sizes(model.terms, [model])];
  const seen = new Map();
  for (const g of Object.values(model.groups)) for (const t of g.terms) if (!seen.has(t.name)) seen.set(t.name, t);
  const values = [...Object.keys(model.groups), ...model.skipped.map((s) => s.value)];
  return [{ name: model.groupBy, kind: 'enum', values, baseline: values[0], group: true }, ...seen.values(), ...coords, ...sizes([...seen.values()], Object.values(model.groups))];
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
  const o = model.outcome;
  let lin = model.intercept; // model space
  let ref = model.intercept;
  const parts = [];
  const warnings = [];
  const filled = [];
  for (const t of model.terms) {
    let raw = record[t.name];
    // A model trained with filled blanks fills them the same way here.
    if (isMissing(raw)) {
      if (t.fill === undefined) throw new Error(`"${t.name}" is required.`);
      raw = t.fill;
      filled.push(t.name);
    }
    let c, input;
    if (t.kind === 'number') {
      input = toNumber(raw);
      if (!Number.isFinite(input)) throw new Error(`"${t.name}" must be a number.`);
      if (t.log && !(input > 0)) throw new Error(`"${t.name}" must be above 0 (it is on a log scale).`);
      if (input < t.min || input > t.max) {
        warnings.push(`${t.name} = ${input} is outside the training range ${t.min} – ${t.max}; the result is an extrapolation.`);
      }
      c = t.weight * termInput(t, input);
    } else {
      input = String(raw).trim();
      if (!(input in t.weights)) {
        warnings.push(`${t.name} = "${input}" was never seen in training; treated as baseline "${t.baseline}".`);
        c = 0;
      } else c = t.weights[input];
    }
    const avg = averageEffect(t);
    lin += c;
    ref += avg;
    parts.push({ name: t.name, input, lin: c - avg });
  }
  let location = null;
  if (model.location && opts.location !== false) {
    location = locationEffect(model.location, record);
    if (location) {
      lin += location.adjustment;
      parts.push({ name: 'location', input: `${location.lat}, ${location.lng}`, lin: location.adjustment, location: true });
    } else {
      warnings.push(`No usable ${model.location.lat}/${model.location.lng}, so no location adjustment.`);
    }
  }

  // Parts in outcome units. On a log scale each part multiplies the running
  // value, so its amount is taken in order; the parts still add up to the
  // value, and `pct` is the part's own percentage effect.
  const reference = toValue(o, ref);
  let acc = ref, prev = reference;
  const contributions = parts.map(({ lin: l, ...p }) => {
    acc += l;
    const cur = toValue(o, acc);
    const c = { ...p, contribution: cur - prev, ...(o.log ? { pct: Math.exp(l) - 1 } : {}) };
    prev = cur;
    return c;
  });
  const formula = toValue(o, lin);
  let value = formula;

  // Comparables: blended in when the record has coordinates and a size.
  let comps = null;
  if (location && model.location.size) {
    comps = comparablesEffect(model.location, record);
    if (comps) {
      value = blendEstimates(formula, comps.value);
      contributions.push({ name: 'comparables', input: `${comps.neighbours.length} similar nearby`, contribution: value - formula, comps: true });
    } else {
      warnings.push(`No usable ${model.location.size}, so no comparables.`);
    }
  }
  return { value, formula, reference, contributions, warnings, filled, location, comps, confidence: confidenceOf(model, value, formula, location, comps) };
}

// The likely range around an estimate and, with comparables, its confidence
// level. Ranges come from left-out errors where there is a location, else
// from the formula's training errors.
function confidenceOf(model, value, formula, location, comps) {
  const loc = model.location;
  let range = model.range ?? null, level = null, source = 'training';
  if (location && loc?.range) {
    range = loc.range;
    source = 'left-out';
    if (comps && loc.confidence) {
      level = levelOf(loc.confidence.thresholds, disagreement(formula, comps.value), comps.spread);
      const own = loc.confidence.levels[level];
      if (own?.n >= MIN_LEVEL_RECORDS) range = own;
    }
  }
  const r = applyRange(range, value);
  return r ? { level, ...r, medianApe: range.medianApe, source } : level ? { level, source } : null;
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

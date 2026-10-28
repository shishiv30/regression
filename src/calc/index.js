// Public API of the calculation layer. The UI only imports from here.
import { buildSpec, encodeDataset, buildModel, evaluate, predict, toModel, toValue, correlation } from './model.js';
import { errorRange, rangeMode } from './confidence.js';
import { analyzeColumns, toNumber } from './schema.js';
import { trainLinear } from './trainer.js';
import { findOutliers, crossFitted, chooseRidge } from './outliers.js';
import { locationFields, buildLocation, combineLocationMetrics, neighbourResiduals, DEFAULT_NEIGHBOURS } from './location.js';

export { parseCsv } from './csv.js';
export { analyzeColumns, suggestRoles, detectCoords } from './schema.js';
export { validateConfig, predict, termImportance, termSign, isGrouped, resolveModel, modelInputs, modelLocation, useLogOutcome, useLogInput } from './model.js';
export { LEVELS } from './confidence.js';
export { DEFAULT_NEIGHBOURS } from './location.js';
export { DEFAULT_OUTLIER_CUTOFF } from './outliers.js';
export { suggestMapping, predictBatch, summarizeResults, resultsToCsv, diagnoseGaps } from './batch.js';
export * as storage from './storage.js';

const MAX_HISTORY_POINTS = 200;
// A group needs at least this many complete records to get its own formula
// (config.minGroupRows overrides it), and at least MIN_ROWS_PER_WEIGHT per
// weight: with fewer, a formula memorises its records and estimates new ones
// no better than a guess. A rejected group's records get no estimate.
export const MIN_GROUP_ROWS = 30;
export const MIN_ROWS_PER_WEIGHT = 2;
// In a group, an input blank for more than this share of records is left out
// of that group's formula (e.g. bedrooms and bathrooms for LAND).
export const MAX_BLANK_SHARE = 0.5;

// Full pipeline: encode → train → convert weights → evaluate.
// With config.groupBy set, one formula is trained per value of that field.
// With latitude/longitude fields marked, each formula also gets a location
// adjustment from its own training records (see location.js).
export async function trainModel(rows, stats, config, opts = {}) {
  // Training row numbers, so a prediction can point at its nearest records.
  const rowIndex = new Map(rows.map((r, i) => [r, i]));
  const model = await (config.groupBy ? trainGrouped(rows, stats, config, { ...opts, rowIndex }) : trainSingle(rows, stats, config, { ...opts, rowIndex }));
  // Fields marked Label, the one that tells records apart best first: they
  // name records in batch results and in the comparables list.
  const share = (s) => (s.type === 'enum' ? s.values.length : s.distinct) / (s.count || 1);
  model.labels = Object.keys(config.roles).filter((c) => config.roles[c] === 'label' && stats[c]).sort((a, b) => share(stats[b]) - share(stats[a]));
  return model;
}

async function trainSingle(rows, stats, config, { onProgress, shouldStop, rowIndex } = {}) {
  const spec = buildSpec(stats, config, rows);
  const encoded = encodeDataset(rows, spec);
  const { dropped, filled } = encoded;
  let { X, y, used } = encoded;

  // Records the formula misses far more than the rest are left out, so they
  // can't drag the weights (or the location residuals) towards themselves.
  // With a location, each record is judged against its own neighbourhood.
  const coords = locationFields(config);
  const local = config.outlierCutoff > 0 && coords && neighbourResiduals(used, coords, config.neighbours ?? DEFAULT_NEIGHBOURS);
  const found = findOutliers(X, y, { cutoff: config.outlierCutoff ?? 0, local });
  const outliers = found.map(({ i, fit, z }) => ({
    row: rowIndex?.get(used[i]) ?? -1,
    actual: toNumber(used[i][spec.outcome.name]),
    fit: toValue(spec.outcome, spec.outcome.offset + fit * spec.outcome.scale),
    z,
  }));
  const holdout = found.map((o) => used[o.i]);
  if (found.length) {
    const out = new Set(found.map((o) => o.i));
    const kept = (_, i) => !out.has(i);
    [X, y, used] = [X.filter(kept), y.filter(kept), used.filter(kept)];
  }
  // Each record's formula estimate from a fit that did not see it (model
  // space), so the likely ranges are not flattered by the formula having
  // memorised its own records.
  // A thin formula can extrapolate wildly without a record, so the estimate
  // is kept within half the training range of it (on a log scale an
  // unbounded one could overflow).
  const clamp = (v) => Math.min(1.5, Math.max(-0.5, Number.isFinite(v) ? v : 0.5));
  // Ridge strength: 'auto' (default) picks it by 5-fold cross-validation,
  // a number fixes it, 0 / null turns it off.
  const ridge = config.ridge === 'auto' || config.ridge === undefined ? chooseRidge(X, y) : Number(config.ridge) || 0;
  const unseen = new Map(crossFitted(X, y, 5, ridge).map((v, i) => [used[i], spec.outcome.offset + clamp(v) * spec.outcome.scale]));

  const result = await trainLinear(X, y, {
    steps: config.steps,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    ridge,
    onProgress,
    shouldStop,
  });
  const model = buildModel(spec, result);

  // Loss history in model units (MSE of the outcome, or of its log on a log
  // scale), down-sampled for storage.
  const s2 = spec.outcome.scale ** 2;
  const every = Math.max(1, Math.ceil(result.history.length / MAX_HISTORY_POINTS));
  const history = [];
  result.history.forEach((l, i) => {
    if (i % every === 0 || i === result.history.length - 1) history.push([i + 1, l * s2]);
  });

  const out = {
    ...model,
    metrics: evaluate(model, used),
    history,
    training: {
      steps: result.history.length,
      requestedSteps: config.steps,
      stopped: result.stopped,
      learningRate: config.learningRate,
      optimizer: config.optimizer,
      ridge,
      rowsUsed: used.length,
      rowsDropped: dropped,
      rowsFilled: filled,
      missing: config.missing === 'median' ? 'median' : 'drop',
      outlierCutoff: config.outlierCutoff > 0 ? config.outlierCutoff : null,
      outliers,
      weights: X[0].length + 1, // incl. intercept; each enum value past the baseline is one weight
      trainedAt: Date.now(),
    },
  };

  // Likely range from the training errors (outliers included: new records
  // will have some too); a location replaces it with one from left-out errors.
  const name = model.outcome.name;
  const formulaOf = (r) => predict(model, r, { location: false }).value;
  out.range = errorRange([
    ...used.map((r) => [toNumber(r[name]), toValue(model.outcome, unseen.get(r))]),
    ...holdout.map((r) => [toNumber(r[name]), formulaOf(r)]),
  ], rangeMode(model.outcome), 'left-out');

  const size = coords ? sizeField(used, model, stats, config.compsBy) : null;
  if (config.outlierCutoff > 0) out.screen = valueScreen(used, model.outcome, size, config.outlierCutoff);

  if (coords) {
    const estimate = (row) => toModel(model.outcome, predict(model, row, { location: false }).value);
    const neighbours = config.neighbours ?? DEFAULT_NEIGHBOURS;
    const { location, skipped } = buildLocation(used, estimate, coords, model.outcome, { neighbours, rowIndex, size, holdout, unseen: (row) => unseen.get(row) });
    if (location) out.location = location;
    else out.locationSkipped = skipped;
  }
  return out;
}

// What a plausible outcome looks like, to screen the actual values of a
// batch: the median and robust spread (1.4826 × MAD) of log(outcome ÷ size)
// over the training records (of log(outcome), or the outcome itself when it
// can be 0 or less, without a size field). An actual value further than
// `cutoff` spreads from it is out of line with every training record of its
// kind, whatever the estimate says.
function valueScreen(rows, outcome, size, cutoff) {
  const vals = [];
  for (const row of rows) {
    const v = toNumber(row[outcome.name]), s = size ? toNumber(row[size]) : 1;
    if (outcome.min > 0 ? v > 0 && s > 0 : Number.isFinite(v)) vals.push(outcome.min > 0 ? Math.log(v / s) : v);
  }
  if (vals.length < 10) return null;
  const mid = median(vals);
  const spread = 1.4826 * median(vals.map((v) => Math.abs(v - mid)));
  return spread > 0 ? { size, log: outcome.min > 0, median: mid, spread, cutoff } : null;
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// The field comparables are matched and priced by (e.g. sqft): the one named,
// or with 'auto' the positive number input whose log tracks the outcome
// most closely (correlation at least 0.3). Null for none.
function sizeField(rows, model, stats, compsBy) {
  if (!compsBy) return null;
  if (compsBy !== 'auto') return stats[compsBy]?.type === 'number' && stats[compsBy].min > 0 ? compsBy : null;
  const y = rows.map((r) => toModel(model.outcome, toNumber(r[model.outcome.name])));
  let best = null;
  for (const t of model.terms) {
    if (t.kind !== 'number' || !(t.min > 0)) continue;
    // Records with a blank (filled) size say nothing about it.
    const ok = rows.map((row) => toNumber(row[t.name]) > 0);
    const r = correlation(rows.filter((_, i) => ok[i]).map((row) => Math.log(toNumber(row[t.name]))), y.filter((_, i) => ok[i]));
    if (r >= 0.3 && (!best || r > best.r)) best = { name: t.name, r };
  }
  return best?.name ?? null;
}

// Fewer than ~2 records per weight lets a formula memorise its training rows:
// its R² looks great but says little about new records.
export const isThin = (model) => model.training.weights && model.training.rowsUsed < 2 * model.training.weights;

// Train one formula per group, each on that group's records only. Ranges,
// averages and enum dictionaries come from the group's own records; inputs
// with nothing to learn in a group are left out of that group's formula:
// mostly blank (e.g. bedrooms for LAND), empty or constant. Groups with too
// few records are skipped.
async function trainGrouped(rows, stats, config, { onProgress, shouldStop, rowIndex } = {}) {
  const g = config.groupBy;
  const columns = Object.keys(stats);
  const values = stats[g].values;
  const groups = {};
  const skipped = [];
  const usedByGroup = {};
  let stopped = false;

  for (const [i, value] of values.entries()) {
    const subset = rows.filter((r) => String(r[g] ?? '').trim() === value);
    if (stopped) {
      skipped.push({ value, rows: subset.length, reason: 'training was stopped before this group' });
      continue;
    }
    const sub = analyzeColumns(columns, subset, config.types);
    const roles = { ...config.roles, [g]: 'ignore' };
    const droppedInputs = [];
    for (const c of columns) {
      if (roles[c] !== 'input') continue;
      const s = sub[c];
      const empty = s.count === 0;
      const sparse = s.missing > MAX_BLANK_SHARE * subset.length;
      const constant = s.type === 'enum' ? s.values.length < 2 : s.min === s.max;
      if (empty || sparse || constant) {
        roles[c] = 'ignore';
        const reason = empty ? 'no values in this group' : sparse ? `${Math.round((100 * s.missing) / subset.length)}% blank in this group` : 'same value for every record';
        droppedInputs.push({ name: c, reason });
      }
    }
    // A group with no usable coordinates simply gets no location adjustment.
    for (const c of columns) if ((roles[c] === 'lat' || roles[c] === 'lng') && sub[c].type !== 'number') roles[c] = 'ignore';
    const cfg = { ...config, roles, groupBy: null };

    let used, weights;
    try {
      const enc = encodeDataset(subset, buildSpec(sub, cfg));
      ({ used } = enc);
      weights = enc.X[0].length + 1;
    } catch (e) {
      skipped.push({ value, rows: subset.length, reason: e.message });
      continue;
    }
    const minRows = config.minGroupRows > 0 ? config.minGroupRows : MIN_GROUP_ROWS;
    if (used.length < minRows) {
      skipped.push({ value, rows: subset.length, reason: `only ${used.length} complete records (needs ${minRows})` });
      continue;
    }
    if (used.length < MIN_ROWS_PER_WEIGHT * weights) {
      skipped.push({ value, rows: subset.length, reason: `only ${used.length} complete records for ${weights} weights (needs ${MIN_ROWS_PER_WEIGHT} per weight); use fewer inputs, e.g. an enum with many values as a number or ignored` });
      continue;
    }

    const model = await trainSingle(subset, sub, cfg, {
      onProgress: (p) => onProgress?.({ ...p, group: value, groupIndex: i, groupCount: values.length }),
      shouldStop,
      rowIndex,
    });
    model.droppedInputs = droppedInputs;
    model.group = value;
    groups[value] = model;
    const out = new Set(model.training.outliers.map((o) => o.row));
    usedByGroup[value] = used.filter((r) => !out.has(rowIndex.get(r)));
    if (model.training.stopped) stopped = true;
  }

  const trained = Object.values(groups);
  if (!trained.length) throw new Error(`No group of "${g}" had enough complete records to train.`);

  const o = stats[trained[0].outcome.name];
  const grouped = { kind: 'grouped', groupBy: g, outcome: { name: o.name, min: o.min, max: o.max }, groups, skipped };
  // Overall fit: every used record predicted by its own group's formula.
  const allUsed = Object.values(usedByGroup).flat();
  grouped.metrics = evaluate(grouped, allUsed);
  // Thin formulas (⚠) are left out of the combined left-out figures: with
  // about as many weights as records, a formula fitted without a record can
  // miss it by any amount, which would swamp every other group.
  const located = trained.filter((m) => m.location && !isThin(m));
  if (located.length) grouped.locationMetrics = combineLocationMetrics(located.map((m) => m.location.metrics));
  grouped.locationMetricsSkipped = trained.filter((m) => m.location && isThin(m)).map((m) => m.group);
  const outliers = trained.flatMap((m) => m.training.outliers.map((o) => ({ ...o, group: m.group })));
  grouped.training = {
    steps: Math.max(...trained.map((m) => m.training.steps)),
    requestedSteps: config.steps,
    stopped,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    rowsUsed: allUsed.length,
    rowsDropped: rows.length - allUsed.length - outliers.length,
    rowsFilled: trained.reduce((a, m) => a + (m.training.rowsFilled ?? 0), 0),
    missing: config.missing === 'median' ? 'median' : 'drop',
    outlierCutoff: config.outlierCutoff > 0 ? config.outlierCutoff : null,
    outliers,
    trainedAt: Date.now(),
  };
  return grouped;
}

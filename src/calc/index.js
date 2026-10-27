// Public API of the calculation layer. The UI only imports from here.
import { buildSpec, encodeDataset, buildModel, evaluate, predict } from './model.js';
import { analyzeColumns } from './schema.js';
import { trainLinear } from './trainer.js';
import { locationFields, buildLocation, combineLocationMetrics, DEFAULT_NEIGHBOURS } from './location.js';

export { parseCsv } from './csv.js';
export { analyzeColumns, suggestRoles } from './schema.js';
export { validateConfig, predict, termImportance, termSign, isGrouped, resolveModel, modelInputs, modelLocation } from './model.js';
export { DEFAULT_NEIGHBOURS } from './location.js';
export { suggestMapping, predictBatch, resultsToCsv } from './batch.js';
export * as storage from './storage.js';

const MAX_HISTORY_POINTS = 200;
// A group needs at least this many complete records to get its own formula.
export const MIN_GROUP_ROWS = 10;

// Full pipeline: encode → train → convert weights → evaluate.
// With config.groupBy set, one formula is trained per value of that field.
// With latitude/longitude fields marked, each formula also gets a location
// adjustment from its own training records (see location.js).
export async function trainModel(rows, stats, config, opts = {}) {
  // Training row numbers, so a prediction can point at its nearest records.
  const rowIndex = new Map(rows.map((r, i) => [r, i]));
  return config.groupBy ? trainGrouped(rows, stats, config, { ...opts, rowIndex }) : trainSingle(rows, stats, config, { ...opts, rowIndex });
}

async function trainSingle(rows, stats, config, { onProgress, shouldStop, rowIndex } = {}) {
  const spec = buildSpec(stats, config);
  const { X, y, used, dropped } = encodeDataset(rows, spec);
  const result = await trainLinear(X, y, {
    steps: config.steps,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    onProgress,
    shouldStop,
  });
  const model = buildModel(spec, result);

  // Loss history in original outcome units (MSE), down-sampled for storage.
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
      rowsUsed: used.length,
      rowsDropped: dropped,
      weights: X[0].length + 1, // incl. intercept; each enum value past the baseline is one weight
      trainedAt: Date.now(),
    },
  };

  const coords = locationFields(config);
  if (coords) {
    const estimate = (row) => predict(model, row, { location: false }).value;
    const neighbours = config.neighbours ?? DEFAULT_NEIGHBOURS;
    const { location, skipped } = buildLocation(used, estimate, coords, model.outcome.name, { neighbours, rowIndex });
    if (location) out.location = location;
    else out.locationSkipped = skipped;
  }
  return out;
}

// Fewer than ~2 records per weight lets a formula memorise its training rows:
// its R² looks great but says little about new records.
export const isThin = (model) => model.training.weights && model.training.rowsUsed < 2 * model.training.weights;

// Train one formula per group, each on that group's records only. Ranges,
// averages and enum dictionaries come from the group's own records; inputs
// with nothing to learn in a group (e.g. no bedrooms for LAND) are left out of
// that group's formula, and groups with too few records are skipped.
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
      const constant = s.type === 'enum' ? s.values.length < 2 : s.min === s.max;
      if (empty || constant) {
        roles[c] = 'ignore';
        droppedInputs.push({ name: c, reason: empty ? 'no values in this group' : 'same value for every record' });
      }
    }
    // A group with no usable coordinates simply gets no location adjustment.
    for (const c of columns) if ((roles[c] === 'lat' || roles[c] === 'lng') && sub[c].type !== 'number') roles[c] = 'ignore';
    const cfg = { ...config, roles, groupBy: null };

    let used;
    try {
      ({ used } = encodeDataset(subset, buildSpec(sub, cfg)));
    } catch (e) {
      skipped.push({ value, rows: subset.length, reason: e.message });
      continue;
    }
    if (used.length < MIN_GROUP_ROWS) {
      skipped.push({ value, rows: subset.length, reason: `only ${used.length} complete records (needs ${MIN_GROUP_ROWS})` });
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
    usedByGroup[value] = used;
    if (model.training.stopped) stopped = true;
  }

  const trained = Object.values(groups);
  if (!trained.length) throw new Error(`No group of "${g}" had enough complete records to train.`);

  const o = stats[trained[0].outcome.name];
  const grouped = { kind: 'grouped', groupBy: g, outcome: { name: o.name, min: o.min, max: o.max }, groups, skipped };
  // Overall fit: every used record predicted by its own group's formula.
  const allUsed = Object.values(usedByGroup).flat();
  grouped.metrics = evaluate(grouped, allUsed);
  const located = trained.filter((m) => m.location);
  if (located.length) grouped.locationMetrics = combineLocationMetrics(located.map((m) => m.location.metrics));
  grouped.training = {
    steps: Math.max(...trained.map((m) => m.training.steps)),
    requestedSteps: config.steps,
    stopped,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    rowsUsed: allUsed.length,
    rowsDropped: rows.length - allUsed.length,
    trainedAt: Date.now(),
  };
  return grouped;
}

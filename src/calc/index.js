// Public API of the calculation layer. The UI only imports from here.
import { buildSpec, encodeDataset, buildModel, evaluate } from './model.js';
import { trainLinear } from './trainer.js';

export { parseCsv } from './csv.js';
export { analyzeColumns, suggestRoles } from './schema.js';
export { validateConfig, predict, termImportance, termSign } from './model.js';
export * as storage from './storage.js';

const MAX_HISTORY_POINTS = 200;

// Full pipeline: encode → train → convert weights → evaluate.
export async function trainModel(rows, stats, config, { onProgress, shouldStop } = {}) {
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

  return {
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
      trainedAt: Date.now(),
    },
  };
}

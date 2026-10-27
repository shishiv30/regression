// localStorage persistence (the client-side "database").
const KEYS = {
  dataset: 'lrlab.dataset',
  config: 'lrlab.config',
  model: 'lrlab.model',
  batch: 'lrlab.batch',
  batchMapping: 'lrlab.batchMapping',
};

function read(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

// Returns false when the browser refuses (quota exceeded, storage disabled).
function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    try { localStorage.removeItem(key); } catch {}
    return false;
  }
}

// Rows are stored as arrays (not objects) to avoid repeating column names.
function writeTable(key, { fileName, columns, rows }) {
  return write(key, { fileName, columns, rows: rows.map((r) => columns.map((c) => r[c])), savedAt: Date.now() });
}

function readTable(key) {
  const d = read(key);
  if (!d?.columns || !Array.isArray(d.rows)) return null;
  const rows = d.rows.map((arr) => Object.fromEntries(d.columns.map((c, i) => [c, arr[i] ?? ''])));
  return { fileName: d.fileName, columns: d.columns, rows };
}

export const saveDataset = (dataset) => writeTable(KEYS.dataset, dataset);
export const loadDataset = () => readTable(KEYS.dataset);
export const saveBatch = (dataset) => writeTable(KEYS.batch, dataset);
export const loadBatch = () => readTable(KEYS.batch);
export const saveBatchMapping = (mapping) => write(KEYS.batchMapping, mapping);
export const loadBatchMapping = () => read(KEYS.batchMapping);

export const saveConfig = (config) => write(KEYS.config, config);
export const loadConfig = () => read(KEYS.config);
export const saveModel = (model) => write(KEYS.model, model);
export const loadModel = () => read(KEYS.model);

export function clearAll() {
  for (const k of Object.values(KEYS)) {
    try { localStorage.removeItem(k); } catch {}
  }
}

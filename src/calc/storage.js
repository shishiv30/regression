// localStorage persistence (the client-side "database").
const KEYS = {
  dataset: 'lrlab.dataset',
  config: 'lrlab.config',
  model: 'lrlab.model',
  batch: 'lrlab.batch',
  batchMapping: 'lrlab.batchMapping',
  fieldSetups: 'lrlab.fieldSetups',
};
const MAX_FIELD_SETUPS = 30;

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

// The field setup (what each column is) remembered per dataset, so loading
// the same file again starts from the last choices. A dataset is its file
// name plus its columns; the most recently used setups are kept.
const datasetKey = ({ fileName, columns }) => JSON.stringify([fileName, columns]);

export function saveFieldSetup(dataset, { roles, types, scales, groupBy }) {
  const key = datasetKey(dataset);
  const all = (read(KEYS.fieldSetups) ?? []).filter((e) => e.key !== key);
  all.unshift({ key, setup: { roles, types, scales, groupBy: groupBy ?? null } });
  return write(KEYS.fieldSetups, all.slice(0, MAX_FIELD_SETUPS));
}

export function loadFieldSetup(dataset) {
  const key = datasetKey(dataset);
  return (read(KEYS.fieldSetups) ?? []).find((e) => e.key === key)?.setup ?? null;
}

export function clearAll() {
  for (const k of Object.values(KEYS)) {
    try { localStorage.removeItem(k); } catch {}
  }
}

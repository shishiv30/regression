// localStorage persistence (the client-side "database").
const KEYS = { dataset: 'lrlab.dataset', config: 'lrlab.config', model: 'lrlab.model' };

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
export function saveDataset({ fileName, columns, rows }) {
  return write(KEYS.dataset, { fileName, columns, rows: rows.map((r) => columns.map((c) => r[c])), savedAt: Date.now() });
}

export function loadDataset() {
  const d = read(KEYS.dataset);
  if (!d?.columns || !Array.isArray(d.rows)) return null;
  const rows = d.rows.map((arr) => Object.fromEntries(d.columns.map((c, i) => [c, arr[i] ?? ''])));
  return { fileName: d.fileName, columns: d.columns, rows };
}

export const saveConfig = (config) => write(KEYS.config, config);
export const loadConfig = () => read(KEYS.config);
export const saveModel = (model) => write(KEYS.model, model);
export const loadModel = () => read(KEYS.model);

export function clearAll() {
  for (const k of Object.values(KEYS)) {
    try { localStorage.removeItem(k); } catch {}
  }
}

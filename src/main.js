import './ui/styles.css';
import { parseCsv, analyzeColumns, suggestRoles, validateConfig, trainModel, predict, storage } from './calc/index.js';
import { renderFields } from './ui/fields.js';
import { renderReport, renderPredictForm, renderPrediction } from './ui/report.js';
import { createGraph } from './ui/graph.js';
import { esc } from './ui/format.js';

const $ = (sel) => document.querySelector(sel);
const DEFAULTS = { roles: {}, types: {}, steps: 500, learningRate: 0.05, optimizer: 'adam' };

const state = {
  dataset: null,
  stats: {},
  config: { ...DEFAULTS },
  model: null,
  training: false,
  stopRequested: false,
};

const graph = createGraph($('#graph'), $('#tooltip'));

// Fingerprint of the field setup, to tell when a stored model is out of date.
const setupKey = (c) => JSON.stringify([c.roles, c.types]);

function setStatus(msg, kind = '') {
  const el = $('#status');
  el.textContent = msg;
  el.className = `status ${kind}`;
}

function persistConfig() {
  storage.saveConfig(state.config);
}

function loadDataset(dataset, { fresh }) {
  state.dataset = dataset;
  if (fresh) state.config = { ...state.config, roles: {}, types: {} };
  state.stats = analyzeColumns(dataset.columns, dataset.rows, state.config.types);
  if (fresh) state.config.roles = suggestRoles(dataset.columns, state.stats);
  renderAll();
}

function renderAll() {
  const { dataset, config, model } = state;
  $('#file-info').innerHTML = dataset
    ? `<b>${esc(dataset.fileName)}</b> · ${dataset.rows.length} records · ${dataset.columns.length} fields`
    : '';
  renderFields($('#fields'), state, onFieldChange);
  $('#steps').value = config.steps;
  $('#lr').value = config.learningRate;
  $('#optimizer').value = config.optimizer;
  renderTrainButton();
  renderReport($('#report'), model, { stale: model && dataset && model.setupKey !== setupKey(config) });
  $('#predict-section').hidden = !model;
  if (model) renderPredictForm($('#predict-form'), model, onPredict);
  renderPrediction($('#prediction'), model, null);
  graph.setModel(model);
}

function renderTrainButton() {
  const btn = $('#train-btn');
  if (state.training) {
    btn.textContent = 'Stop';
    btn.disabled = false;
    return;
  }
  btn.textContent = 'Train';
  const err = state.dataset ? validateConfig(state.stats, state.config) : 'Import a CSV first.';
  btn.disabled = !!err;
  btn.title = err || '';
  if (!state.training) setStatus(err && state.dataset ? err : '', err ? 'error' : '');
}

function onFieldChange(col, key, value) {
  const { config } = state;
  if (key === 'role') {
    // Only one outcome: demote the previous one.
    if (value === 'outcome') {
      for (const c of Object.keys(config.roles)) if (config.roles[c] === 'outcome') config.roles[c] = 'ignore';
    }
    config.roles[col] = value;
  } else {
    config.types[col] = value;
    state.stats = analyzeColumns(state.dataset.columns, state.dataset.rows, config.types);
  }
  persistConfig();
  renderAll();
}

async function onTrain() {
  if (state.training) {
    state.stopRequested = true;
    return;
  }
  const config = state.config;
  config.steps = Math.max(1, Math.min(100000, Math.round(Number($('#steps').value) || DEFAULTS.steps)));
  config.learningRate = Number($('#lr').value) > 0 ? Number($('#lr').value) : DEFAULTS.learningRate;
  config.optimizer = $('#optimizer').value;
  persistConfig();

  state.training = true;
  state.stopRequested = false;
  renderTrainButton();
  const progress = $('#progress');
  progress.hidden = false;
  progress.value = 0;
  setStatus('Training…');
  try {
    const model = await trainModel(state.dataset.rows, state.stats, config, {
      onProgress: ({ step, steps, loss }) => {
        progress.value = step / steps;
        setStatus(`Step ${step} / ${steps} · normalised loss ${loss.toExponential(3)}`);
      },
      shouldStop: () => state.stopRequested,
    });
    model.setupKey = setupKey(config);
    state.model = model;
    const saved = storage.saveModel(model);
    state.training = false;
    renderAll();
    setStatus(`Done — R² ${model.metrics.r2.toFixed(4)}${saved ? '' : ' (model could not be saved to localStorage)'}`, 'ok');
  } catch (e) {
    state.training = false;
    renderTrainButton();
    setStatus(e.message, 'error');
  } finally {
    progress.hidden = true;
  }
}

function onPredict(record) {
  try {
    const result = predict(state.model, record);
    renderPrediction($('#prediction'), state.model, result);
    graph.setPrediction(result);
    $('#prediction').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } catch (e) {
    renderPrediction($('#prediction'), state.model, null, e.message);
    graph.setPrediction(null);
  }
}

async function importText(text, fileName) {
  try {
    const { columns, rows } = parseCsv(text);
    const dataset = { fileName, columns, rows };
    const saved = storage.saveDataset(dataset);
    loadDataset(dataset, { fresh: true });
    persistConfig();
    if (!saved) setStatus('Dataset is too large for localStorage; it will not survive a reload.', 'error');
  } catch (e) {
    setStatus(`Could not read CSV: ${e.message}`, 'error');
  }
}

$('#file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) await importText(await file.text(), file.name);
  e.target.value = '';
});

// Test data collection (public/test-data/, built by scripts/make-test-data.mjs).
const sampleSelect = $('#sample-select');
fetch(`${import.meta.env.BASE_URL}test-data/index.json`)
  .then((res) => (res.ok ? res.json() : []))
  .then((list) => {
    for (const d of list) sampleSelect.add(new Option(`${d.title} · ${d.rows.toLocaleString()} rows`, `test-data/${d.file}`));
  })
  .catch(() => {});

sampleSelect.addEventListener('change', async () => {
  const path = sampleSelect.value;
  if (!path) return;
  setStatus('Loading…');
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await importText(await res.text(), `${path.split('/').pop()} (synthetic)`);
  } catch (e) {
    setStatus(`Could not load ${path}: ${e.message}`, 'error');
  }
  sampleSelect.value = '';
});

$('#clear-btn').addEventListener('click', () => {
  if (!confirm('Remove the saved dataset, field setup and model from this browser?')) return;
  storage.clearAll();
  Object.assign(state, { dataset: null, stats: {}, config: { ...DEFAULTS, roles: {}, types: {} }, model: null });
  renderAll();
});

$('#train-btn').addEventListener('click', onTrain);

// Restore the previous session from localStorage.
const savedConfig = storage.loadConfig();
if (savedConfig) state.config = { ...DEFAULTS, ...savedConfig };
state.model = storage.loadModel();
const savedDataset = storage.loadDataset();
if (savedDataset) loadDataset(savedDataset, { fresh: false });
else renderAll();

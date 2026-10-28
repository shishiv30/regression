import './ui/styles.css';
import {
  parseCsv,
  analyzeColumns,
  suggestRoles,
  validateConfig,
  trainModel,
  predict,
  isGrouped,
  resolveModel,
  suggestMapping,
  predictBatch,
  summarizeResults,
  diagnoseGaps,
  resultsToCsv,
  modelLocation,
  detectCoords,
  DEFAULT_NEIGHBOURS,
  DEFAULT_OUTLIER_CUTOFF,
  MIN_GROUP_ROWS,
  storage,
} from './calc/index.js';
import { renderFields } from './ui/fields.js';
import { renderReport, renderPredictForm, renderPrediction } from './ui/report.js';
import { createGraph } from './ui/graph.js';
import { createGapChart } from './ui/gapChart.js';
import { renderBatch } from './ui/batch.js';
import { createDataTable } from './ui/dataTable.js';
import { createGroupGraphs } from './ui/groupGraphs.js';
import { esc } from './ui/format.js';

const $ = (sel) => document.querySelector(sel);
const DEFAULTS = { roles: {}, types: {}, scales: {}, groupBy: null, steps: 500, learningRate: 0.05, optimizer: 'adam', neighbours: DEFAULT_NEIGHBOURS, outlierCutoff: DEFAULT_OUTLIER_CUTOFF, logOutcome: 'auto', compsBy: 'auto', missing: 'drop', ridge: 'auto', minGroupRows: MIN_GROUP_ROWS };

const state = {
  dataset: null,
  stats: {},
  config: { ...DEFAULTS },
  model: null,
  selectedGroup: null, // which group's formula the report and graph show
  training: false,
  stopRequested: false,
  batch: null, // second CSV whose rows are estimated in bulk
  mapping: null, // { inputs: { term: column|null }, actual, label }
};

const graph = createGraph($('#graph'), $('#tooltip'));
const gapChart = createGapChart({ canvas: $('#gap'), scroller: $('#gap-scroll'), bar: $('#gap-bar'), linkSelect: $('#gap-link'), hint: $('#gap-hint'), tooltip: $('#tooltip') });
const dataTable = createDataTable($('#view-table'));
const groupGraphs = createGroupGraphs($('#group-graphs'), $('#tooltip'), { onGroup: (g) => selectGroup(g) });

// Fingerprint of the field setup, to tell when a stored model is out of date.
// Input scales join it only when set, so models saved before them stay current.
const setupKey = (c) => JSON.stringify([c.roles, c.types, c.groupBy ?? null, ...(Object.keys(c.scales ?? {}).length ? [c.scales] : [])]);

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
  if (fresh) state.config = { ...state.config, roles: {}, types: {}, scales: {}, groupBy: null };
  state.stats = analyzeColumns(dataset.columns, dataset.rows, state.config.types);
  if (fresh) state.config.roles = suggestRoles(dataset.columns, state.stats);
  renderAll();
}

// The single formula on display: for a grouped model, the selected group's.
function shownModel() {
  const { model } = state;
  if (!isGrouped(model)) return model;
  if (!model.groups[state.selectedGroup]) state.selectedGroup = Object.keys(model.groups)[0];
  return resolveModel(model, state.selectedGroup);
}

function selectGroup(group) {
  state.selectedGroup = group;
  renderModelViews();
  renderBatchViews();
}

// Report, prediction form and weight graph for the shown formula.
function renderModelViews() {
  const { dataset, config, model } = state;
  renderReport($('#report'), model, {
    stale: model && dataset && model.setupKey !== setupKey(config),
    shown: shownModel(),
    onGroup: selectGroup,
  });
  $('#predict-section').hidden = !model;
  if (model) renderPredictForm($('#predict-form'), model, onPredict, { group: state.selectedGroup, onGroup: selectGroup });
  renderPrediction($('#prediction'), model, null);
  // A grouped model gets one weight graph per group instead of a single one.
  const grouped = isGrouped(model);
  $('#graph').hidden = grouped;
  $('#group-graphs').hidden = !grouped;
  groupGraphs.setModel(grouped ? model : null);
  if (grouped) {
    shownModel();
    groupGraphs.select(state.selectedGroup);
    groupGraphs.setPrediction(null);
  } else graph.setModel(model);
}

function renderAll() {
  const { dataset, config } = state;
  $('#file-info').innerHTML = dataset
    ? `<b>${esc(dataset.fileName)}</b> · ${dataset.rows.length} records · ${dataset.columns.length} fields`
    : '';
  renderFields($('#fields'), state, onFieldChange);
  $('#steps').value = config.steps;
  $('#lr').value = config.learningRate;
  $('#optimizer').value = config.optimizer;
  $('#ridge').value = config.ridge === 0 ? 'off' : 'auto';
  $('#neighbours').value = config.neighbours;
  $('#min-group').value = config.minGroupRows ?? MIN_GROUP_ROWS;
  $('#min-group-ctl').hidden = !config.groupBy;
  $('#outliers').value = String(config.outlierCutoff);
  $('#missing').value = config.missing === 'median' ? 'median' : 'drop';
  $('#log-outcome').value = config.logOutcome === true ? 'log' : config.logOutcome === false ? 'linear' : 'auto';
  const located = Object.values(config.roles).includes('lat');
  $('#neighbours-ctl').hidden = !located;
  $('#comps-ctl').hidden = !located;
  const sizes = Object.keys(state.stats).filter((c) => state.stats[c].type === 'number' && state.stats[c].min > 0 && ['input', 'ignore', undefined].includes(config.roles[c]));
  $('#comps').innerHTML = ['<option value="">Off</option>', '<option value="auto">Auto</option>', ...sizes.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`)].join('');
  $('#comps').value = config.compsBy && (config.compsBy === 'auto' || sizes.includes(config.compsBy)) ? config.compsBy : config.compsBy ? 'auto' : '';
  renderTrainButton();
  renderModelViews();
  renderBatchSection();
}

function setView(view) {
  for (const b of document.querySelectorAll('.view-tabs [data-view]')) b.setAttribute('aria-selected', String(b.dataset.view === view));
  $('#view-weights').hidden = view !== 'weights';
  $('#view-gap').hidden = view !== 'gap';
  $('#view-table').hidden = view !== 'table';
  $('#tooltip').hidden = true;
}

// Keep the user's column choices across retraining; suggest for new inputs.
function reconcileMapping() {
  const { model, batch, mapping } = state;
  const fresh = suggestMapping(model, batch.columns, batch.rows);
  if (!mapping) return fresh;
  const has = (c) => c === null || batch.columns.includes(c);
  const inputs = {};
  for (const name of Object.keys(fresh.inputs)) inputs[name] = name in mapping.inputs && has(mapping.inputs[name]) ? mapping.inputs[name] : fresh.inputs[name];
  return {
    inputs,
    actual: mapping.actual !== undefined && has(mapping.actual) ? mapping.actual : fresh.actual,
    label: mapping.label !== undefined && has(mapping.label) ? mapping.label : fresh.label,
  };
}

function renderBatchSection() {
  const { model, batch } = state;
  if (!model || !batch) {
    state.batchOutput = null;
    renderBatch($('#batch'), {});
    gapChart.setData(null);
    dataTable.setData(null);
    return;
  }
  state.mapping = reconcileMapping();
  state.batchOutput = predictBatch(model, batch.rows, state.mapping);
  renderBatchViews();
}

// For a grouped model, the gap chart, data table and batch summary show only
// the rows of the selected group; the download keeps every row.
function renderBatchViews() {
  const { model, batch, batchOutput: output } = state;
  if (!model || !batch || !output) return;
  let group = null;
  if (isGrouped(model)) {
    shownModel(); // makes sure a group with a formula is selected
    group = state.selectedGroup;
  }
  const results = group === null ? output.results : output.results.filter((r) => r.group === group);
  const shown = group === null ? output : { results, summary: summarizeResults(results) };
  const scope = group === null ? null : { groupBy: model.groupBy, group, total: output.results.length };
  // The same rows under the model trained before this one, when the batch file is the same.
  const prev = state.previousRun?.fileName === batch.fileName && state.previousRun.results.length === output.results.length ? state.previousRun.results : null;
  const previous = prev ? summarizeResults(group === null ? prev : prev.filter((r) => r.group === group)) : null;
  const diag = state.diagnosis?.output === output && state.diagnosis.group === group ? state.diagnosis.value : undefined;
  renderBatch($('#batch'), { batch, model, mapping: state.mapping, output: shown, scope, previous, diagnosis: diag }, {
    onDiagnose() {
      if (state.training) return;
      setStatus('Diagnosing gaps…');
      // Let the status paint before the (up to ~1 s) work.
      setTimeout(() => {
        state.diagnosis = { output, group, value: diagnoseGaps(model, batch.rows, state.mapping, output.results, group) };
        setStatus('');
        renderBatchViews();
      }, 20);
    },
    onIgnore(cols) {
      for (const c of cols) if (state.config.roles[c] !== undefined) state.config.roles[c] = 'ignore';
      persistConfig();
      renderAll();
      setStatus(`${cols.join(', ')} set to Ignore. Press Train to retrain; the batch figures will show the previous run's for comparison.`);
    },
    onMapping(kind, term, column) {
      if (kind === 'input') state.mapping.inputs[term] = column;
      else state.mapping[kind] = column;
      storage.saveBatchMapping(state.mapping);
      renderBatchSection();
    },
    onDownload() {
      const csv = resultsToCsv(batch.columns, batch.rows, output.results, model.outcome.name);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      a.download = batch.fileName.replace(/\.csv$/i, '').replace(/[^\w.-]+/g, '_') + '-estimates.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    },
  });
  const loc = modelLocation(model);
  gapChart.setData(gapData(batch, results, model, loc));
  dataTable.setData({
    batch,
    results,
    mapping: state.mapping,
    outcomeName: model.outcome.name,
    groupBy: model.groupBy ?? null,
    locationFields: loc ? [loc.lat, loc.lng] : [],
  });
}

// What the gap chart needs besides the results: where each batch row is (the
// model's coordinate columns, else a pair whose values look like
// coordinates), the fields marked Label in training (shown on hover) and a
// column to link records by: the enum input with the most values that still
// average 2+ training records each (an area-like field). Not the group-by:
// the chart shows one group at a time, so it would link every record.
function gapData(batch, results, model, loc) {
  const { columns } = batch;
  const mapped = loc && { lat: state.mapping.inputs[loc.lat], lng: state.mapping.inputs[loc.lng] };
  const found = mapped?.lat && mapped?.lng ? null : detectCoords(columns, analyzeColumns(columns, batch.rows));
  const coords = mapped?.lat && mapped?.lng ? mapped : found?.lat ? found : null;
  const labelFields = (model.labels ?? []).filter((c) => columns.includes(c));
  const formulas = isGrouped(model) ? Object.values(model.groups) : [model];
  const enums = formulas
    .flatMap((f) => f.terms.filter((t) => t.kind === 'enum' && t.values.length * 2 <= f.training.rowsUsed && state.mapping.inputs[t.name]))
    .sort((a, b) => b.values.length - a.values.length);
  const linkBy = (enums[0] && state.mapping.inputs[enums[0].name]) ?? null;
  return { results, rows: batch.rows, columns, outcomeName: model.outcome.name, actualColumn: state.mapping.actual, labelColumn: state.mapping.label, coords, labelFields, linkBy };
}

function importBatchText(text, fileName) {
  try {
    const { columns, rows } = parseCsv(text);
    state.batch = { fileName, columns, rows };
    state.mapping = null;
    const saved = storage.saveBatch(state.batch);
    renderBatchSection();
    storage.saveBatchMapping(state.mapping);
    setView('gap');
    if (!saved) setStatus('Batch file is too large for localStorage; it will not survive a reload.', 'error');
  } catch (e) {
    setStatus(`Could not read CSV: ${e.message}`, 'error');
  }
}

function renderTrainButton() {
  const btn = $('#train-btn');
  const stop = $('#stop-btn');
  // Train is disabled for the whole calculation; Stop is its own button.
  stop.hidden = !state.training;
  stop.disabled = state.stopRequested;
  stop.textContent = state.stopRequested ? 'Stopping…' : 'Stop';
  if (state.training) {
    btn.textContent = 'Training…';
    btn.disabled = true;
    btn.title = 'Training is running';
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
  if (key === 'groupBy') {
    config.groupBy = value ? col : null;
  } else if (key === 'scale') {
    config.scales = { ...config.scales };
    if (value === 'auto') delete config.scales[col];
    else config.scales[col] = value;
  } else {
    // value is a kind from the field dropdown: number / enum (inputs), outcome,
    // lat, lng, label or ignore.
    const role = value === 'number' || value === 'enum' ? 'input' : value;
    // Only one outcome, latitude and longitude: demote the previous one.
    if (role === 'outcome' || role === 'lat' || role === 'lng') {
      for (const c of Object.keys(config.roles)) if (config.roles[c] === role) config.roles[c] = 'ignore';
    }
    config.roles[col] = role;
    // Enum is forced; the numeric uses read the column as a number again.
    const type = value === 'enum' ? 'enum' : ['number', 'outcome', 'lat', 'lng'].includes(value) ? undefined : config.types[col];
    if (type !== config.types[col]) {
      if (type) config.types[col] = type;
      else delete config.types[col];
      state.stats = analyzeColumns(state.dataset.columns, state.dataset.rows, config.types);
    }
    if (config.groupBy === col && (state.stats[col].type !== 'enum' || role === 'outcome')) config.groupBy = null;
  }
  persistConfig();
  renderAll();
}

function onStop() {
  if (!state.training) return;
  state.stopRequested = true;
  renderTrainButton();
}

async function onTrain() {
  if (state.training) return;
  const config = state.config;
  config.steps = Math.max(1, Math.min(100000, Math.round(Number($('#steps').value) || DEFAULTS.steps)));
  config.learningRate = Number($('#lr').value) > 0 ? Number($('#lr').value) : DEFAULTS.learningRate;
  config.optimizer = $('#optimizer').value;
  config.ridge = $('#ridge').value === 'off' ? 0 : 'auto';
  config.neighbours = Math.max(1, Math.min(100, Math.round(Number($('#neighbours').value) || DEFAULTS.neighbours)));
  config.minGroupRows = Math.max(2, Math.round(Number($('#min-group').value) || MIN_GROUP_ROWS));
  config.outlierCutoff = Number($('#outliers').value) || 0;
  config.logOutcome = { auto: 'auto', linear: false, log: true }[$('#log-outcome').value];
  config.compsBy = $('#comps').value || null;
  config.missing = $('#missing').value;
  persistConfig();

  state.training = true;
  state.stopRequested = false;
  renderTrainButton();
  const progress = $('#progress');
  progress.hidden = false;
  progress.value = 0;
  setStatus('Training…');
  // Let the disabled button paint before the first stretch of work.
  await new Promise((r) => setTimeout(r, 0));
  try {
    // A snapshot: field changes made while training apply to the next run.
    const trained = structuredClone(config);
    const model = await trainModel(state.dataset.rows, state.stats, trained, {
      onProgress: ({ step, steps, loss, group, groupIndex, groupCount }) => {
        const done = group === undefined ? step / steps : (groupIndex + step / steps) / groupCount;
        progress.value = done;
        const where = group === undefined ? '' : `${config.groupBy} = ${group} (${groupIndex + 1}/${groupCount}) · `;
        setStatus(`${where}Step ${step} / ${steps} · normalised loss ${loss.toExponential(3)}`);
      },
      shouldStop: () => state.stopRequested,
    });
    model.setupKey = setupKey(trained);
    // Keep the old model's batch results to compare the new one with.
    if (state.batch && state.batchOutput) state.previousRun = { fileName: state.batch.fileName, results: state.batchOutput.results };
    state.model = model;
    const saved = storage.saveModel(model);
    // Batch estimates are recomputed here, still under the disabled button.
    renderAll();
    state.training = false;
    renderTrainButton();
    const groups = isGrouped(model) ? ` · ${Object.keys(model.groups).length} formulas${model.skipped.length ? `, ${model.skipped.length} group(s) skipped` : ''}` : '';
    setStatus(`Done — R² ${model.metrics.r2.toFixed(4)}${groups}${saved ? '' : ' (model could not be saved to localStorage)'}`, 'ok');
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
    renderPrediction($('#prediction'), state.model, result, null, state.dataset);
    if (isGrouped(state.model)) groupGraphs.setPrediction(result);
    else graph.setPrediction(result);
    setView('weights');
    $('#prediction').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } catch (e) {
    renderPrediction($('#prediction'), state.model, null, e.message);
    graph.setPrediction(null);
    groupGraphs.setPrediction(null);
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

$('#batch-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) importBatchText(await file.text(), file.name);
  e.target.value = '';
});

for (const b of document.querySelectorAll('.view-tabs [data-view]')) b.addEventListener('click', () => setView(b.dataset.view));

// Test data collection (public/test-data/, built by scripts/make-test-data.mjs),
// offered both as training data and as batch files.
fetch(`${import.meta.env.BASE_URL}test-data/index.json`)
  .then((res) => (res.ok ? res.json() : []))
  .then((list) => {
    for (const d of list) {
      const label = `${d.title} · ${d.rows.toLocaleString()} rows`;
      $('#sample-select').add(new Option(label, `test-data/${d.file}`));
      $('#batch-sample').add(new Option(label, `test-data/${d.file}`));
    }
  })
  .catch(() => {});

function onSampleChange(select, load) {
  select.addEventListener('change', async () => {
    const path = select.value;
    if (!path) return;
    setStatus('Loading…');
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await load(await res.text(), path.split('/').pop());
      if ($('#status').textContent === 'Loading…') setStatus('');
    } catch (e) {
      setStatus(`Could not load ${path}: ${e.message}`, 'error');
    }
    select.value = '';
  });
}
onSampleChange($('#sample-select'), importText);
onSampleChange($('#batch-sample'), importBatchText);

$('#clear-btn').addEventListener('click', () => {
  if (!confirm('Remove the saved dataset, field setup and model from this browser?')) return;
  storage.clearAll();
  Object.assign(state, { dataset: null, stats: {}, config: { ...DEFAULTS, roles: {}, types: {}, scales: {} }, model: null, batch: null, mapping: null });
  setView('weights');
  renderAll();
});

$('#train-btn').addEventListener('click', onTrain);
$('#stop-btn').addEventListener('click', onStop);

// Restore the previous session from localStorage.
const savedConfig = storage.loadConfig();
if (savedConfig) state.config = { ...DEFAULTS, ...savedConfig };
state.model = storage.loadModel();
state.batch = storage.loadBatch();
state.mapping = storage.loadBatchMapping();
const savedDataset = storage.loadDataset();
if (savedDataset) loadDataset(savedDataset, { fresh: false });
else renderAll();

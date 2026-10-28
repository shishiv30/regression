// Outcome report (left) and the prediction form (right of the graph).
import { termImportance, isThin } from '../calc/index.js';
import { fmt, signed, esc } from './format.js';

export function renderReport(el, model, { stale, shown, onGroup } = {}) {
  if (!model) {
    el.innerHTML = '<p class="muted">No model yet. Mark one numeric field as the outcome, pick inputs, then press <b>Train</b>.</p>';
    return;
  }
  const tr = model.training;
  const grouped = !!model.groups;
  const header = `
    ${stale ? '<div class="notice">Field setup changed since this model was trained — press Train to update.</div>' : ''}
    <div class="kv">
      <span>Outcome</span><b>${esc(model.outcome.name)}</b>
      ${grouped ? `<span>Grouped by</span><b>${esc(model.groupBy)} · ${Object.keys(model.groups).length} formulas</b>` : ''}
      <span>Records used</span><b>${tr.rowsUsed}${tr.rowsDropped ? ` <span class="warn">(${tr.rowsDropped} dropped: missing values${grouped ? ' or skipped group' : ''})</span>` : ''}${tr.outliers?.length ? ` <span class="warn">(${tr.outliers.length} outliers)</span>` : ''}</b>
      <span>Steps</span><b>${tr.steps}${tr.stopped ? ` of ${tr.requestedSteps} (stopped)` : ''} · ${tr.optimizer} · lr ${tr.learningRate}</b>
      <span>Trained</span><b>${new Date(tr.trainedAt).toLocaleString()}</b>
    </div>
    ${outliersHtml(tr, grouped)}
    ${grouped ? '<h3>All groups together</h3>' : ''}
    ${metricsHtml(model.metrics)}
    ${grouped && model.locationMetrics ? locationFitHtml(model.locationMetrics) : ''}`;

  el.innerHTML = grouped ? header + groupsHtml(model, shown) + detailHtml(shown, { group: true }) : header + detailHtml(model);

  el.querySelectorAll('[data-group]').forEach((row) => row.addEventListener('click', () => onGroup?.(row.dataset.group)));
  el.querySelector('#report-group')?.addEventListener('change', (e) => onGroup?.(e.target.value));
  const loss = el.querySelector('canvas.loss');
  if (loss) new ResizeObserver(() => drawLoss(loss, shown.history)).observe(loss);
}

// Records left out of training because the formula missed them by far more
// than it misses a typical record (z = robust standard deviations of error).
function outliersHtml(tr, grouped) {
  if (!tr.outliers?.length) return '';
  const rows = tr.outliers
    .slice(0, 200)
    .map((o) => `<tr><td class="num">${o.row + 1}</td>${grouped ? `<td>${esc(o.group)}</td>` : ''}<td class="num">${fmt(o.actual)}</td><td class="num">${fmt(o.fit)}</td><td class="num">${signed(o.z)}σ</td></tr>`)
    .join('');
  const more = tr.outliers.length > 200 ? `<p class="muted small">…${tr.outliers.length - 200} more</p>` : '';
  return `
    <details class="outliers">
      <summary>${tr.outliers.length} outliers left out (more than ${tr.outlierCutoff}σ off)</summary>
      <p class="muted small">Each record's error against a first fit, measured in robust standard deviations (from the median error). Records this far off are usually bad data, e.g. a wrong price or size.</p>
      <table class="weights">
        <thead><tr><th>Row</th>${grouped ? '<th>Group</th>' : ''}<th>Actual</th><th>Fit</th><th>Off by</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>${more}
    </details>`;
}

// Formula alone vs. formula + location, each record left out of its own
// neighbours so the gain is not flattered by the record itself.
const locationFitHtml = (m) => `
  <div class="kv small location-fit">
    <span>R² with location</span><b>${fmt(m.r2Before)} → <span class="${m.r2After >= m.r2Before ? 'pos' : 'neg'}">${fmt(m.r2After)}</span></b>
    <span>RMSE with location</span><b>${fmt(m.rmseBefore)} → ${fmt(m.rmseAfter)}</b>
  </div>`;

const metricsHtml = (m) => `
  <div class="metrics">
    <div><span>R²</span><b>${fmt(m.r2)}</b></div>
    <div><span>RMSE</span><b>${fmt(m.rmse)}</b></div>
    <div><span>MAE</span><b>${fmt(m.mae)}</b></div>
  </div>`;

// One row per group: how many records trained it and how well it fits.
function groupsHtml(model, shown) {
  const rows = Object.entries(model.groups)
    .map(
      ([v, g]) => `<tr data-group="${esc(v)}" class="${g === shown ? 'selected' : ''}">
        <td>${esc(v)}${isThin(g) ? ' <span class="warn" title="Too few records for its weights">⚠</span>' : ''}</td><td class="num">${g.training.rowsUsed}</td><td class="num">${fmt(g.metrics.r2)}</td><td class="num">${fmt(g.metrics.rmse)}</td></tr>`,
    )
    .join('');
  const skipped = model.skipped
    .map((s) => `<tr><td class="muted">${esc(s.value)}</td><td class="num muted">${s.rows}</td><td colspan="2" class="warn small">${esc(s.reason)}</td></tr>`)
    .join('');
  const options = Object.keys(model.groups)
    .map((v) => `<option value="${esc(v)}" ${model.groups[v] === shown ? 'selected' : ''}>${esc(v)}</option>`)
    .join('');
  return `
    <h3>Formulas by ${esc(model.groupBy)}</h3>
    <table class="weights groups">
      <thead><tr><th>Group</th><th class="num">Records</th><th class="num">R²</th><th class="num">RMSE</th></tr></thead>
      <tbody>${rows}${skipped}</tbody>
    </table>
    <label class="group-pick" for="report-group">Showing formula for
      <select id="report-group">${options}</select>
    </label>`;
}

// Loss curve, weights table and equation of one formula.
function detailHtml(model, { group = false } = {}) {
  const total = model.terms.reduce((a, t) => a + termImportance(t), 0) + (model.location?.spread ?? 0) || 1;
  const weightRows = model.terms
    .map((t) => {
      const imp = termImportance(t);
      const share = `<div class="bar"><span style="width:${((imp / total) * 100).toFixed(1)}%"></span></div>`;
      if (t.kind === 'number') {
        return `<tr><td>${esc(t.name)}</td><td class="num ${t.weight >= 0 ? 'pos' : 'neg'}">${fmt(t.weight)}</td><td>${share}</td></tr>`;
      }
      const cats = Object.entries(t.weights)
        .filter(([k]) => k !== t.baseline)
        .map(([k, w]) => `<tr class="sub"><td>${esc(k)}</td><td class="num ${w >= 0 ? 'pos' : 'neg'}">${signed(w)}</td><td></td></tr>`)
        .join('');
      return `<tr><td>${esc(t.name)} <span class="muted">vs ${esc(t.baseline)}</span></td><td class="num muted">enum</td><td>${share}</td></tr>${cats}`;
    })
    .join('');

  const eq = [`${fmt(model.intercept)}`]
    .concat(model.terms.map((t) => (t.kind === 'number' ? `${signed(t.weight)}·${esc(t.name)}` : `+ w[${esc(t.name)}]`)))
    .concat(model.location ? ['+ location'] : [])
    .join(' ');
  const dropped = model.droppedInputs?.length
    ? `<div class="notice">Not used in this group: ${model.droppedInputs.map((d) => `${esc(d.name)} (${esc(d.reason)})`).join(', ')}</div>`
    : '';

  const thin = isThin(model)
    ? `<div class="notice">⚠ Only ${model.training.rowsUsed} records for ${model.training.weights} weights. This formula can memorise its training records, so its R² overstates how well it estimates new ones. Use fewer inputs (e.g. zipCode as a number or ignored) or more records.</div>`
    : '';
  return `
    ${group ? metricsHtml(model.metrics) : ''}
    ${thin}
    ${dropped}
    <h3>Loss (MSE) per step</h3>
    <canvas class="loss" height="90"></canvas>
    <h3>Weights</h3>
    <table class="weights">
      <thead><tr><th>Field</th><th class="num">Weight</th><th>Effect share</th></tr></thead>
      <tbody>
        <tr><td>intercept</td><td class="num">${fmt(model.intercept)}</td><td></td></tr>
        ${weightRows}
        ${model.location ? `<tr><td>location <span class="muted">nearby records</span></td><td class="num muted">±${fmt(model.location.spread / 2)}</td><td><div class="bar"><span class="loc" style="width:${((model.location.spread / total) * 100).toFixed(1)}%"></span></div></td></tr>` : ''}
      </tbody>
    </table>
    <p class="equation"><span class="muted">${esc(model.outcome.name)}${group ? ` [${esc(model.group)}]` : ''} =</span> ${eq}</p>
    ${locationHtml(model)}`;
}

function locationHtml(model) {
  if (model.locationSkipped) return `<div class="notice">No location adjustment: ${esc(model.locationSkipped)}.</div>`;
  const loc = model.location;
  if (!loc) return '';
  const m = loc.metrics;
  const helps = m.r2After > m.r2Before;
  return `
    <h3>Location adjustment</h3>
    <p class="small">Each estimate is shifted by how much the formula missed on the <b>${loc.neighbours}</b> nearest training records (by ${esc(loc.lat)}/${esc(loc.lng)}), weighted by distance.</p>
    <div class="kv small">
      <span>Records with coordinates</span><b>${loc.points.length}</b>
      <span title="Picked in training. Records this far away count about 60% as much as one next door.">Distance scale</span><b>${fmt(loc.scaleKm)} km <span class="muted">(picked in training)</span></b>
      <span>Typical shift</span><b>±${fmt(loc.spread / 2)}</b>
    </div>
    ${locationFitHtml(m)}
    ${helps ? '' : '<div class="notice">Location does not improve the estimates for these records; nearby records miss in unrelated ways.</div>'}
    <p class="muted small">R² and RMSE: formula alone → formula + location, with each record left out of its own neighbours.</p>`;
}

function drawLoss(canvas, history) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  if (history.length < 2) return;
  const s = getComputedStyle(document.documentElement);
  // Log scale: loss usually drops by orders of magnitude.
  const ys = history.map(([, l]) => Math.log10(Math.max(l, 1e-12)));
  const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
  const maxStep = history[history.length - 1][0];
  ctx.strokeStyle = s.getPropertyValue('--accent').trim();
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach(([step], i) => {
    const x = 4 + ((step - 1) / Math.max(maxStep - 1, 1)) * (w - 8);
    const y = 6 + (1 - (ys[i] - lo) / span) * (h - 22);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = s.getPropertyValue('--muted').trim();
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'left';
  ctx.fillText(`start ${fmt(history[0][1])}`, 4, h);
  ctx.textAlign = 'right';
  ctx.fillText(`final ${fmt(history[history.length - 1][1])}`, w - 4, h);
}

export function renderPredictForm(form, model, onSubmit, { group, onGroup } = {}) {
  // Keep what was typed when the form is rebuilt (e.g. on a group change).
  const typed = Object.fromEntries([...form.querySelectorAll('[name]')].map((el) => [el.name, el.value]));
  const formula = model.groups ? model.groups[group] : model;
  const loc = formula.location;
  const groupField = model.groups
    ? `<label for="p-group">${esc(model.groupBy)}</label>
       <select id="p-group" name="${esc(model.groupBy)}">${Object.keys(model.groups)
         .map((v) => `<option ${v === group ? 'selected' : ''}>${esc(v)}</option>`)
         .join('')}</select>`
    : '';
  form.innerHTML =
    groupField +
    formula.terms
      .map((t, i) => {
        const id = `p-${i}`;
        const control =
          t.kind === 'number'
            ? `<input id="${id}" name="${esc(t.name)}" type="number" step="any" required placeholder="${fmt(t.min)} – ${fmt(t.max)}" />`
            : `<select id="${id}" name="${esc(t.name)}" required>${t.values.map((v) => `<option>${esc(v)}</option>`).join('')}</select>`;
        return `<label for="${id}">${esc(t.name)}</label>${control}`;
      })
      .join('') +
    (loc
      ? [loc.lat, loc.lng]
          .map((f, i) => `<label for="p-loc-${i}">${esc(f)} <span class="muted">(optional)</span></label><input id="p-loc-${i}" name="${esc(f)}" type="number" step="any" placeholder="${i ? 'longitude' : 'latitude'}" />`)
          .join('')
      : '') +
    '<button class="btn primary" type="submit">Predict</button>';
  formula.terms.forEach((t, i) => {
    const el = form.querySelector(`#p-${i}`);
    if (typed[t.name] && (t.kind === 'number' || t.values.includes(typed[t.name]))) el.value = typed[t.name];
  });
  if (loc) [loc.lat, loc.lng].forEach((f, i) => (form.querySelector(`#p-loc-${i}`).value = typed[f] ?? ''));
  form.querySelector('#p-group')?.addEventListener('change', (e) => onGroup?.(e.target.value));
  form.onsubmit = (e) => {
    e.preventDefault();
    const record = model.groups ? { [model.groupBy]: group } : {};
    formula.terms.forEach((t, i) => (record[t.name] = form.querySelector(`#p-${i}`).value));
    if (loc) [loc.lat, loc.lng].forEach((f, i) => (record[f] = form.querySelector(`#p-loc-${i}`).value));
    onSubmit(record);
  };
}

export function renderPrediction(el, model, result, error, dataset) {
  if (error) {
    el.innerHTML = `<div class="field-error">${esc(error)}</div>`;
    return;
  }
  if (!result) {
    el.innerHTML = '';
    return;
  }
  const rows = result.contributions
    .map((c) => `<tr><td>${esc(c.name)} <span class="muted">${c.location ? `from ${result.location.neighbours.length} nearby records` : `= ${esc(typeof c.input === 'number' ? fmt(c.input) : c.input)}`}</span></td><td class="num ${c.contribution >= 0 ? 'pos' : 'neg'}">${signed(c.contribution)}</td></tr>`)
    .join('');
  el.innerHTML = `
    <div class="result">
      <span class="muted">Predicted ${esc(model.outcome.name)}${result.group ? ` · ${esc(model.groupBy)} = ${esc(result.group)} formula` : ''}</span>
      <b>${fmt(result.value)}</b>
    </div>
    <table class="weights">
      <tr><td>average record <span class="muted">(starting point)</span></td><td class="num">${fmt(result.reference)}</td></tr>
      ${rows}
    </table>
    <p class="muted small">Each row is how that value moves the prediction compared with an average record.</p>
    ${result.location ? neighboursHtml(result.location, model.outcome.name, dataset) : ''}
    ${result.warnings.map((w) => `<div class="notice">${esc(w)}</div>`).join('')}`;
}

// The training records behind the location adjustment: how far away they are
// and how much the formula missed on each.
function neighboursHtml(loc, outcome, dataset) {
  const labelOf = (row) => {
    const r = dataset?.rows[row];
    const key = r && dataset.columns.find((c) => /mls|address|^id$/i.test(c));
    return key && r[key] ? esc(r[key]) : row >= 0 ? `#${row + 1}` : '—';
  };
  const rows = loc.neighbours
    .map((n) => `<tr><td>${labelOf(n.row)}</td><td class="num">${n.distanceKm < 1 ? `${Math.round(n.distanceKm * 1000)} m` : `${fmt(n.distanceKm)} km`}</td><td class="num">${fmt(n.actual)}</td><td class="num ${n.residual >= 0 ? 'pos' : 'neg'}">${signed(n.residual)}</td><td class="num">${Math.round(n.share * 100)}%</td></tr>`)
    .join('');
  return `
    <h3>Nearest training records</h3>
    <table class="weights neighbours">
      <thead><tr><th>Record</th><th class="num">Away</th><th class="num">${esc(outcome)}</th><th class="num" title="Actual minus the formula's estimate">Formula missed</th><th class="num" title="Share of the location adjustment">Weight</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="muted small">Location adjustment = weighted average of "formula missed", shrunk towards 0 when the records are few or far away.</p>`;
}

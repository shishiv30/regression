// Left panel: training report, weights, loss curve and the prediction form.
import { termImportance } from '../calc/index.js';
import { fmt, signed, esc } from './format.js';

export function renderReport(el, model, { stale }) {
  if (!model) {
    el.innerHTML = '<p class="muted">No model yet. Mark one numeric field as the outcome, pick inputs, then press <b>Train</b>.</p>';
    return;
  }
  const m = model.metrics;
  const tr = model.training;
  const total = model.terms.reduce((a, t) => a + termImportance(t), 0) || 1;

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
    .join(' ');

  el.innerHTML = `
    ${stale ? '<div class="notice">Field setup changed since this model was trained — press Train to update.</div>' : ''}
    <div class="kv">
      <span>Outcome</span><b>${esc(model.outcome.name)}</b>
      <span>Records used</span><b>${tr.rowsUsed}${tr.rowsDropped ? ` <span class="warn">(${tr.rowsDropped} dropped: missing values)</span>` : ''}</b>
      <span>Steps</span><b>${tr.steps}${tr.stopped ? ` of ${tr.requestedSteps} (stopped)` : ''} · ${tr.optimizer} · lr ${tr.learningRate}</b>
      <span>Trained</span><b>${new Date(tr.trainedAt).toLocaleString()}</b>
    </div>
    <div class="metrics">
      <div><span>R²</span><b>${fmt(m.r2)}</b></div>
      <div><span>RMSE</span><b>${fmt(m.rmse)}</b></div>
      <div><span>MAE</span><b>${fmt(m.mae)}</b></div>
    </div>
    <h3>Loss (MSE) per step</h3>
    <canvas class="loss" height="90"></canvas>
    <h3>Weights</h3>
    <table class="weights">
      <thead><tr><th>Field</th><th class="num">Weight</th><th>Effect share</th></tr></thead>
      <tbody>
        <tr><td>intercept</td><td class="num">${fmt(model.intercept)}</td><td></td></tr>
        ${weightRows}
      </tbody>
    </table>
    <p class="equation"><span class="muted">${esc(model.outcome.name)} =</span> ${eq}</p>`;

  const loss = el.querySelector('canvas.loss');
  new ResizeObserver(() => drawLoss(loss, model.history)).observe(loss);
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

export function renderPredictForm(form, model, onSubmit) {
  form.innerHTML =
    model.terms
      .map((t, i) => {
        const id = `p-${i}`;
        const control =
          t.kind === 'number'
            ? `<input id="${id}" name="${esc(t.name)}" type="number" step="any" required placeholder="${fmt(t.min)} – ${fmt(t.max)}" />`
            : `<select id="${id}" name="${esc(t.name)}" required>${t.values.map((v) => `<option>${esc(v)}</option>`).join('')}</select>`;
        return `<label for="${id}">${esc(t.name)}</label>${control}`;
      })
      .join('') + '<button class="btn primary" type="submit">Predict</button>';
  form.onsubmit = (e) => {
    e.preventDefault();
    const record = {};
    model.terms.forEach((t, i) => (record[t.name] = form.querySelector(`#p-${i}`).value));
    onSubmit(record);
  };
}

export function renderPrediction(el, model, result, error) {
  if (error) {
    el.innerHTML = `<div class="field-error">${esc(error)}</div>`;
    return;
  }
  if (!result) {
    el.innerHTML = '';
    return;
  }
  const rows = result.contributions
    .map((c) => `<tr><td>${esc(c.name)} <span class="muted">= ${esc(typeof c.input === 'number' ? fmt(c.input) : c.input)}</span></td><td class="num ${c.contribution >= 0 ? 'pos' : 'neg'}">${signed(c.contribution)}</td></tr>`)
    .join('');
  el.innerHTML = `
    <div class="result">
      <span class="muted">Predicted ${esc(model.outcome.name)}</span>
      <b>${fmt(result.value)}</b>
    </div>
    <table class="weights">
      <tr><td>average record <span class="muted">(starting point)</span></td><td class="num">${fmt(result.reference)}</td></tr>
      ${rows}
    </table>
    <p class="muted small">Each row is how that value moves the prediction compared with an average record.</p>
    ${result.warnings.map((w) => `<div class="notice">${esc(w)}</div>`).join('')}`;
}

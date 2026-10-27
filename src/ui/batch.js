// Right panel, batch section: column mapping, gap summary and results table.
import { modelInputs } from '../calc/index.js';
import { fmt, signed, esc, missingAlert } from './format.js';

const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);
const signedPct = (v) => (v === null ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}%`);

export function renderBatch(el, { batch, model, mapping, output }, { onMapping, onDownload } = {}) {
  if (!batch) {
    el.innerHTML = '';
    return;
  }
  const { columns, rows, fileName } = batch;
  const opts = (selected, empty) =>
    `<option value="">${esc(empty)}</option>` +
    columns.map((c) => `<option value="${esc(c)}"${c === selected ? ' selected' : ''}>${esc(c)}</option>`).join('');
  const inputs = modelInputs(model);
  const unmapped = inputs.filter((t) => !t.group && !t.location && !mapping.inputs[t.name]);
  const groupInput = inputs.find((t) => t.group);
  const s = output.summary;

  el.innerHTML = `
    <p class="file-line"><b>${esc(fileName)}</b> · ${rows.length} records</p>

    ${groupInput && !mapping.inputs[groupInput.name] ? `<div class="notice">Pick the column that holds <b>${esc(groupInput.name)}</b>; each row is estimated with its group's formula.</div>` : ''}
    <details class="mapping" ${unmapped.length || !mapping.actual || (groupInput && !mapping.inputs[groupInput.name]) ? 'open' : ''}>
      <summary>Column matching${unmapped.length ? ` · <span class="warn">${unmapped.length} input${unmapped.length > 1 ? 's' : ''} use the training average</span>` : ''}</summary>
      <div class="map-grid">
        <span class="map-head">Model input</span><span class="map-head">Read from column</span>
        ${inputs
          .map(
            (t) => `<label for="map-${esc(t.name)}" class="${mapping.inputs[t.name] ? '' : 'warn'}">${t.group ? `<b>${esc(t.name)}</b> <span class="muted">(group)</span>` : t.location ? `${esc(t.name)} <span class="muted">(location)</span>` : esc(t.name)}</label>
              <select id="map-${esc(t.name)}" data-kind="input" data-term="${esc(t.name)}">${opts(mapping.inputs[t.name], t.group ? '— none: no estimates —' : t.location ? '— none: no location adjustment —' : '— training average —')}</select>`,
          )
          .join('')}
        <label for="map-actual"><b>Actual value</b></label>
        <select id="map-actual" data-kind="actual">${opts(mapping.actual, '— none —')}</select>
        <label for="map-label">Row label</label>
        <select id="map-label" data-kind="label">${opts(mapping.label, '— row number —')}</select>
      </div>
    </details>

    ${summaryHtml(s, mapping, model)}

    <div class="batch-actions">
      <button type="button" class="btn" id="batch-download">Download results CSV</button>
    </div>

    <table class="weights results">
      <thead><tr><th>Record</th><th class="num">Estimate</th><th class="num">Actual</th><th class="num">Gap</th></tr></thead>
      <tbody>
        ${output.results
          .map(
            (r) => `<tr>
              <td>${esc(r.label)}</td>
              <td class="num"${r.error ? ` title="${esc(r.error)}"` : ''}>${r.estimate === null ? '<span class="warn">no formula</span>' : fmt(r.estimate) + missingAlert(r.imputed, r.missingLocation)}</td>
              <td class="num">${r.actual === null ? '<span class="muted">—</span>' : fmt(r.actual)}</td>
              <td class="num">${signedPct(r.gapPct)}</td>
            </tr>`,
          )
          .join('')}
      </tbody>
    </table>
    ${s.imputedRows ? `<p class="muted small"><span class="alert-icon">⚠</span> The record lacks a field the formula uses; the training average was used, or for missing latitude/longitude no location adjustment was made (hover for which). Fields the formula does not use are ignored.</p>` : ''}
    ${s.noEstimate ? `<p class="muted small"><span class="warn">no formula</span>: the row's group was skipped in training or never seen (hover for why).</p>` : ''}`;

  el.querySelectorAll('.map-grid select').forEach((sel) =>
    sel.addEventListener('change', () => onMapping(sel.dataset.kind, sel.dataset.term, sel.value || null)),
  );
  el.querySelector('#batch-download').addEventListener('click', onDownload);
}

function summaryHtml(s, mapping, model) {
  if (!mapping.actual) {
    return `<div class="notice">Pick the <b>Actual value</b> column to compare estimates with real ${esc(model.outcome.name)} values.</div>`;
  }
  const noEstimate = s.noEstimate ? `<div class="notice">${s.noEstimate} row${s.noEstimate > 1 ? 's have' : ' has'} no formula for its group, so no estimate.</div>` : '';
  if (!s.withActual) {
    return `${noEstimate}<div class="notice">No row has both an estimate and a number in <b>${esc(mapping.actual)}</b>, so there is nothing to compare yet.</div>`;
  }
  // Only call out a direction when the average gap is a real share of the typical gap.
  const bias = Math.abs(s.meanGap) < 0.1 * s.mae ? 'no clear bias' : s.meanGap > 0 ? 'estimates run low' : 'estimates run high';
  const noActual = s.rows - s.noEstimate - s.withActual;
  return `
    ${noEstimate}
    ${noActual ? `<div class="notice">${noActual} of ${s.rows} rows have no ${esc(mapping.actual)}; they get an estimate but no gap.</div>` : ''}
    <div class="metrics">
      <div><span>Compared</span><b>${s.withActual}</b></div>
      <div><span>Median |gap|</span><b>${pct(s.medianApe)}</b></div>
      <div><span>Within ±10%</span><b>${pct(s.within10)}</b></div>
    </div>
    <div class="kv small">
      <span>Mean |gap|</span><b>${fmt(s.mae)} (${pct(s.mape)})</b>
      <span>Average gap</span><b>${signed(s.meanGap)} · ${bias}</b>
      <span>RMSE</span><b>${fmt(s.rmse)}</b>
    </div>`;
}

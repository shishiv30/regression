// Right panel, batch section: column mapping, gap summary and results table.
import { modelInputs } from '../calc/index.js';
import { fmt, signed, esc, missingAlert, rangeText, confidenceBadge, suspectAlert } from './format.js';

const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);
const signedPct = (v) => (v === null ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}%`);

export function renderBatch(el, { batch, model, mapping, output, scope, previous = null, diagnosis }, { onMapping, onDownload, onDiagnose, onIgnore } = {}) {
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
    ${scope ? `<div class="notice">Showing the ${output.results.length} of ${scope.total} rows with <b>${esc(scope.groupBy)}</b> = <b>${esc(scope.group)}</b>, the group selected on the left. Pick another group there to see its rows; the download has every row.</div>` : ''}

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
    ${previous && s.withActual && previous.withActual ? previousHtml(s, previous) : ''}
    ${mapping.actual && s.withActual ? diagnosisHtml(diagnosis, s) : ''}

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
              <td class="num"${r.error ? ` title="${esc(r.error)}"` : r.low !== null ? ` title="likely ${rangeText(r)}"` : ''}>${r.estimate === null ? '<span class="warn">no formula</span>' : fmt(r.estimate) + missingAlert(r.imputed, r.missingLocation)}${r.confidence ? ` ${confidenceBadge(r.confidence)}` : ''}</td>
              <td class="num">${r.actual === null ? '<span class="muted">—</span>' : fmt(r.actual)}</td>
              <td class="num">${signedPct(r.gapPct)}${suspectAlert(r.suspect)}</td>
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
  el.querySelector('#diagnose-btn')?.addEventListener('click', onDiagnose);
  el.querySelectorAll('[data-ignore]').forEach((b) => b.addEventListener('click', () => onIgnore(JSON.parse(b.dataset.ignore))));
}

// The same rows' figures from the model trained before this one.
function previousHtml(s, p) {
  const d = s.medianApe - p.medianApe;
  const verdict = Math.abs(d) < 0.0005 ? 'about the same' : d < 0 ? '<span class="pos">better</span>' : '<span class="neg">worse</span>';
  return `<p class="muted small">Before the last retrain: median |gap| ${pct(p.medianApe)}, within ±10% ${pct(p.within10)}, mean |gap| ${pct(p.mape)} (${p.withActual} rows). Now ${verdict}.</p>`;
}

// What drives the big gaps: one row per input, then plain-language hints.
function diagnosisHtml(d, s) {
  if (d === undefined) {
    return `<div class="batch-actions"><button type="button" class="btn" id="diagnose-btn" title="Re-estimate the compared rows with each field left out, and look at what the biggest gaps have in common">Find what drives the big gaps</button></div>`;
  }
  if (d === null) return `<p class="muted small">Too few compared rows to diagnose (needs 20).</p>`;
  const share = ({ big, rest }) => `${pct(big)} <span class="muted">vs ${pct(rest)}</span>`;
  const verdict = { hurts: '<span class="neg">hurts</span>', helps: '<span class="pos">helps</span>', unclear: '<span class="muted">unclear</span>' };
  const rows = d.fields
    .map((f) => `<tr><td>${esc(f.name)}</td><td class="num">${pct(f.without.medianApe)}</td><td>${verdict[f.verdict]}</td><td class="num${f.driverPattern ? ' warn' : ''}">${share(f.driver)}</td><td class="num${f.unusualPattern ? ' warn' : ''}">${share(f.unusual)}</td></tr>`)
    .join('');
  const hints = [];
  for (const f of d.fields) {
    const name = `<b>${esc(f.name)}</b>`;
    if (f.verdict === 'hurts') {
      hints.push(`<li>Estimates were better with ${name} left out: median |gap| ${pct(d.before.medianApe)} → ${pct(f.without.medianApe)}, on most rows (paired test, 95%). Try ignoring it and retrain; the result shows next to this run's. <button type="button" class="btn small" data-ignore='${esc(JSON.stringify(f.columns))}'>Ignore ${esc(f.name)}</button></li>`);
    }
    if (f.driverPattern) hints.push(`<li>${name} pushed ${pct(f.driver.big)} of the biggest gaps further off, against ${pct(f.driver.rest)} of the other rows. Check its values, its Scale (Log / Linear), or whether it means the same in these records as in training.</li>`);
    if (f.unusualPattern) hints.push(`<li>${pct(f.unusual.big)} of the biggest gaps have an unusual ${name} (blank, outside the training range, or rarely seen in training), against ${pct(f.unusual.rest)} of the other rows. More training records like these would help; for an enum with many rare values, try it as a number or Ignore.</li>`);
  }
  if (!hints.length) hints.push(`<li>No field stands out: leaving any one out doesn't help, and the biggest gaps look like the other rows. They likely come from things the data doesn't record.</li>`);
  return `
    <details class="diagnosis" open>
      <summary>What drives the big gaps</summary>
      <p class="muted small">${d.rows} compared rows; the biggest gaps are the ${d.bigRows} with |gap| ≥ ${pct(d.bigCut)}. Checked on this file, so confirm a change by retraining.</p>
      <ul class="hints">${hints.join('')}</ul>
      <table class="weights diag">
        <thead><tr><th>Field</th><th class="num" title="Median |gap| with this field set to its training average (for the location: no location adjustment)">Without it</th><th title="Paired test on each row's |gap| with and without the field (95%)">Field</th><th class="num" title="Share of the biggest gaps (vs the other rows) where this field moved the estimate furthest towards its error">Main driver</th><th class="num" title="Share of the biggest gaps (vs the other rows) with a blank value, a value outside the training range or one rarely seen in training">Unusual value</th></tr></thead>
        <tbody><tr><td class="muted">all fields</td><td class="num">${pct(d.before.medianApe)}</td><td></td><td></td><td></td></tr>${rows}</tbody>
      </table>
    </details>`;
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
    ${s.screened ? `<div class="notice">${s.screened} row${s.screened > 1 ? 's are' : ' is'} left out of the gap figures (⚑ in the table, hover for why): the actual value is out of line with every training record of its kind, or an input is far outside the training range. They still get an estimate. Usually a typo or a different kind of record.</div>` : ''}
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
      ${s.inRange === null ? '' : `<span title="Share of actual values inside the estimate's likely range; about 80% means the ranges are honest">In likely range</span><b>${pct(s.inRange)} <span class="muted">(aim ≈ 80%)</span></b>`}
      ${Object.entries(s.byLevel ?? {}).map(([level, v]) => `<span>${confidenceBadge(level)} confidence</span><b>median |gap| ${pct(v.medianApe)} <span class="muted">· ${v.n} rows</span></b>`).join('')}
    </div>`;
}

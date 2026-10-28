// Top bar: one card per CSV column to pick what it is (a number or enum
// input, the outcome, a coordinate, a label or ignored), showing the numeric
// range or the full enum dictionary.
import { fmt, esc } from './format.js';

// One dropdown covers both the role and the type: inputs are trained on as a
// number or an enum; a label is never trained on, only shown on hover.
export const kindOf = (role, type) => (role === 'input' ? type : role || 'ignore');

export function renderFields(el, { dataset, stats, config }, onChange) {
  if (!dataset) {
    el.innerHTML = '<p class="muted hint">Import a CSV of real records. Each column becomes a field you can mark as an input (x), the outcome (y), a coordinate or a label.</p>';
    return;
  }
  el.replaceChildren(
    ...dataset.columns.map((col) => {
      const s = stats[col];
      const role = config.roles[col] || 'ignore';
      const grouped = config.groupBy === col;
      const card = document.createElement('div');
      card.className = `field role-${grouped ? 'group' : role === 'lat' || role === 'lng' ? 'location' : role}`;
      const num = s.numericOk ? '' : 'disabled';

      const error = role === 'outcome' && s.type !== 'number'
        ? '<div class="field-error">Outcome must be numeric</div>'
        : (role === 'lat' || role === 'lng') && s.type !== 'number'
          ? '<div class="field-error">Coordinates must be numeric</div>'
        : !grouped && role === 'input' && s.type === 'enum' && s.values.length < 2
          ? '<div class="field-error">Needs ≥ 2 distinct values</div>'
          : '';

      card.innerHTML = `
        <div class="field-name" title="${esc(col)}">${esc(col)}</div>
        <div class="field-controls">
          <select data-k="kind" aria-label="Use of ${esc(col)}" ${grouped ? 'disabled title="Grouped: one formula per value instead of an input"' : ''}>
            <option value="number" ${num}>Number (x)</option>
            <option value="enum">Enum (x)</option>
            <option value="outcome" ${num}>Outcome (y)</option>
            <option value="lat" ${num}>Latitude</option>
            <option value="lng" ${num}>Longitude</option>
            <option value="label" title="Not trained on; shown when hovering a record">Label</option>
            <option value="ignore">Ignore</option>
          </select>
        </div>
        ${
          s.type === 'enum' && role !== 'outcome'
            ? `<label class="group-toggle" title="Train a separate formula for each value, using only that value's records">
                <input type="checkbox" data-k="groupBy" ${grouped ? 'checked' : ''} />
                Group by${grouped ? ` <span class="muted">· ${s.values.length} formulas</span>` : ''}</label>`
            : ''
        }
        <div class="field-summary">${summary(s)}</div>
        ${error}`;
      const sel = card.querySelector('[data-k=kind]');
      sel.value = kindOf(role, s.type);
      sel.addEventListener('change', () => onChange(col, 'kind', sel.value));
      card.querySelector('[data-k=groupBy]')?.addEventListener('change', (e) => onChange(col, 'groupBy', e.target.checked));
      return card;
    }),
  );
}

function summary(s) {
  const missing = s.missing ? ` · <span class="warn">${s.missing} missing</span>` : '';
  if (s.type === 'number') {
    return `<span class="range">${fmt(s.min)} – ${fmt(s.max)}</span><span class="muted"> · mean ${fmt(s.mean)}</span>${missing}`;
  }
  const chips = s.values
    .slice(0, 200)
    .map((v) => `<span class="chip" title="${s.counts[v]} records">${esc(v)}</span>`)
    .join('');
  const more = s.values.length > 200 ? `<span class="muted">…${s.values.length - 200} more</span>` : '';
  return `<details><summary>${s.values.length} values${missing}</summary><div class="chips">${chips}${more}</div></details>`;
}

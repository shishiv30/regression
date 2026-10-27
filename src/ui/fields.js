// Top bar: one card per CSV column to pick its role and type, showing the
// numeric range or the full enum dictionary.
import { fmt, esc } from './format.js';

export function renderFields(el, { dataset, stats, config }, onChange) {
  if (!dataset) {
    el.innerHTML = '<p class="muted hint">Import a CSV of real records. Each column becomes a field you can mark as an input (x) or the outcome (y).</p>';
    return;
  }
  el.replaceChildren(
    ...dataset.columns.map((col) => {
      const s = stats[col];
      const role = config.roles[col] || 'ignore';
      const grouped = config.groupBy === col;
      const card = document.createElement('div');
      card.className = `field role-${grouped ? 'group' : role === 'lat' || role === 'lng' ? 'location' : role}`;

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
          <select data-k="role" aria-label="Role of ${esc(col)}" ${grouped ? 'disabled title="Grouped: one formula per value instead of an input"' : ''}>
            <option value="ignore">Ignore</option>
            <option value="input">Input (x)</option>
            <option value="outcome">Outcome (y)</option>
            <option value="lat" ${s.type === 'number' ? '' : 'disabled'}>Latitude</option>
            <option value="lng" ${s.type === 'number' ? '' : 'disabled'}>Longitude</option>
          </select>
          <select data-k="type" aria-label="Type of ${esc(col)}">
            <option value="number" ${s.numericOk ? '' : 'disabled'}>Number</option>
            <option value="enum">Enum</option>
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
      card.querySelector('[data-k=role]').value = role;
      card.querySelector('[data-k=type]').value = s.type;
      card.querySelectorAll('select').forEach((sel) =>
        sel.addEventListener('change', () => onChange(col, sel.dataset.k, sel.value)),
      );
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

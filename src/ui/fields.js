// Top bar: one card per CSV column to pick what it is (a number or enum
// input, the outcome, a coordinate, a label or ignored), showing the numeric
// range or the full enum dictionary.
import { fmt, esc } from './format.js';

// One dropdown covers both the role and the type: inputs are trained on as a
// number or an enum; a label is never trained on, only shown on hover.
export const kindOf = (role, type) => (role === 'input' ? type : role || 'ignore');

export function renderFields(el, { dataset, stats, config }, onChange) {
  hidePopup();
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
            <option value="number" ${num}>Metric (x)</option>
            <option value="enum">Dimension (x)</option>
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
        ${
          !grouped && role === 'input' && s.type === 'number' && s.min > 0
            ? `<label class="scale-pick" title="Log: each 10% more has the same effect at any size, so big values don't run away (e.g. sqft vs. price). Auto picks log when it fits the outcome better and the values span at least 2×.">
                Scale <select data-k="scale" aria-label="Scale of ${esc(col)}">
                  <option value="auto">Auto</option>
                  <option value="linear">Linear</option>
                  <option value="log">Log</option>
                </select></label>`
            : ''
        }
        <div class="field-summary">${summary(s)}</div>
        ${error}`;
      const sel = card.querySelector('[data-k=kind]');
      sel.value = kindOf(role, s.type);
      sel.addEventListener('change', () => onChange(col, 'kind', sel.value));
      const scale = card.querySelector('[data-k=scale]');
      if (scale) {
        scale.value = config.scales?.[col] ?? 'auto';
        scale.addEventListener('change', () => onChange(col, 'scale', scale.value));
      }
      card.querySelector('[data-k=groupBy]')?.addEventListener('change', (e) => onChange(col, 'groupBy', e.target.checked));
      const values = card.querySelector('.enum-values');
      if (values) bindValues(values, col, s);
      return card;
    }),
  );
}

function summary(s) {
  const missing = s.missing ? ` · <span class="warn">${s.missing} missing</span>` : '';
  if (s.type === 'number') {
    return `<span class="range">${fmt(s.min)} – ${fmt(s.max)}</span><span class="muted"> · mean ${fmt(s.mean)}</span>${missing}`;
  }
  return `<button type="button" class="enum-values" aria-haspopup="true" aria-expanded="false">${s.values.length} values ▾</button>${missing}`;
}

// One shared overlay lists an enum's values below its card. The cards sit in
// a scrolling row, so it is fixed to the window rather than inside the card.
// Hovering shows it; clicking keeps it open until a click elsewhere or Esc.
let popup = null;
let owner = null;
let pinned = false;
let hideTimer = 0;

function ensurePopup() {
  if (popup) return popup;
  popup = document.createElement('div');
  popup.className = 'values-popup';
  popup.setAttribute('role', 'dialog');
  popup.hidden = true;
  popup.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  popup.addEventListener('mouseleave', () => !pinned && scheduleHide());
  document.body.append(popup);
  document.addEventListener('mousedown', (e) => {
    if (!popup.hidden && !popup.contains(e.target) && !owner?.contains(e.target)) hidePopup();
  });
  document.addEventListener('keydown', (e) => e.key === 'Escape' && hidePopup());
  window.addEventListener('resize', hidePopup);
  document.addEventListener('scroll', (e) => !popup.contains(e.target) && hidePopup(), true);
  return popup;
}

function showPopup(btn, col, s) {
  const el = ensurePopup();
  clearTimeout(hideTimer);
  if (owner !== btn) {
    owner?.setAttribute('aria-expanded', 'false');
    owner = btn;
    const byCount = [...s.values].sort((a, b) => s.counts[b] - s.counts[a]);
    el.innerHTML = `<div class="values-head"><b>${esc(col)}</b> <span class="muted">· ${s.values.length} values${s.missing ? ` · ${s.missing} missing` : ''}</span></div>
      <div class="chips">${byCount.map((v) => `<span class="chip">${esc(v)} <span class="muted">${s.counts[v]}</span></span>`).join('')}</div>`;
  }
  btn.setAttribute('aria-expanded', 'true');
  el.hidden = false;
  // Below the button, kept inside the window.
  const r = btn.getBoundingClientRect();
  const w = el.offsetWidth;
  el.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - w - 8))}px`;
  el.style.top = `${r.bottom + 4}px`;
  el.style.maxHeight = `${Math.max(120, window.innerHeight - r.bottom - 16)}px`;
}

function hidePopup() {
  clearTimeout(hideTimer);
  pinned = false;
  if (!popup || popup.hidden) return;
  popup.hidden = true;
  owner?.setAttribute('aria-expanded', 'false');
  owner = null;
}

function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(hidePopup, 150);
}

function bindValues(btn, col, s) {
  btn.addEventListener('mouseenter', () => showPopup(btn, col, s));
  btn.addEventListener('mouseleave', () => !pinned && scheduleHide());
  btn.addEventListener('click', () => {
    if (pinned && owner === btn) return hidePopup();
    showPopup(btn, col, s);
    pinned = true;
  });
}

export function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e15 || (a !== 0 && a < 1e-4)) return v.toExponential(2);
  if (a >= 1e6) return v.toLocaleString(undefined, { notation: 'compact', maximumSignificantDigits: 4 });
  return v.toLocaleString(undefined, { maximumFractionDigits: a >= 100 ? 1 : a >= 1 ? 3 : 4 });
}

export const signed = (v) => (v >= 0 ? '+' : '−') + fmt(Math.abs(v));

// A weight or effect in model space as text: on a log scale it multiplies the
// outcome, so it reads as a percentage; otherwise in outcome units.
export const pctText = (r) => `${r >= 0 ? '+' : '−'}${fmt(Math.abs(r * 100))}%`;
export const effect = (model, w) => (model.outcome.log ? pctText(Math.exp(w) - 1) : signed(w));
// A ± spread in model space.
export const spreadText = (model, s) => (model.outcome.log ? `±${fmt((Math.exp(s) - 1) * 100)}%` : `±${fmt(s)}`);

// "low – high" around an estimate.
export const rangeText = (c) => (c?.low != null ? `${fmt(c.low)} – ${fmt(c.high)}` : '');

const LEVEL_TEXT = { high: 'High', medium: 'Medium', low: 'Low' };
export const confidenceBadge = (level) => (level ? `<span class="conf conf-${level}">${LEVEL_TEXT[level]}</span>` : '');

// Alert after an estimate whose formula needed fields the record did not
// provide (no matching column or a blank cell); the training average was used.
// Missing coordinates mean no location adjustment instead.
export function missingText(fields, location = []) {
  const parts = [];
  if (fields.length) parts.push(`Missing from this record: ${fields.join(', ')}. The training average was used, so the estimate is less reliable.`);
  if (location.length) parts.push(`No usable ${location.join(', ')}, so no location adjustment.`);
  return parts.join(' ');
}

export function missingAlert(fields, location = []) {
  const text = missingText(fields, location);
  if (!text) return '';
  return ` <span class="alert-icon" role="img" aria-label="${esc(text)}" title="${esc(text)}">⚠</span>`;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

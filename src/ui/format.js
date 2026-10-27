export function fmt(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e15 || (a !== 0 && a < 1e-4)) return v.toExponential(2);
  if (a >= 1e6) return v.toLocaleString(undefined, { notation: 'compact', maximumSignificantDigits: 4 });
  return v.toLocaleString(undefined, { maximumFractionDigits: a >= 100 ? 1 : a >= 1 ? 3 : 4 });
}

export const signed = (v) => (v >= 0 ? '+' : '−') + fmt(Math.abs(v));

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

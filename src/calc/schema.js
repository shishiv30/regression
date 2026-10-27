// Field analysis: decide whether a column is a number or an enum and
// summarise it (numeric range, or the deduplicated enum dictionary).

const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

export const isMissing = (v) => v == null || String(v).trim() === '';

export function toNumber(v) {
  if (typeof v === 'number') return v;
  const s = String(v ?? '').replace(/,/g, '').trim();
  return NUM_RE.test(s) ? Number(s) : NaN;
}

// forcedType: 'number' | 'enum' | undefined (auto-detect)
export function analyzeColumn(rows, name, forcedType) {
  const present = [];
  let missing = 0;
  for (const row of rows) {
    if (isMissing(row[name])) missing++;
    else present.push(String(row[name]).trim());
  }
  const numericOk = present.length > 0 && present.every((v) => Number.isFinite(toNumber(v)));
  const type = forcedType === 'enum' || !numericOk ? 'enum' : 'number';

  if (type === 'number') {
    let min = Infinity, max = -Infinity, sum = 0;
    for (const v of present) {
      const n = toNumber(v);
      if (n < min) min = n;
      if (n > max) max = n;
      sum += n;
    }
    return { name, type, numericOk, count: present.length, missing, min, max, mean: sum / present.length };
  }

  const counts = new Map();
  for (const v of present) counts.set(v, (counts.get(v) || 0) + 1);
  // Dictionary sorted by frequency (desc), then alphabetically.
  const values = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
  return { name, type, numericOk, count: present.length, missing, values, counts: Object.fromEntries(counts) };
}

export function analyzeColumns(columns, rows, types = {}) {
  const out = {};
  for (const c of columns) out[c] = analyzeColumn(rows, c, types[c]);
  return out;
}

// Sensible starting roles: last numeric column is the outcome; other numeric
// columns and low-cardinality enums are inputs; ids / free text are ignored.
export function suggestRoles(columns, stats) {
  const roles = {};
  const numeric = columns.filter((c) => stats[c].type === 'number');
  const outcome = numeric[numeric.length - 1];
  for (const c of columns) {
    const s = stats[c];
    const looksLikeId = /^id$|[_\s-]id$/i.test(c);
    if (c === outcome) roles[c] = 'outcome';
    else if (looksLikeId) roles[c] = 'ignore';
    else if (s.type === 'number') roles[c] = 'input';
    else roles[c] = s.values.length >= 2 && s.values.length <= 30 ? 'input' : 'ignore';
  }
  return roles;
}

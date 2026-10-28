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
    let min = Infinity, max = -Infinity, sum = 0, whole = 0;
    const nums = new Float64Array(present.length);
    const digits = new Float64Array(present.length);
    present.forEach((v, i) => {
      const n = (nums[i] = toNumber(v));
      if (n < min) min = n;
      if (n > max) max = n;
      sum += n;
      if (Number.isInteger(n)) whole++;
      digits[i] = /\.(\d+)/.exec(v)?.[1].replace(/0+$/, '').length ?? 0;
    });
    nums.sort();
    digits.sort();
    const m = nums.length >> 1;
    const median = nums.length % 2 ? nums[m] : (nums[m - 1] + nums[m]) / 2;
    return {
      name, type, numericOk, count: present.length, missing, min, max, mean: sum / present.length, median,
      distinct: new Set(nums).size,
      wholeShare: whole / present.length, // share of whole numbers
      decimals: digits[m], // typical number of decimal places
    };
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

// Suggestions below go by the values only, never by column names, so they
// work the same for any subject.

// An enum with more distinct values than this is free text or an id: a label.
export const MAX_ENUM_VALUES = 30;
// Coordinates are recorded to about 1 m or finer: 4+ decimal places on
// typical values and almost no whole numbers.
const COORD_DECIMALS = 4;

const coordLike = (s, limit) =>
  s?.type === 'number' && s.decimals >= COORD_DECIMALS && s.wholeShare < 0.05 && s.min >= -limit && s.max <= limit && s.max > s.min;

// The first pair of columns that look like latitude and longitude (both
// within ±180, one within ±90), or nulls. With both within ±90 the first
// column is taken as the latitude.
export function detectCoords(columns, stats) {
  const cands = columns.filter((c) => coordLike(stats[c], 180));
  const lat = cands.find((c) => coordLike(stats[c], 90));
  const lng = lat && cands.find((c) => c !== lat);
  return lat && lng ? { lat, lng } : { lat: null, lng: null };
}

// A row counter (1, 2, 3, … in any order): whole numbers, all different, no
// gaps. It numbers records rather than measuring them.
const counterLike = (s) =>
  s.type === 'number' && s.count >= 10 && s.wholeShare === 1 && s.distinct === s.count && s.max - s.min + 1 === s.count;

// Sensible starting roles: a coordinate-like pair is the location; the last
// other numeric column is the outcome; the remaining numeric columns and enums
// with 2–30 values are inputs; row counters and enums with more values (ids,
// free text) become labels (shown on hover, never trained on).
export function suggestRoles(columns, stats) {
  const roles = {};
  const { lat, lng } = detectCoords(columns, stats);
  const outcome = columns.filter((c) => stats[c].type === 'number' && c !== lat && c !== lng && !counterLike(stats[c])).pop();
  for (const c of columns) {
    const s = stats[c];
    if (c === lat) roles[c] = 'lat';
    else if (c === lng) roles[c] = 'lng';
    else if (c === outcome) roles[c] = 'outcome';
    else if (counterLike(s)) roles[c] = 'label';
    else if (s.type === 'number') roles[c] = 'input';
    else if (s.values.length > MAX_ENUM_VALUES) roles[c] = 'label';
    else roles[c] = s.values.length >= 2 ? 'input' : 'ignore';
  }
  return roles;
}

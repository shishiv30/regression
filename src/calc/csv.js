import Papa from 'papaparse';

// Parse CSV text into { columns, rows } where every cell is a trimmed string.
export function parseCsv(text) {
  const res = Papa.parse(text, {
    header: true,
    skipEmptyLines: 'greedy',
    transformHeader: (h) => h.trim(),
  });
  const columns = (res.meta.fields || []).filter(Boolean);
  if (!columns.length) {
    throw new Error(res.errors[0]?.message || 'CSV has no header row.');
  }
  const rows = res.data.map((r) => {
    const row = {};
    for (const c of columns) row[c] = r[c] == null ? '' : String(r[c]).trim();
    return row;
  });
  if (!rows.length) throw new Error('CSV has a header but no records.');
  return { columns, rows };
}

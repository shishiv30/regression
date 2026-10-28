// Main-area "Data table" tab: every row of the batch CSV with all its
// columns, plus the estimate and gap. Click a header to sort.
import { fmt, signed, esc, missingAlert, rangeText, confidenceBadge, suspectAlert } from './format.js';

const LEVEL_ORDER = { high: 0, medium: 1, low: 2 };
const MAX_ROWS = 2000; // keep the DOM responsive on big files

export function createDataTable(el) {
  let data = null;
  let sort = null; // { key, dir }

  function render() {
    if (!data) {
      el.innerHTML = '<p class="muted hint">Import a CSV under <b>Predict many records</b> (right panel) after training to see every record with its estimate here.</p>';
      return;
    }
    const { batch, results, mapping, outcomeName, locationFields = [] } = data;
    // Which input each mapped column feeds, for header badges and imputed cells.
    const termOf = {};
    for (const [term, col] of Object.entries(mapping.inputs)) if (col) termOf[col] = term;

    const field = (c) => ({ key: `c:${c}`, label: c, column: c, value: (r) => batch.rows[r.index][c] });
    const cols = [
      { key: 'est', label: `est_${outcomeName}`, value: (r) => r.estimate },
      ...(mapping.actual ? [field(mapping.actual)] : []),
      { key: '#', label: '#', value: (r) => r.index + 1 },
      ...(results.some((r) => r.confidence) ? [{ key: 'conf', label: 'confidence', value: (r) => LEVEL_ORDER[r.confidence] }] : []),
      ...(results.some((r) => r.low !== null) ? [{ key: 'range', label: 'likely range', value: (r) => r.low }] : []),
      ...(locationFields.length ? [{ key: 'loc', label: 'location adj', value: (r) => r.locationAdj }] : []),
      ...(results.some((r) => r.compsAdj !== null) ? [{ key: 'comps', label: 'comparables adj', value: (r) => r.compsAdj }] : []),
      ...(mapping.actual ? [
        { key: 'gap', label: 'gap', value: (r) => r.gap },
        { key: 'gapPct', label: 'gap %', value: (r) => r.gapPct },
      ] : []),
      ...batch.columns.filter((c) => c !== mapping.actual).map(field),
    ];

    let rows = results;
    if (sort) {
      const col = cols.find((c) => c.key === sort.key);
      if (col) rows = [...results].sort((a, b) => compare(col.value(a), col.value(b), sort.dir));
    }
    const shown = rows.slice(0, MAX_ROWS);

    const head = cols
      .map((c) => {
        const dir = sort?.key === c.key ? (sort.dir > 0 ? ' ▲' : ' ▼') : '';
        const located = locationFields.includes(termOf[c.column]);
        const cls = [c.column ? '' : 'est', c.column === mapping.actual ? 'actual' : '', termOf[c.column] ? 'input' : '', located ? 'location' : ''].filter(Boolean).join(' ');
        const role = termOf[c.column] === data.groupBy ? 'picks formula' : located ? 'location' : 'input';
        const badge = termOf[c.column] ? ` <span class="tag">${termOf[c.column] === c.column ? role : `${role} → ${esc(termOf[c.column])}`}</span>`
          : c.column && c.column === mapping.actual ? ' <span class="tag">actual</span>' : '';
        return `<th class="${cls}" data-key="${esc(c.key)}" title="Sort by ${esc(c.label)}">${esc(c.label)}${badge}${dir}</th>`;
      })
      .join('');

    const body = shown
      .map((r) => {
        const cells = cols.map((c) => {
          if (c.key === '#') return `<td class="num muted">${r.index + 1}</td>`;
          if (c.key === 'est') {
            if (r.estimate === null) return `<td class="num est" title="${esc(r.error ?? '')}"><span class="warn">no formula</span></td>`;
            return `<td class="num est"><b>${fmt(r.estimate)}</b>${missingAlert(r.imputed, r.missingLocation)}${suspectAlert(r.suspect)}</td>`;
          }
          if (c.key === 'conf') return `<td class="est">${confidenceBadge(r.confidence)}</td>`;
          if (c.key === 'range') return `<td class="num est">${r.low === null ? '<span class="muted">—</span>' : rangeText(r)}</td>`;
          if (c.key === 'comps') return `<td class="num est loc-adj">${r.compsAdj === null ? '<span class="muted">—</span>' : signed(r.compsAdj)}</td>`;
          if (c.key === 'loc') return `<td class="num est loc-adj">${r.locationAdj === null ? '<span class="muted">—</span>' : signed(r.locationAdj)}</td>`;
          if (c.key === 'gap') return `<td class="num est">${r.gap === null ? '<span class="muted">—</span>' : gapHtml(r.gap, fmt(Math.abs(r.gap)))}</td>`;
          if (c.key === 'gapPct') return `<td class="num est">${r.gapPct === null ? '<span class="muted">—</span>' : gapHtml(r.gapPct, `${(Math.abs(r.gapPct) * 100).toFixed(1)}%`)}</td>`;
          const raw = c.value(r) ?? '';
          const term = termOf[c.column];
          if (term && r.imputed.includes(term)) {
            return `<td class="imputed" title="Blank or unusable: the training average was used for ${esc(term)}">${raw === '' ? 'avg' : esc(raw)}</td>`;
          }
          if (term && r.missingLocation.includes(term)) {
            return `<td class="imputed" title="Blank or unusable: no location adjustment for this record">${raw === '' ? 'none' : esc(raw)}</td>`;
          }
          return `<td>${esc(raw)}</td>`;
        });
        return `<tr>${cells.join('')}</tr>`;
      })
      .join('');

    const more = rows.length > shown.length ? ` · showing the first ${MAX_ROWS} (download the CSV for all)` : '';
    el.innerHTML = `
      <p class="table-caption muted small">${esc(batch.fileName)} · ${results.length} records${more}. Click a header to sort.</p>
      <table class="data-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>`;
    el.querySelectorAll('th[data-key]').forEach((th) =>
      th.addEventListener('click', () => {
        const key = th.dataset.key;
        sort = sort?.key === key ? (sort.dir > 0 ? { key, dir: -1 } : null) : { key, dir: 1 };
        render();
      }),
    );
  }

  return {
    setData(next) {
      // Keep the sort only while the same columns are shown.
      if (!next || !data || next.batch !== data.batch) sort = null;
      data = next;
      render();
    },
  };
}

// Gap = actual − estimate: positive means the actual value is above the estimate.
function gapHtml(v, text) {
  if (v === 0) return text;
  return `<span class="${v > 0 ? 'above' : 'below'}">${v > 0 ? '+' : '−'}${text}</span>`;
}

// Numbers sort numerically, text alphabetically, blanks last in either direction.
function compare(a, b, dir) {
  const blank = (v) => v === null || v === undefined || String(v).trim() === '';
  if (blank(a) || blank(b)) return blank(a) - blank(b);
  const na = Number(String(a).replace(/,/g, '')), nb = Number(String(b).replace(/,/g, ''));
  if (Number.isFinite(na) && Number.isFinite(nb)) return (na - nb) * dir;
  return String(a).localeCompare(String(b), undefined, { numeric: true }) * dir;
}

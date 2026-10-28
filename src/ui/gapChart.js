// Gap chart: every record that got an estimate, in one of three layouts.
//  · Map: each record is a node at its latitude/longitude. Fill colour is the
//    direction of the gap (blue = actual above the estimate, orange-red =
//    below), size its magnitude (|gap %|); records without a real value are
//    hollow rings. Records sharing the "Link by" value (e.g. an area code) hang
//    off a small hub at their centre.
//  · Estimate vs actual: actual across, estimate up. On the diagonal the
//    estimate is exact; each node's stem is its gap back to the diagonal.
//  · List: one row per record, sorted by gap %, which is printed at the right
//    so colour is never the only cue.
// Map and scatter zoom with the wheel and pan by dragging. Hovering a node
// shows its labels (fields marked Label) and numbers, and highlights the
// records it is linked with.
import { fmt, signed, esc, missingText, confidenceBadge, rangeText } from './format.js';
import { toNumber } from '../calc/schema.js';

const ROW = 28;
const HEAD = 82; // list: title, legend and the value axis
const AXIS = 16;
const LABEL_W = 130;
const PCT_W = 64;
const TOP = 64; // map / scatter: title and legend
const KM_PER_DEG = 111.32;
const R_MIN = 3.5, R_MAX = 12; // node radius from |gap| 0 to 50%+

function tokens() {
  const s = getComputedStyle(document.documentElement);
  const v = (n) => s.getPropertyValue(n).trim();
  return { fg: v('--fg'), muted: v('--muted'), border: v('--border'), panel: v('--panel'), bg: v('--bg'), above: v('--above'), below: v('--below'), accent: v('--accent') };
}

// ~6 round tick values covering [lo, hi].
function niceTicks(lo, hi, count = 6) {
  const raw = (hi - lo) / count || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

// 1-2-5 ticks on a log axis covering [lo, hi] (both > 0); falls back to
// linear ticks when the span is under ~3×.
function logTicks(lo, hi) {
  if (hi / lo < 3) return niceTicks(lo, hi, 5);
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of hi / lo > 300 ? [1] : [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) ticks.push(v);
    }
  }
  return ticks;
}

const radius = (r) => (r.gapPct === null ? R_MIN + 1 : R_MIN + (R_MAX - R_MIN) * Math.sqrt(Math.min(Math.abs(r.gapPct), 0.5) / 0.5));
const pctText = (g) => (g === null ? '—' : `${g >= 0 ? '+' : '−'}${Math.abs(g * 100).toFixed(1)}%`);
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function createGapChart({ canvas, scroller, bar, linkSelect, hint, tooltip }) {
  const ctx = canvas.getContext('2d');
  let data = null;
  let layout = null; // 'map' | 'scatter' | 'list'; null = best for the data
  let linkBy = undefined; // undefined = suggested; null = none
  let view = { k: 1, x: 0, y: 0 }; // zoom and pan of map / scatter
  let nodes = []; // { r, sx, sy, rad, link } in screen space
  let hubs = []; // { key, sx, sy, members }
  let rows = []; // list layout
  let geom = null;
  let hover = null; // { node } | { hub }
  let drag = null;
  let origin = [0, 0]; // the screen point zoom scales around

  const layoutButtons = [...bar.querySelectorAll('[data-layout]')];
  const hasCoords = () => !!data?.coords;
  const current = () => layout === 'map' && !hasCoords() ? 'scatter' : layout ?? (hasCoords() ? 'map' : 'scatter');

  function coordsOf(r) {
    const row = data.rows[r.index];
    const lat = toNumber(row[data.coords.lat]), lng = toNumber(row[data.coords.lng]);
    return Math.abs(lat) <= 90 && Math.abs(lng) <= 180 && (lat || lng) ? [lat, lng] : null;
  }
  const linkOf = (r) => {
    if (!linkBy) return null;
    const v = data.rows[r.index][linkBy];
    return v == null || String(v).trim() === '' ? null : String(v).trim();
  };

  // ---------- toolbar ----------
  function renderBar() {
    bar.hidden = !data;
    if (!data) return;
    const cols = data.columns;
    if (linkBy !== null && !cols.includes(linkBy)) linkBy = data.linkBy ?? null;
    linkSelect.innerHTML = ['<option value="">None</option>', ...cols.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`)].join('');
    linkSelect.value = linkBy ?? '';
    const now = current();
    for (const b of layoutButtons) {
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', String(b.dataset.layout === now));
      b.disabled = b.dataset.layout === 'map' && !hasCoords();
      b.dataset.tip ??= b.title;
      b.title = b.disabled ? 'Needs latitude/longitude columns in the batch file' : b.dataset.tip;
    }
  }
  for (const b of layoutButtons) {
    b.addEventListener('click', () => {
      layout = b.dataset.layout;
      resetView();
      renderBar();
      draw();
    });
  }
  linkSelect.addEventListener('change', () => {
    linkBy = linkSelect.value || null;
    draw();
  });

  function resetView() {
    view = { k: 1, x: 0, y: 0 };
    hover = null;
  }

  // ---------- drawing ----------
  function draw() {
    const width = scroller.clientWidth;
    if (!width) return;
    const mode = current();
    const listRows = mode === 'list' && data?.actualColumn ? data.results.filter((r) => r.gap !== null) : [];
    const height = mode === 'list' ? Math.max(scroller.clientHeight, listRows.length ? HEAD + listRows.length * ROW + AXIS : 0) : scroller.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = tokens();
    ctx.clearRect(0, 0, width, height);
    nodes = [];
    hubs = [];
    rows = [];
    canvas.classList.toggle('pannable', mode !== 'list' && !!data);
    hint.textContent = '';

    const message = (lines) => {
      ctx.fillStyle = c.muted;
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      lines.forEach((l, i) => ctx.fillText(l, width / 2, height / 2 + (i - (lines.length - 1) / 2) * 22));
    };
    if (!data) return message(['Import a CSV under “Predict many records”', 'to compare estimates with real values.']);
    if (mode === 'map') return drawMap(width, height, c, message);
    if (!data.actualColumn) return message(['Pick the “Actual value” column', 'in the right panel to see the gaps.']);
    if (mode === 'scatter') return drawScatter(width, height, c, message);
    rows = listRows.sort((a, b) => (b.gapPct ?? 0) - (a.gapPct ?? 0));
    if (!rows.length) return message([`No row has both an estimate and a value in “${data.actualColumn}”.`]);
    drawList(width, c);
  }

  function header(c, title, sub, items, width) {
    const pad = 16;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = c.fg;
    ctx.font = '600 15px system-ui, sans-serif';
    ctx.fillText(title, pad, 24);
    ctx.fillStyle = c.muted;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(sub, pad, 42);
    // Legend on the next line when it does not fit beside the subtitle.
    let lx = pad + ctx.measureText(sub).width + 24;
    let ly = 42;
    const need = items.reduce((a, [, t]) => a + 32 + ctx.measureText(t).width, 0);
    if (lx + need > width - pad) [lx, ly] = [pad, 58];
    for (const [mark, text] of items) {
      mark(lx + 6, ly - 4);
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'left';
      ctx.fillText(text, lx + 16, ly);
      lx += 16 + ctx.measureText(text).width + 16;
    }
    return ly > 42 ? TOP + 14 : TOP;
  }

  const gapLegend = (c) => [
    [(x, y) => node(x, y, 5, c.above, c), 'Actual above estimate'],
    [(x, y) => node(x, y, 5, c.below, c), 'Below'],
    [(x, y) => { node(x - 3, y, 2.5, c.muted, c); node(x + 5, y, 4.5, c.muted, c); }, 'Size = |gap|, up to 50%'],
  ];

  function drawMap(width, height, c, message) {
    const placed = [];
    let noCoords = 0;
    for (const r of data.results) {
      if (r.estimate === null) continue;
      const p = coordsOf(r);
      if (p) placed.push({ r, lat: p[0], lng: p[1] });
      else noCoords++;
    }
    if (!placed.length) return message([`No row has usable ${data.coords.lat} / ${data.coords.lng}.`]);
    const withActual = placed.filter((p) => p.r.gap !== null).length;
    const items = [...(withActual ? gapLegend(c) : []), [(x, y) => ring(x, y, 4.5, c), 'No actual value']];
    if (linkBy) items.push([(x, y) => hub(x, y, c, false), `Same ${linkBy}`]);
    const top = header(c, `Estimate gaps on the map · ${data.outcomeName}`, `${placed.length} records${withActual ? ` · ${withActual} with ${data.actualColumn}` : ''}`, items, width);
    hint.textContent = `Scroll to zoom · drag to pan · double-click to reset${noCoords ? ` · ${noCoords} record${noCoords > 1 ? 's' : ''} without coordinates not shown` : ''}`;

    // Equirectangular projection, fitted to the plot keeping distances true.
    const pad = 24;
    const plot = { x0: pad, y0: top + 8, x1: width - pad, y1: height - pad };
    const lats = placed.map((p) => p.lat), lngs = placed.map((p) => p.lng);
    const [la0, la1, ln0, ln1] = [Math.min(...lats), Math.max(...lats), Math.min(...lngs), Math.max(...lngs)];
    const cos = Math.cos((((la0 + la1) / 2) * Math.PI) / 180);
    const spanX = Math.max((ln1 - ln0) * cos, 1e-4), spanY = Math.max(la1 - la0, 1e-4);
    const s = Math.min((plot.x1 - plot.x0) / spanX, (plot.y1 - plot.y0) / spanY);
    const cx = (plot.x0 + plot.x1) / 2, cy = (plot.y0 + plot.y1) / 2;
    const mx = (ln0 + ln1) / 2, my = (la0 + la1) / 2;
    origin = [cx, cy];
    const project = (lat, lng) => [cx + ((lng - mx) * cos * s) * view.k + view.x, cy - ((lat - my) * s) * view.k + view.y];

    nodes = placed.map(({ r, lat, lng }) => {
      const [sx, sy] = project(lat, lng);
      return { r, sx, sy, rad: r.gap === null ? 4.5 : radius(r), link: linkOf(r) };
    });
    buildHubs();

    clip(plot, () => {
      drawLinks(c, true);
      for (const h of hubs) hub(h.sx, h.sy, c, hover && hubKey() === h.key);
      drawNodes(c);
      hubLabels(c, plot);
    });
    scaleBar(c, plot, s * view.k);
  }

  function drawScatter(width, height, c, message) {
    const placed = data.results.filter((r) => r.gap !== null);
    if (!placed.length) return message([`No row has both an estimate and a value in “${data.actualColumn}”.`]);
    const items = [...gapLegend(c), [(x, y) => dashed(x - 7, y + 4, x + 7, y - 4, c.muted), 'Exact']];
    const top = header(c, `Estimated vs. actual ${data.outcomeName}`, `${placed.length} records with ${data.actualColumn}`, items, width);
    hint.textContent = 'Scroll to zoom · drag to pan · double-click to reset';

    // One shared scale for both axes, so the diagonal is 45°; logarithmic
    // when the values are all above 0 and span 10× or more.
    const values = placed.flatMap((r) => [r.estimate, r.actual]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const log = lo > 0 && hi / lo >= 10;
    const tr = log ? Math.log10 : (v) => v, un = log ? (v) => 10 ** v : (v) => v;
    let tlo = tr(lo), thi = tr(hi);
    const span = thi - tlo || Math.abs(thi) || 1;
    tlo -= span * 0.05;
    thi += span * 0.05;
    const plot = { x0: 72, y0: top + 8, x1: width - 24, y1: height - 40 };
    const size = Math.min(plot.x1 - plot.x0, plot.y1 - plot.y0);
    plot.x1 = plot.x0 + size;
    plot.y0 = plot.y1 - size;
    origin = [plot.x0, plot.y1];
    const sx = (v) => plot.x0 + ((tr(v) - tlo) / (thi - tlo)) * size * view.k + view.x;
    const sy = (v) => plot.y1 - ((tr(v) - tlo) / (thi - tlo)) * size * view.k + view.y;
    const vx = (px) => un(tlo + ((px - plot.x0 - view.x) / (size * view.k)) * (thi - tlo));
    const vy = (py) => un(tlo + ((plot.y1 + view.y - py) / (size * view.k)) * (thi - tlo));

    // Axes follow the zoom: ticks for the visible range only.
    const ticksOf = (a, b) => (log ? logTicks(a, b) : niceTicks(a, b, 5));
    ctx.font = '11px system-ui, sans-serif';
    ctx.lineWidth = 1;
    for (const t of ticksOf(vx(plot.x0), vx(plot.x1))) {
      const x = sx(t);
      line(x, plot.y0, x, plot.y1, c.border);
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(fmt(t), x, plot.y1 + 6);
    }
    for (const t of ticksOf(vy(plot.y1), vy(plot.y0))) {
      const y = sy(t);
      line(plot.x0, y, plot.x1, y, c.border);
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(fmt(t), plot.x0 - 6, y);
    }
    ctx.fillStyle = c.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`Actual ${data.actualColumn}${log ? ' (log scale)' : ''}`, (plot.x0 + plot.x1) / 2, plot.y1 + 22);
    ctx.save();
    ctx.translate(16, (plot.y0 + plot.y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.fillText(`Estimated ${data.outcomeName}`, 0, 0);
    ctx.restore();

    nodes = placed.map((r) => ({ r, sx: sx(r.actual), sy: sy(r.estimate), base: sy(r.actual), rad: radius(r), link: linkOf(r) }));
    clip(plot, () => {
      const a = vx(plot.x0), b = vx(plot.x1);
      dashed(sx(a), sy(a), sx(b), sy(b), c.muted);
      // Gap stems back to the diagonal.
      for (const n of nodes) line(n.sx, n.base, n.sx, n.sy, n.r.gap >= 0 ? c.above : c.below, dim(n) ? 0.15 : 0.6, 2);
      drawLinks(c, false);
      drawNodes(c);
    });
  }

  function drawList(width, c) {
    const pad = 16;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = c.fg;
    ctx.font = '600 15px system-ui, sans-serif';
    ctx.fillText(`Estimated vs. actual ${data.outcomeName}`, pad, 24);
    ctx.fillStyle = c.muted;
    ctx.font = '12px system-ui, sans-serif';
    const sub = `${rows.length} records with ${data.actualColumn} · sorted by gap`;
    ctx.fillText(sub, pad, 42);

    let lx = Math.max(pad + ctx.measureText(sub).width + 24, width - 500);
    const legendY = 42;
    const item = (mark, text) => {
      mark(lx + 6, legendY - 4);
      ctx.fillStyle = c.muted;
      ctx.fillText(text, lx + 16, legendY);
      lx += 16 + ctx.measureText(text).width + 16;
    };
    if (lx + 480 <= width) {
      item((x, y) => ring(x, y, 5, c), 'Estimate');
      item((x, y) => node(x, y, 5.5, c.fg, c), 'Actual');
      item((x, y) => line(x - 6, y, x + 6, y, c.above, 1, 3), 'Actual above estimate');
      item((x, y) => line(x - 6, y, x + 6, y, c.below, 1, 3), 'Below');
    }

    const values = rows.flatMap((r) => [r.estimate, r.actual]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || Math.abs(hi) || 1;
    lo -= span * 0.04;
    hi += span * 0.04;
    const x0 = pad + LABEL_W, x1 = width - pad - PCT_W;
    const x = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
    const top = HEAD;
    const bottom = HEAD + rows.length * ROW;
    geom = { top, bottom };

    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (const t of niceTicks(lo, hi)) {
      line(x(t), top - 2, x(t), bottom, c.border);
      ctx.fillStyle = c.muted;
      ctx.fillText(fmt(t), x(t), top - 8);
    }

    rows.forEach((r, i) => {
      const y = top + i * ROW + ROW / 2;
      const color = r.gap >= 0 ? c.above : c.below;
      ctx.fillStyle = c.fg;
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(truncate(r.label + (r.imputed.length || r.missingLocation.length ? ' ⚠' : ''), LABEL_W - 10), x0 - 10, y);
      ctx.lineCap = 'round';
      line(x(r.estimate), y, x(r.actual), y, color, 1, 3);
      ring(x(r.estimate), y, 5, c);
      node(x(r.actual), y, 5.5, color, c);
      ctx.textAlign = 'right';
      ctx.fillStyle = c.fg;
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.fillText(pctText(r.gapPct), width - pad, y);
    });
  }

  // ---------- links and nodes ----------
  function buildHubs() {
    if (!linkBy) return;
    const by = new Map();
    for (const n of nodes) if (n.link !== null) (by.get(n.link) ?? by.set(n.link, []).get(n.link)).push(n);
    for (const [key, members] of by) {
      if (members.length < 2) continue;
      const sx = members.reduce((a, n) => a + n.sx, 0) / members.length;
      const sy = members.reduce((a, n) => a + n.sy, 0) / members.length;
      hubs.push({ key, sx, sy, members });
    }
  }

  const hubKey = () => (hover?.hub ? hover.hub.key : hover?.node?.link ?? null);
  // Records outside the hovered link group fade back.
  const dim = (n) => hover && hubKey() !== null && n.link !== hubKey() && n !== hover.node;

  // Map: every member to its hub. Scatter: only the hovered record's group,
  // drawn from that record (a centre means nothing on these axes).
  function drawLinks(c, all) {
    if (!linkBy) return;
    const key = hubKey();
    if (all) {
      for (const h of hubs) {
        const on = key === h.key;
        for (const n of h.members) line(h.sx, h.sy, n.sx, n.sy, on ? c.accent : c.muted, on ? 0.9 : key !== null ? 0.08 : 0.3, on ? 1.5 : 1);
      }
    } else if (key !== null && hover.node) {
      for (const n of nodes) if (n !== hover.node && n.link === key) line(hover.node.sx, hover.node.sy, n.sx, n.sy, c.accent, 0.6, 1);
    }
  }

  function drawNodes(c) {
    // Big gaps last, so they sit on top; the hovered record over everything.
    const order = [...nodes].sort((a, b) => a.rad - b.rad);
    for (const n of order) {
      ctx.globalAlpha = dim(n) ? 0.2 : 1;
      if (n.r.gap === null) ring(n.sx, n.sy, n.rad, c);
      else node(n.sx, n.sy, n.rad, n.r.gap >= 0 ? c.above : c.below, c);
    }
    ctx.globalAlpha = 1;
    const h = hover?.node;
    if (h) {
      ctx.beginPath();
      ctx.arc(h.sx, h.sy, h.rad + 3, 0, Math.PI * 2);
      ctx.strokeStyle = c.fg;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Hub names where there is room for them: few hubs, or zoomed in.
  function hubLabels(c, plot) {
    const shown = hubs.filter((h) => h.sx > plot.x0 && h.sx < plot.x1 && h.sy > plot.y0 && h.sy < plot.y1);
    if (shown.length > 40 * view.k) return;
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const h of shown) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = c.bg;
      ctx.strokeText(h.key, h.sx + 7, h.sy);
      ctx.fillStyle = c.fg;
      ctx.fillText(h.key, h.sx + 7, h.sy);
    }
  }

  function scaleBar(c, plot, pxPerDeg) {
    const pxPerKm = pxPerDeg / KM_PER_DEG;
    const raw = 100 / pxPerKm;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const km = [1, 2, 5, 10].map((m) => m * mag).filter((v) => v <= raw).pop() ?? mag;
    const w = km * pxPerKm, x = plot.x0, y = plot.y1 + 4;
    line(x, y, x + w, y, c.muted, 1, 2);
    line(x, y - 4, x, y + 1, c.muted, 1, 2);
    line(x + w, y - 4, x + w, y + 1, c.muted, 1, 2);
    ctx.fillStyle = c.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${km >= 1 ? km : km.toPrecision(1)} km`, x + w + 6, y);
  }

  // ---------- primitives ----------
  function clip(p, body) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x0 - R_MAX, p.y0 - R_MAX, p.x1 - p.x0 + 2 * R_MAX, p.y1 - p.y0 + 2 * R_MAX);
    ctx.clip();
    body();
    ctx.restore();
  }
  function line(xa, ya, xb, yb, color, alpha = 1, width = 1) {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(xa, ya);
    ctx.lineTo(xb, yb);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  function dashed(xa, ya, xb, yb, color) {
    ctx.setLineDash([5, 4]);
    line(xa, ya, xb, yb, color, 0.8, 1.5);
    ctx.setLineDash([]);
  }
  function ring(cx, cy, r, c) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = c.panel;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.muted;
    ctx.stroke();
  }
  // A filled node with a surface ring, so overlapping nodes stay apart.
  function node(cx, cy, r, color, c) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = c.bg;
    ctx.stroke();
  }
  function hub(cx, cy, c, on) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - 5);
    ctx.lineTo(cx + 5, cy);
    ctx.lineTo(cx, cy + 5);
    ctx.lineTo(cx - 5, cy);
    ctx.closePath();
    ctx.fillStyle = on ? c.accent : c.muted;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = c.bg;
    ctx.stroke();
  }
  function truncate(text, max) {
    if (ctx.measureText(text).width <= max) return text;
    let s = text;
    while (s.length > 1 && ctx.measureText(s + '…').width > max) s = s.slice(0, -1);
    return s + '…';
  }

  // ---------- hover, zoom and pan ----------
  function hitTest(px, py) {
    if (current() === 'list') {
      const i = geom ? Math.floor((py - geom.top) / ROW) : -1;
      return i >= 0 && i < rows.length ? { node: { r: rows[i] } } : null;
    }
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      const d = Math.hypot(n.sx - px, n.sy - py);
      if (d <= Math.max(n.rad + 3, 8) && d < bestD) [best, bestD] = [{ node: n }, d];
    }
    for (const h of hubs) {
      const d = Math.hypot(h.sx - px, h.sy - py);
      if (d <= 8 && d < bestD) [best, bestD] = [{ hub: h }, d];
    }
    return best;
  }

  function nodeTip(r) {
    const row = data.rows[r.index];
    const extra = data.labelFields.filter((f) => f !== data.labelColumn && String(row[f] ?? '').trim() !== '');
    const link = linkBy && linkBy !== data.labelColumn && !extra.includes(linkBy) ? linkOf(r) : null;
    const warn = r.imputed.length || r.missingLocation.length;
    return `<strong>${esc(r.label)}</strong>${r.group ? ` <span class="muted">· ${esc(r.group)} formula</span>` : ''}
      ${extra.length || link !== null ? `<table>${extra.map((f) => `<tr><td class="muted">${esc(f)}</td><td>${esc(row[f])}</td></tr>`).join('')}${link !== null ? `<tr><td class="muted">${esc(linkBy)}</td><td>${esc(link)}</td></tr>` : ''}</table>` : ''}
      <table>
        <tr><td>Estimate</td><td>${fmt(r.estimate)}${warn ? ' ⚠' : ''} ${confidenceBadge(r.confidence)}</td></tr>
        ${r.low !== null ? `<tr><td>Likely range</td><td>${rangeText(r)}</td></tr>` : ''}
        ${r.locationAdj !== null ? `<tr><td>incl. location</td><td>${signed(r.locationAdj)}</td></tr>` : ''}
        ${r.actual !== null ? `<tr><td>Actual</td><td>${fmt(r.actual)}</td></tr>` : ''}
        ${r.gap !== null ? `<tr><td>Gap</td><td>${r.gap >= 0 ? '+' : '−'}${fmt(Math.abs(r.gap))} (${pctText(r.gapPct)})</td></tr>` : ''}
      </table>
      ${warn ? `<div class="warn">⚠ ${esc(missingText(r.imputed, r.missingLocation))}</div>` : ''}
      ${r.suspect?.length ? `<div class="warn">⚑ Left out of the gap figures: ${esc(r.suspect.join('; '))}</div>` : ''}`;
  }

  function hubTip(h) {
    const gaps = h.members.map((n) => n.r.gapPct).filter((g) => g !== null);
    const above = gaps.filter((g) => g >= 0).length;
    return `<strong>${esc(linkBy)} ${esc(h.key)}</strong>
      <table>
        <tr><td>Records</td><td>${h.members.length}</td></tr>
        ${gaps.length ? `<tr><td>Median gap</td><td>${pctText(median(gaps))}</td></tr>
        <tr><td>Median |gap|</td><td>${(median(gaps.map(Math.abs)) * 100).toFixed(1)}%</td></tr>
        <tr><td>Actual above / below</td><td>${above} / ${gaps.length - above}</td></tr>` : ''}
      </table>`;
  }

  function showTip(e, hit) {
    if (!hit) {
      tooltip.hidden = true;
      return;
    }
    tooltip.innerHTML = hit.hub ? hubTip(hit.hub) : nodeTip(hit.node.r);
    tooltip.hidden = false;
    const host = tooltip.offsetParent.getBoundingClientRect();
    const px = e.clientX - host.left, py = e.clientY - host.top;
    tooltip.style.left = `${Math.max(8, Math.min(px + 16, host.width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(py + 16, host.height - tooltip.offsetHeight - 8))}px`;
  }

  const local = (e) => {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  canvas.addEventListener('mousemove', (e) => {
    if (!data) return;
    const [px, py] = local(e);
    if (drag) {
      view.x = drag.vx + px - drag.px;
      view.y = drag.vy + py - drag.py;
      drag.moved = true;
      tooltip.hidden = true;
      draw();
      return;
    }
    const hit = hitTest(px, py);
    const same = hit?.node === hover?.node && hit?.hub === hover?.hub;
    hover = hit && current() !== 'list' ? hit : null;
    if (!same && current() !== 'list') draw();
    showTip(e, hit);
  });
  canvas.addEventListener('mouseleave', () => {
    tooltip.hidden = true;
    drag = null;
    canvas.classList.remove('panning');
    if (hover) {
      hover = null;
      draw();
    }
  });
  canvas.addEventListener('mousedown', (e) => {
    if (!data || current() === 'list' || e.button !== 0) return;
    const [px, py] = local(e);
    drag = { px, py, vx: view.x, vy: view.y };
    canvas.classList.add('panning');
  });
  window.addEventListener('mouseup', () => {
    drag = null;
    canvas.classList.remove('panning');
  });
  canvas.addEventListener('wheel', (e) => {
    if (!data || current() === 'list') return;
    e.preventDefault();
    const [px, py] = local(e);
    const k = Math.min(64, Math.max(1, view.k * Math.exp(-e.deltaY * 0.0015)));
    // Keep the point under the cursor in place. Positions are base·k + offset
    // around a fixed origin, so the offset scales with the zoom.
    const f = k / view.k;
    const o = origin;
    view.x = px - o[0] - (px - o[0] - view.x) * f;
    view.y = py - o[1] - (py - o[1] - view.y) * f;
    view.k = k;
    if (k === 1) view.x = view.y = 0;
    draw();
  }, { passive: false });
  canvas.addEventListener('dblclick', () => {
    resetView();
    draw();
  });

  new ResizeObserver(draw).observe(scroller);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

  return {
    setData(d) {
      const fresh = d?.results !== data?.results;
      data = d;
      if (fresh) resetView();
      renderBar();
      draw();
    },
    redraw: draw,
  };
}

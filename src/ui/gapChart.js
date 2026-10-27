// Gap chart: one row per record that has a real value. A hollow dot marks the
// estimate, a filled dot the actual value, and the line between them is the
// gap (blue = actual above the estimate, orange-red = below). Rows are sorted
// by gap %, which is also printed at the right so colour is never the only cue.
import { fmt, signed, esc, missingText } from './format.js';

const ROW = 28;
const HEAD = 82; // title, legend and the value axis (kept on top so it stays near long lists)
const AXIS = 16;
const LABEL_W = 130;
const PCT_W = 64;

function tokens() {
  const s = getComputedStyle(document.documentElement);
  const v = (n) => s.getPropertyValue(n).trim();
  return { fg: v('--fg'), muted: v('--muted'), border: v('--border'), panel: v('--panel'), bg: v('--bg'), above: v('--above'), below: v('--below') };
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

export function createGapChart(canvas, scroller, tooltip) {
  const ctx = canvas.getContext('2d');
  let data = null;
  let rows = [];
  let geom = null;

  function draw() {
    const width = scroller.clientWidth;
    if (!width) return;
    rows = data ? data.results.filter((r) => r.gap !== null).sort((a, b) => (b.gapPct ?? 0) - (a.gapPct ?? 0)) : [];
    const height = Math.max(scroller.clientHeight, rows.length ? HEAD + rows.length * ROW + AXIS : 0);
    const dpr = window.devicePixelRatio || 1;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = tokens();
    ctx.clearRect(0, 0, width, height);

    const message = (lines) => {
      ctx.fillStyle = c.muted;
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      lines.forEach((l, i) => ctx.fillText(l, width / 2, height / 2 + (i - (lines.length - 1) / 2) * 22));
    };
    if (!data) return message(['Import a CSV under “Predict many records”', 'to compare estimates with real values.']);
    if (!data.actualColumn) return message(['Pick the “Actual value” column', 'in the right panel to see the gaps.']);
    if (!rows.length) return message([`No row has both an estimate and a value in “${data.actualColumn}”.`]);

    // Title + legend.
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
    const item = (draw, text) => {
      draw(lx + 6, legendY - 4);
      ctx.fillStyle = c.muted;
      ctx.fillText(text, lx + 16, legendY);
      lx += 16 + ctx.measureText(text).width + 16;
    };
    if (lx + 480 <= width) {
      item((x, y) => hollow(x, y, c), 'Estimate');
      item((x, y) => solid(x, y, c.fg, c), 'Actual');
      item((x, y) => seg(x - 6, x + 6, y, c.above), 'Actual above estimate');
      item((x, y) => seg(x - 6, x + 6, y, c.below), 'Below');
    }

    // Scale.
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

    // Recessive vertical grid + axis labels above the first row.
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (const t of niceTicks(lo, hi)) {
      const tx = x(t);
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tx, top - 2);
      ctx.lineTo(tx, bottom);
      ctx.stroke();
      ctx.fillStyle = c.muted;
      ctx.fillText(fmt(t), tx, top - 8);
    }

    rows.forEach((r, i) => {
      const y = top + i * ROW + ROW / 2;
      const color = r.gap >= 0 ? c.above : c.below;
      ctx.fillStyle = c.fg;
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(truncate(r.label + (r.imputed.length || r.missingLocation.length ? ' ⚠' : ''), LABEL_W - 10), x0 - 10, y);

      seg(x(r.estimate), x(r.actual), y, color);
      hollow(x(r.estimate), y, c);
      solid(x(r.actual), y, color, c);

      ctx.textAlign = 'right';
      ctx.fillStyle = c.fg;
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.fillText(r.gapPct === null ? '—' : `${r.gapPct >= 0 ? '+' : '−'}${Math.abs(r.gapPct * 100).toFixed(1)}%`, width - pad, y);
    });
  }

  function seg(xa, xb, y, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(xa, y);
    ctx.lineTo(xb, y);
    ctx.stroke();
  }
  function hollow(cx, cy, c) {
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = c.panel;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.muted;
    ctx.stroke();
  }
  function solid(cx, cy, color, c) {
    ctx.beginPath();
    ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.bg;
    ctx.stroke();
  }
  function truncate(text, max) {
    if (ctx.measureText(text).width <= max) return text;
    let s = text;
    while (s.length > 1 && ctx.measureText(s + '…').width > max) s = s.slice(0, -1);
    return s + '…';
  }

  // Hover anywhere on a row shows its numbers.
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const i = geom ? Math.floor((y - geom.top) / ROW) : -1;
    const r = i >= 0 && i < rows.length ? rows[i] : null;
    if (!r) {
      tooltip.hidden = true;
      return;
    }
    tooltip.innerHTML = `<strong>${esc(r.label)}</strong>${r.group ? ` <span class="muted">· ${esc(r.group)} formula</span>` : ''}
      <table>
        <tr><td>Estimate</td><td>${fmt(r.estimate)}${r.imputed.length || r.missingLocation.length ? ' ⚠' : ''}</td></tr>${r.locationAdj !== null ? `<tr><td>incl. location</td><td>${signed(r.locationAdj)}</td></tr>` : ''}
        <tr><td>Actual</td><td>${fmt(r.actual)}</td></tr>
        <tr><td>Gap</td><td>${r.gap >= 0 ? '+' : '−'}${fmt(Math.abs(r.gap))} (${r.gapPct === null ? '—' : (r.gapPct * 100).toFixed(1) + '%'})</td></tr>
      </table>
      ${r.imputed.length || r.missingLocation.length ? `<div class="warn">⚠ ${esc(missingText(r.imputed, r.missingLocation))}</div>` : ''}`;
    tooltip.hidden = false;
    const host = tooltip.offsetParent.getBoundingClientRect();
    const px = e.clientX - host.left, py = e.clientY - host.top;
    tooltip.style.left = `${Math.min(px + 16, host.width - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${Math.min(py + 16, host.height - tooltip.offsetHeight - 8)}px`;
  });
  canvas.addEventListener('mouseleave', () => (tooltip.hidden = true));

  new ResizeObserver(draw).observe(scroller);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

  return {
    setData(d) {
      data = d;
      draw();
    },
    redraw: draw,
  };
}

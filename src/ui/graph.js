// Canvas "link net": every input field is a node with an edge pointing to the
// outcome node. Edge width = effect size, colour = sign of the weight.
import { termImportance, termSign } from '../calc/index.js';
import { fmt, signed, esc } from './format.js';

function palette() {
  const s = getComputedStyle(document.documentElement);
  const v = (n) => s.getPropertyValue(n).trim();
  return {
    fg: v('--fg'), muted: v('--muted'), panel: v('--panel'), border: v('--border'),
    pos: v('--pos'), neg: v('--neg'), enum: v('--enum'), accent: v('--accent'), loc: v('--loc'),
  };
}

// The location adjustment is drawn as one more node, after the inputs.
const LOC_KEY = '\0location';
const keyOf = (t) => (t.kind === 'location' ? LOC_KEY : t.name);

function bezier(p0, p1, p2, p3, t) {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

function pill(ctx, text, x, y, color, c) {
  ctx.font = '600 12px system-ui, sans-serif';
  const w = ctx.measureText(text).width + 12;
  ctx.fillStyle = c.panel;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - 10, w, 20, 10);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5);
}

export function createGraph(canvas, tooltip) {
  const ctx = canvas.getContext('2d');
  let model = null;
  let prediction = null;
  let hits = [];

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height;
    const c = palette();
    ctx.clearRect(0, 0, W, H);
    hits = [];

    if (!model) {
      ctx.fillStyle = c.muted;
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Import a CSV, mark the fields and train —', W / 2, H / 2 - 11);
      ctx.fillText('the weight graph will appear here.', W / 2, H / 2 + 11);
      return;
    }

    const terms = model.location ? [...model.terms, { ...model.location, name: 'location', kind: 'location' }] : model.terms;
    const imps = terms.map((t) => (t.kind === 'location' ? t.spread : termImportance(t)));
    const maxImp = Math.max(...imps, 1e-12);
    const contribByName = Object.fromEntries((prediction?.contributions || []).map((x) => [x.location ? LOC_KEY : x.name, x]));

    const n = terms.length;
    const pad = 48;
    const gap = Math.min(96, (H - pad * 2) / Math.max(n, 1));
    const top = H / 2 - (gap * (n - 1)) / 2;
    const ix = Math.min(Math.max(150, W * 0.26), W * 0.42);
    const ox = Math.max(W - Math.max(110, W * 0.2), ix + 140);
    const oy = H / 2;
    const R = 40;
    const r = Math.max(7, Math.min(13, gap / 4));

    // Edges first so nodes sit on top.
    const nodes = terms.map((t, i) => {
      const y = n === 1 ? oy : top + i * gap;
      const sign = termSign(t);
      const color = t.kind === 'location' ? c.loc : sign > 0 ? c.pos : sign < 0 ? c.neg : c.enum;
      return { t, x: ix, y, color, imp: imps[i] };
    });

    for (const nd of nodes) {
      const x0 = nd.x + r, x3 = ox - R, mx = (x0 + x3) / 2;
      ctx.strokeStyle = nd.color;
      ctx.globalAlpha = 0.75;
      ctx.lineWidth = 1.5 + 12 * (nd.imp / maxImp);
      ctx.beginPath();
      ctx.moveTo(x0, nd.y);
      ctx.bezierCurveTo(mx, nd.y, mx, oy, x3, oy);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Arrow head into the outcome node.
      const ang = Math.atan2(oy - bezier(nd.y, nd.y, oy, oy, 0.97), x3 - bezier(x0, mx, mx, x3, 0.97));
      ctx.fillStyle = nd.color;
      ctx.beginPath();
      ctx.moveTo(x3, oy);
      ctx.lineTo(x3 - 10 * Math.cos(ang - 0.4), oy - 10 * Math.sin(ang - 0.4));
      ctx.lineTo(x3 - 10 * Math.cos(ang + 0.4), oy - 10 * Math.sin(ang + 0.4));
      ctx.fill();

      const t = 0.4;
      const lx = bezier(x0, mx, mx, x3, t), ly = bezier(nd.y, nd.y, oy, oy, t);
      const pc = contribByName[keyOf(nd.t)];
      let label;
      let labelColor = nd.color;
      if (pc) {
        label = signed(pc.contribution);
        labelColor = pc.contribution >= 0 ? c.pos : c.neg;
      } else if (nd.t.kind === 'location') label = `±${fmt(nd.imp / 2)} · ${nd.t.neighbours} nearby`;
      else if (nd.t.kind === 'number') label = `w ${fmt(nd.t.weight)}`;
      else label = `Δ ${fmt(nd.imp)} · ${nd.t.values.length} values`;
      pill(ctx, label, lx, ly, labelColor, c);
    }

    // Input nodes.
    for (const nd of nodes) {
      ctx.fillStyle = nd.color;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.panel;
      ctx.stroke();

      const pc = contribByName[keyOf(nd.t)];
      const sub = nd.t.kind === 'location'
        ? pc ? `${prediction.location.neighbours.length} nearby records` : `${nd.t.points.length} records · ${fmt(nd.t.scaleKm)} km scale`
        : pc
        ? `= ${typeof pc.input === 'number' ? fmt(pc.input) : pc.input}`
        : nd.t.kind === 'number'
          ? `${fmt(nd.t.min)} – ${fmt(nd.t.max)}`
          : `enum · base ${nd.t.baseline}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = c.fg;
      ctx.font = '600 13px system-ui, sans-serif';
      ctx.fillText(truncate(ctx, nd.t.name, ix - r - 16), nd.x - r - 10, nd.y - 1);
      ctx.fillStyle = c.muted;
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(truncate(ctx, sub, ix - r - 16), nd.x - r - 10, nd.y + 14);
      hits.push({ x: nd.x, y: nd.y, r: Math.max(r, 18), term: nd.t, imp: nd.imp });
    }

    // Outcome node.
    ctx.fillStyle = c.accent;
    ctx.beginPath();
    ctx.arc(ox, oy, R, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const centre = prediction ? fmt(prediction.value) : 'ŷ';
    let size = 16;
    do ctx.font = `700 ${size}px system-ui, sans-serif`;
    while (size-- > 9 && ctx.measureText(centre).width > R * 2 - 10);
    ctx.fillText(truncate(ctx, centre, R * 2 - 8), ox, oy);
    ctx.fillStyle = c.fg;
    ctx.font = '600 14px system-ui, sans-serif';
    ctx.fillText(truncate(ctx, model.outcome.name, W - ox + 60), ox, oy - R - 16);
    ctx.fillStyle = c.muted;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(prediction ? `average record ${fmt(prediction.reference)}` : `intercept ${fmt(model.intercept)}`, ox, oy + R + 16);
  }

  function truncate(ctx, text, max) {
    if (ctx.measureText(text).width <= max) return text;
    let s = text;
    while (s.length > 1 && ctx.measureText(s + '…').width > max) s = s.slice(0, -1);
    return s + '…';
  }

  function tooltipHtml({ term: t, imp }) {
    if (t.kind === 'location') {
      const m = t.metrics;
      return `<strong>location</strong> <span class="muted">${esc(t.lat)} / ${esc(t.lng)}</span><br>
        shifts each estimate by how much the formula<br>missed on its ${t.neighbours} nearest training records<br>
        distance scale: ${fmt(t.scaleKm)} km · typical shift ±${fmt(imp / 2)}<br>
        R² ${fmt(m.r2Before)} → ${fmt(m.r2After)} <span class="muted">(left-out check)</span>`;
    }
    if (t.kind === 'number') {
      return `<strong>${esc(t.name)}</strong> <span class="muted">number</span><br>
        weight: <b>${fmt(t.weight)}</b> per unit<br>
        range: ${fmt(t.min)} – ${fmt(t.max)}<br>
        effect across range: ${fmt(imp)}`;
    }
    const rows = Object.entries(t.weights)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([k, w]) => `<tr><td>${esc(k)}${k === t.baseline ? ' <i>(base)</i>' : ''}</td><td>${signed(w)}</td></tr>`)
      .join('');
    const more = t.values.length > 15 ? `<div class="muted">…${t.values.length - 15} more</div>` : '';
    return `<strong>${esc(t.name)}</strong> <span class="muted">enum · ${t.values.length} values</span>
      <table>${rows}</table>${more}`;
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const hit = hits.find((h) => Math.hypot(h.x - x, h.y - y) <= h.r);
    if (!hit) {
      tooltip.hidden = true;
      canvas.style.cursor = '';
      return;
    }
    canvas.style.cursor = 'help';
    tooltip.innerHTML = tooltipHtml(hit);
    tooltip.hidden = false;
    // The tooltip is positioned within <main>, which also holds the view tabs.
    const host = tooltip.offsetParent.getBoundingClientRect();
    const px = e.clientX - host.left, py = e.clientY - host.top;
    tooltip.style.left = `${Math.min(px + 16, host.width - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${Math.min(py + 16, host.height - tooltip.offsetHeight - 8)}px`;
  });
  canvas.addEventListener('mouseleave', () => (tooltip.hidden = true));

  new ResizeObserver(draw).observe(canvas);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

  return {
    setModel(m) {
      model = m;
      prediction = null;
      draw();
    },
    setPrediction(p) {
      prediction = p;
      draw();
    },
  };
}

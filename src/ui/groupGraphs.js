// Weights view for a grouped model: one weight graph per value of the
// group-by field, each drawn from that group's own formula.
import { isThin } from '../calc/index.js';
import { createGraph } from './graph.js';
import { fmt, esc } from './format.js';

export function createGroupGraphs(el, tooltip, { onGroup } = {}) {
  let model = null;
  let graphs = {}; // group value -> graph

  function build() {
    graphs = {};
    if (!model) {
      el.replaceChildren();
      return;
    }
    const cards = Object.entries(model.groups).map(([value, g]) => {
      const card = document.createElement('section');
      card.className = 'group-graph';
      card.dataset.group = value;
      card.innerHTML = `
        <header title="Show this formula in the report and the predict form">
          <b>${esc(model.groupBy)} = ${esc(value)}</b>
          <span class="muted">${g.training.rowsUsed} records · R² ${fmt(g.metrics.r2)}${g.location ? ` · on unseen records ${fmt(g.location.metrics.r2Comps ?? g.location.metrics.r2After)}` : ''} · ${g.terms.length} inputs</span>
          ${isThin(g) ? '<span class="warn" title="Fewer than 2 records per weight: the formula can memorise its records">⚠ few records</span>' : ''}
          ${g.droppedInputs?.length ? `<span class="muted small" title="${esc(g.droppedInputs.map((d) => `${d.name}: ${d.reason}`).join('\n'))}">${g.droppedInputs.length} input(s) not used</span>` : ''}
        </header>
        <canvas aria-label="Weight graph for ${esc(model.groupBy)} = ${esc(value)}"></canvas>`;
      // Room for every input node.
      card.querySelector('canvas').style.height = `${Math.max(260, (g.terms.length + (g.location ? 1 : 0)) * 44 + 96)}px`;
      card.querySelector('header').addEventListener('click', () => onGroup?.(value));
      graphs[value] = createGraph(card.querySelector('canvas'), tooltip);
      graphs[value].setModel(g);
      return card;
    });
    const skipped = model.skipped.map((s) => {
      const card = document.createElement('section');
      card.className = 'group-graph skipped';
      card.innerHTML = `<header><b>${esc(model.groupBy)} = ${esc(s.value)}</b><span class="warn">no formula: ${esc(s.reason)}</span></header>`;
      return card;
    });
    el.replaceChildren(...cards, ...skipped);
  }

  return {
    setModel(next) {
      if (next === model) return;
      model = next;
      build();
    },
    select(group) {
      for (const card of el.querySelectorAll('.group-graph')) card.classList.toggle('selected', card.dataset.group === group);
    },
    // Only the matched group's graph shows the record's contributions.
    setPrediction(result) {
      for (const [value, graph] of Object.entries(graphs)) graph.setPrediction(result && result.group === value ? result : null);
      if (result) el.querySelector(`.group-graph[data-group="${CSS.escape(result.group)}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    },
  };
}

// Trains on every dataset in public/test-data/ and checks the learned weights
//   1. against the exact least-squares solution (is gradient descent converging?)
//   2. against the formula the data was generated from (index.json `truth`).
import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { parseCsv, analyzeColumns, suggestRoles, trainModel, predict } from '../src/calc/index.js';
import { buildSpec, encodeDataset, buildModel } from '../src/calc/model.js';

const dir = new URL('../public/test-data/', import.meta.url);
// Only synthetic datasets have a known generating formula.
const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8')).filter((d) => d.truth);

// Errors are measured as effect across the field's range, relative to the
// outcome's range. Noise and correlated inputs (e.g. car weight vs. cylinders)
// mean even the exact fit differs from the generating formula, hence the
// looser tolerance for `truth`.
const TOL_EXACT = 0.01;
const TOL_TRUTH = 0.1;

// Exact least squares via normal equations (Gauss-Jordan with pivoting).
function solveExact(X, y) {
  const A = X.map((r) => [...r, 1]);
  const d = A[0].length;
  const M = Array.from({ length: d }, (_, i) => {
    const row = new Array(d + 1).fill(0);
    for (let k = 0; k < A.length; k++) {
      for (let j = 0; j < d; j++) row[j] += A[k][i] * A[k][j];
      row[d] += A[k][i] * y[k];
    }
    return row;
  });
  for (let c = 0; c < d; c++) {
    let p = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < d; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= d; k++) M[r][k] -= f * M[c][k];
    }
  }
  const sol = M.map((r, i) => r[d] / r[i]);
  return { weights: sol.slice(0, -1), bias: sol[d - 1] };
}

// Flatten a model into comparable { key: [weight, span] } entries.
function effects(model) {
  const out = {};
  for (const t of model.terms) {
    if (t.kind === 'number') out[t.name] = [t.weight, t.max - t.min];
    else for (const [v, w] of Object.entries(t.weights)) out[`${t.name}=${v}`] = [w, 1];
  }
  return out;
}

describe.each(index)('$file', ({ file, truth }) => {
  it('matches the exact fit and the generating formula', async () => {
    const { columns, rows } = parseCsv(readFileSync(new URL(file, dir), 'utf8'));
    const stats = analyzeColumns(columns, rows);
    const roles = suggestRoles(columns, stats);
    expect(roles[truth.outcome]).toBe('outcome');
    // No ridge: this checks gradient descent against plain least squares.
    const config = { roles, steps: 1500, learningRate: 0.05, optimizer: 'adam', ridge: 0 };

    const model = await trainModel(rows, stats, config);
    const yRange = model.outcome.max - model.outcome.min;
    expect(model.metrics.r2).toBeGreaterThan(0.85);

    const spec = buildSpec(stats, config);
    const { X, y } = encodeDataset(rows, spec);
    const exact = effects(buildModel(spec, solveExact(X, y)));
    for (const [key, [w, span]] of Object.entries(effects(model))) {
      const err = (Math.abs(w - exact[key][0]) * span) / yRange;
      expect(err, `${key}: learned ${w}, exact ${exact[key][0]}`).toBeLessThan(TOL_EXACT);
    }

    for (const t of model.terms) {
      if (t.kind === 'number') {
        const expected = truth.weights?.[t.name];
        if (expected === undefined) continue;
        const err = (Math.abs(t.weight - expected) * (t.max - t.min)) / yRange;
        expect(err, `${t.name}: learned ${t.weight}, true ${expected}`).toBeLessThan(TOL_TRUTH);
      } else {
        const te = truth.enums?.[t.name];
        if (!te) continue;
        for (const [v, w] of Object.entries(t.weights)) {
          const expected = te[v] - te[t.baseline];
          expect(Math.abs(w - expected) / yRange, `${t.name}=${v}: learned ${w}, true ${expected}`).toBeLessThan(TOL_TRUTH);
        }
      }
    }

    // Prediction breakdown adds up: reference + Σ contributions = value.
    const record = Object.fromEntries(model.terms.map((t) => [t.name, t.kind === 'number' ? t.max : t.values.at(-1)]));
    const p = predict(model, record);
    const sum = p.reference + p.contributions.reduce((a, c) => a + c.contribution, 0);
    expect(Math.abs(sum - p.value)).toBeLessThan(1e-6 * Math.max(1, Math.abs(p.value)));
  }, 120000);
});

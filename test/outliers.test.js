import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel } from '../src/calc/index.js';
import { findOutliers, leastSquares } from '../src/calc/outliers.js';

// price = 50k + 200·sqft + 10k·beds + noise, with a seeded generator.
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const noise = () => (rand() + rand() + rand() - 1.5) * 20000;
const clean = Array.from({ length: 300 }, () => {
  const sqft = 800 + Math.round(rand() * 3000), beds = 1 + Math.floor(rand() * 5);
  return { sqft, beds, kind: rand() < 0.5 ? 'a' : 'b', price: 50000 + 200 * sqft + 10000 * beds + noise() };
});
// Bad records: an extra zero on the price, a price typed in thousands, and a
// size typed with an extra zero.
const bad = [
  { sqft: 1500, beds: 3, kind: 'a', price: 3800000 },
  { sqft: 2600, beds: 4, kind: 'b', price: 6200000 },
  { sqft: 2000, beds: 3, kind: 'a', price: 480 },
  { sqft: 21000, beds: 3, kind: 'b', price: 480000 },
];
const rows = [...clean, ...bad];
const columns = ['sqft', 'beds', 'kind', 'price'];
const stats = analyzeColumns(columns, rows);
const config = { roles: { sqft: 'input', beds: 'input', price: 'outcome' }, types: {}, groupBy: null, steps: 1500, learningRate: 0.05, optimizer: 'adam' };

describe('outliers', () => {
  it('solves least squares exactly', () => {
    const X = [[0], [1], [2], [3]], y = [1, 3, 5, 7];
    const { weights, bias } = leastSquares(X, y);
    expect(weights[0]).toBeCloseTo(2, 6);
    expect(bias).toBeCloseTo(1, 6);
  });

  it('flags nothing in clean data with a constant-size error', () => {
    const X = clean.map((r) => [r.sqft / 4000, r.beds / 5]);
    const y = clean.map((r) => r.price / 1e6);
    expect(findOutliers(X, y).length).toBeLessThanOrEqual(1);
  });

  it('leaves out the bad records and recovers the weights', async () => {
    const poisoned = await trainModel(rows, stats, { ...config, outlierCutoff: 0 });
    const m = await trainModel(rows, stats, { ...config, outlierCutoff: 3.5 });
    const flagged = m.training.outliers.map((o) => o.row);
    for (let i = clean.length; i < rows.length; i++) expect(flagged).toContain(i);
    expect(m.training.rowsUsed).toBe(rows.length - flagged.length);
    expect(m.training.outlierCutoff).toBe(3.5);
    const sqft = (model) => model.terms.find((t) => t.name === 'sqft').weight;
    expect(Math.abs(sqft(m) - 200)).toBeLessThan(10);
    expect(Math.abs(sqft(poisoned) - 200)).toBeGreaterThan(Math.abs(sqft(m) - 200) * 3);
    // The worst record comes first, with its actual value and the fit it missed.
    expect(m.training.outliers[0]).toMatchObject({ row: clean.length + 1, actual: 6200000 });
    expect(m.training.outliers[0].z).toBeGreaterThan(3.5);
  });

  it('screens each group on its own records', async () => {
    const m = await trainModel(rows, stats, { ...config, roles: { ...config.roles, kind: 'input' }, groupBy: 'kind', outlierCutoff: 3.5 });
    const byRow = Object.fromEntries(m.training.outliers.map((o) => [o.row, o.group]));
    expect(byRow[clean.length]).toBe('a');
    expect(byRow[clean.length + 1]).toBe('b');
    const perGroup = Object.values(m.groups).reduce((a, g) => a + g.training.outliers.length, 0);
    expect(m.training.outliers).toHaveLength(perGroup);
    expect(m.training.rowsUsed + m.training.outliers.length + m.training.rowsDropped).toBe(rows.length);
  });
});

import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel, predict, predictBatch, useLogInput } from '../src/calc/index.js';

// Seeded so the data (and the auto choice) is the same on every run.
function rng(seed) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

// price = 2000 · sqft^0.7 · noise: each 10% more sqft adds about 6.9%, at any
// size; a straight line in sqft would overshoot the big houses.
function powerData(n = 400) {
  const r = rng(7);
  return Array.from({ length: n }, () => {
    const sqft = Math.round(600 * Math.exp(r() * Math.log(10)));
    const age = Math.round(r() * 60);
    const price = 2000 * sqft ** 0.7 * Math.exp(-0.004 * age) * Math.exp((r() - 0.5) * 0.1);
    return { sqft: String(sqft), age: String(age), price: String(Math.round(price)) };
  });
}

const columns = ['sqft', 'age', 'price'];
const roles = { sqft: 'input', age: 'input', price: 'outcome' };
const base = { roles, steps: 3000, learningRate: 0.05, optimizer: 'adam', logOutcome: true };

describe('log-scale inputs', () => {
  const rows = powerData();
  const stats = analyzeColumns(columns, rows);

  it('auto picks log for a size-like input on a log outcome only', () => {
    expect(useLogInput('sqft', stats, base, rows)).toBe(true);
    // age starts at 0, so it can't be on a log scale.
    expect(useLogInput('age', stats, base, rows)).toBe(false);
    const narrow = rows.map((row) => ({ ...row, sqft: String(1000 + (+row.sqft % 900)) }));
    expect(useLogInput('sqft', analyzeColumns(columns, narrow), base, narrow)).toBe(false);
    expect(useLogInput('sqft', stats, { ...base, scales: { sqft: 'linear' } }, rows)).toBe(false);
    // With a plain outcome a plain input is no runaway: auto leaves it be.
    expect(useLogInput('sqft', stats, { ...base, logOutcome: false }, rows)).toBe(false);
    expect(useLogInput('sqft', stats, { ...base, logOutcome: false, scales: { sqft: 'log' } }, rows)).toBe(true);
  });

  it('learns the elasticity and does not run away on big values', async () => {
    const model = await trainModel(rows, stats, base);
    const sqft = model.terms.find((t) => t.name === 'sqft');
    expect(sqft.log).toBe(true);
    expect(sqft.weight).toBeCloseTo(0.7, 1);
    const at = (s) => predict(model, { sqft: s, age: 20 }).value;
    // Doubling sqft multiplies the price by 2^0.7 ≈ 1.62, small or big.
    expect(at(2000) / at(1000)).toBeCloseTo(2 ** 0.7, 1);
    expect(at(6000) / at(3000)).toBeCloseTo(2 ** 0.7, 1);

    const linear = await trainModel(rows, stats, { ...base, scales: { sqft: 'linear' } });
    expect(linear.terms.find((t) => t.name === 'sqft').log).toBeUndefined();
    expect(model.metrics.rmse).toBeLessThan(linear.metrics.rmse);
  }, 120000);

  it('rejects 0 in a prediction and fills it in a batch', async () => {
    const model = await trainModel(rows, stats, { ...base, steps: 300 });
    expect(() => predict(model, { sqft: 0, age: 10 })).toThrow(/above 0/);
    const { results } = predictBatch(model, [{ sqft: '0', age: '10', price: '500000' }], { inputs: { sqft: 'sqft', age: 'age' }, actual: 'price', label: null });
    expect(results[0].imputed).toEqual(['sqft']);
    expect(Number.isFinite(results[0].estimate)).toBe(true);
  }, 60000);
});

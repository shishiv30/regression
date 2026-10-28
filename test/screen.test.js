import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel, predictBatch } from '../src/calc/index.js';
import { chooseRidge } from '../src/calc/outliers.js';

function rng(seed) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

// price ∝ size^0.3 around a few towns: the value grows far slower than the
// size, as with land, where comparables priced per size unit overshoot.
function landData(n, seed) {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const lat = 37 + Math.floor(r() * 3) * 0.1 + r() * 0.01, lng = -121.9 + r() * 0.01;
    const size = Math.round(2000 * Math.exp(r() * Math.log(500)));
    return { size: String(size), lat: String(lat.toFixed(6)), lng: String(lng.toFixed(6)), price: String(Math.round(80000 * size ** 0.3 * Math.exp((r() - 0.5) * 0.2))) };
  });
}

const columns = ['size', 'lat', 'lng', 'price'];
const roles = { size: 'input', lat: 'lat', lng: 'lng', price: 'outcome' };
const config = { roles, steps: 800, learningRate: 0.05, optimizer: 'adam', logOutcome: true, compsBy: 'auto', outlierCutoff: 3.5 };

describe('comparables scale by size, batch screen', () => {
  const rows = landData(300, 3);
  const stats = analyzeColumns(columns, rows);
  let model;

  it('learns how value grows with size for comparables', async () => {
    model = await trainModel(rows, stats, config);
    expect(model.location.size).toBe('size');
    expect(model.location.sizePower).toBeGreaterThan(0.15);
    expect(model.location.sizePower).toBeLessThan(0.5);
    // A big parcel among small ones is not priced as many small ones.
    const big = { size: '900000', lat: '37.005', lng: '-121.895' };
    const { results } = predictBatch(model, [{ ...big, price: String(Math.round(80000 * 900000 ** 0.3)) }], { inputs: { size: 'size', lat: 'lat', lng: 'lng' }, actual: 'price', label: null });
    expect(Math.abs(results[0].gapPct)).toBeLessThan(0.3);
  });

  it('leaves implausible actual values and far-out inputs out of the gap figures', () => {
    const fresh = landData(40, 9);
    const typo = { ...fresh[0], price: String(+fresh[0].price * 100) };
    const huge = { ...fresh[1], size: String(stats.size.max * 3) };
    const out = predictBatch(model, [...fresh.slice(2), typo, huge], { inputs: { size: 'size', lat: 'lat', lng: 'lng' }, actual: 'price', label: null });
    const [a, b] = out.results.slice(-2);
    expect(a.suspect.join()).toMatch(/σ above/);
    expect(b.suspect.join()).toMatch(/far outside the training range/);
    expect(Number.isFinite(a.estimate)).toBe(true);
    expect(out.summary.screened).toBe(2);
    expect(out.summary.withActual).toBe(38);
    expect(out.results.slice(0, -2).every((r) => !r.suspect.length)).toBe(true);
  });
});

describe('ridge', () => {
  it('stays near 0 with plenty of records and grows when weights outnumber them', () => {
    const r = rng(5);
    const make = (n, d) => {
      const X = Array.from({ length: n }, () => Array.from({ length: d }, () => (r() < 0.2 ? 1 : 0)));
      // Only the first column matters; the rest is noise to memorise.
      return { X, y: X.map((x) => 0.5 * x[0] + (r() - 0.5) * 0.3) };
    };
    const many = make(2000, 5), few = make(30, 25);
    expect(chooseRidge(many.X, many.y)).toBeLessThanOrEqual(1e-3);
    expect(chooseRidge(few.X, few.y)).toBeGreaterThan(1e-3);
  });
});

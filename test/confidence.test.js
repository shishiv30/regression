import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseCsv, analyzeColumns, suggestRoles, trainModel, predict, suggestMapping, predictBatch, resultsToCsv, useLogOutcome, validateConfig } from '../src/calc/index.js';

const load = (f) => parseCsv(readFileSync(new URL(`../public/test-data/${f}`, import.meta.url), 'utf8'));
const train = load('location-train.csv');
const verify = load('location-verify.csv');
const stats = analyzeColumns(train.columns, train.rows);
const roles = suggestRoles(train.columns, stats);
const base = { types: {}, groupBy: null, steps: 1500, learningRate: 0.05, optimizer: 'adam', neighbours: 10, roles };

describe('log scale', () => {
  it('turns on for a positive outcome that spans 10× and validates', () => {
    const s = { y: { type: 'number', min: 1, max: 50 }, x: { type: 'number', min: 0, max: 1 } };
    const r = { y: 'outcome', x: 'input' };
    expect(useLogOutcome(s, { roles: r, logOutcome: 'auto' })).toBe(true);
    expect(useLogOutcome({ ...s, y: { ...s.y, max: 5 } }, { roles: r, logOutcome: 'auto' })).toBe(false);
    expect(useLogOutcome(s, { roles: r })).toBe(false);
    expect(validateConfig({ ...s, y: { ...s.y, min: 0 } }, { roles: r, logOutcome: true })).toMatch(/Log scale/);
  });

  it('estimates in outcome units and the breakdown still adds up', async () => {
    // sqft kept linear: this checks the per-unit meaning of a weight.
    const m = await trainModel(train.rows, stats, { ...base, logOutcome: true, scales: { sqft: 'linear' } });
    expect(m.outcome.log).toBe(true);
    const p = predict(m, { sqft: 2000, beds: 3, latitude: 30.27, longitude: -97.74 });
    expect(p.value).toBeGreaterThan(100000);
    const sum = p.reference + p.contributions.reduce((a, c) => a + c.contribution, 0);
    expect(sum).toBeCloseTo(p.value, 4);
    // Each part also carries its own percentage effect.
    expect(p.contributions.find((c) => c.name === 'sqft').pct).toBeTypeOf('number');
    // A bigger house is estimated higher by the sqft weight's percentage.
    const big = predict(m, { sqft: 2100, beds: 3 }, { location: false }).value;
    const small = predict(m, { sqft: 2000, beds: 3 }, { location: false }).value;
    expect(big / small).toBeCloseTo(Math.exp(100 * m.terms[0].weight), 6);
  });
});

describe('comparables, confidence and ranges', () => {
  let model;
  it('picks the size field and blends comparables in', async () => {
    model = await trainModel(train.rows, stats, { ...base, logOutcome: 'auto', compsBy: 'auto' });
    expect(model.location.size).toBe('sqft');
    expect(model.location.metrics.r2Comps).toBeTypeOf('number');
    const p = predict(model, { sqft: 2000, beds: 3, latitude: 30.27, longitude: -97.74 });
    expect(p.comps.neighbours).toHaveLength(10);
    expect(p.contributions.at(-1)).toMatchObject({ name: 'comparables', comps: true });
    expect(p.value).toBeCloseTo(Math.sqrt(p.formula * p.comps.value), 6);
    expect(p.confidence.level).toMatch(/high|medium|low/);
    expect(p.confidence.low).toBeLessThan(p.value);
    expect(p.confidence.high).toBeGreaterThan(p.value);
  });

  it('gives an honest range on held-out records, tighter where confidence is high', () => {
    const { results, summary } = predictBatch(model, verify.rows, suggestMapping(model, verify.columns, verify.rows));
    // 80% ranges: most held-out actual values fall inside.
    expect(summary.inRange).toBeGreaterThan(0.65);
    expect(summary.inRange).toBeLessThan(0.97);
    const width = (level) => {
      const rs = results.filter((r) => r.confidence === level);
      return rs.reduce((a, r) => a + (r.high - r.low) / r.estimate, 0) / rs.length;
    };
    expect(width('high')).toBeLessThan(width('low'));
    // Rows without coordinates still get a range, but no confidence level.
    const noCoords = results.find((r) => r.missingLocation.length);
    expect(noCoords.low).not.toBeNull();
    expect(noCoords.confidence).toBeNull();
    const head = resultsToCsv(verify.columns, verify.rows, results, 'price').split('\n')[0];
    expect(head).toContain('est_price,est_low,est_high,confidence,location_adj,comparables_adj,gap');
  });

  it('has a training-error range without a location', async () => {
    const m = await trainModel(train.rows, stats, { ...base, steps: 300, roles: { sqft: 'input', beds: 'input', price: 'outcome' } });
    expect(m.range).toMatchObject({ mode: 'ratio', source: 'left-out' });
    const p = predict(m, { sqft: 2000, beds: 3 });
    expect(p.confidence).toMatchObject({ level: null });
    expect(p.confidence.low).toBeLessThan(p.value);
  });
});

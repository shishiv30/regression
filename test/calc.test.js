import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { parseCsv, analyzeColumns, suggestRoles, trainModel, predict, validateConfig } from '../src/calc/index.js';

const csv = readFileSync(new URL('../public/sample-housing.csv', import.meta.url), 'utf8');

describe('schema', () => {
  const { columns, rows } = parseCsv(csv);
  const stats = analyzeColumns(columns, rows);

  it('detects numbers with ranges and enums with dictionaries', () => {
    expect(stats.sqft.type).toBe('number');
    expect(stats.sqft.min).toBeGreaterThanOrEqual(600);
    expect(stats.sqft.max).toBeLessThanOrEqual(3000);
    expect(stats.neighborhood.type).toBe('enum');
    expect([...stats.neighborhood.values].sort()).toEqual(['Downtown', 'Rural', 'Suburb']);
  });

  it('suggests roles and validates config', () => {
    const roles = suggestRoles(columns, stats);
    expect(roles).toMatchObject({ id: 'ignore', price: 'outcome', sqft: 'input', neighborhood: 'input' });
    expect(validateConfig(stats, { roles })).toBeNull();
    expect(validateConfig(stats, { roles: { ...roles, price: 'input' } })).toMatch(/outcome/);
  });

  it('honours forced enum type for numeric columns', () => {
    const s = analyzeColumns(columns, rows, { bedrooms: 'enum' });
    expect(s.bedrooms.type).toBe('enum');
    expect(s.bedrooms.values).toHaveLength(5);
  });
});

describe('training', () => {
  it('recovers the generating weights (150·sqft + 10000·beds − 800·age + hood)', async () => {
    const { columns, rows } = parseCsv(csv);
    const stats = analyzeColumns(columns, rows);
    const roles = suggestRoles(columns, stats);
    const model = await trainModel(rows, stats, { roles, steps: 3000, learningRate: 0.05, optimizer: 'adam' });

    const t = Object.fromEntries(model.terms.map((x) => [x.name, x]));
    expect(t.sqft.weight).toBeCloseTo(150, -1);
    expect(t.bedrooms.weight).toBeGreaterThan(7000);
    expect(t.bedrooms.weight).toBeLessThan(13000);
    expect(t.age_years.weight).toBeLessThan(-500);
    expect(t.neighborhood.weights.Downtown - t.neighborhood.weights.Rural).toBeGreaterThan(65000);
    expect(model.metrics.r2).toBeGreaterThan(0.9);

    const p = predict(model, { sqft: 2000, bedrooms: 3, age_years: 10, neighborhood: 'Suburb' });
    const truth = 50000 + 150 * 2000 + 10000 * 3 - 800 * 10 + 30000;
    expect(Math.abs(p.value - truth) / truth).toBeLessThan(0.05);
    expect(p.contributions).toHaveLength(4);
  }, 60000);
});

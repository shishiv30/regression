import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { parseCsv, analyzeColumns, suggestRoles, trainModel, suggestMapping, predictBatch, resultsToCsv, predict } from '../src/calc/index.js';

const read = (path) => parseCsv(readFileSync(new URL(path, import.meta.url), 'utf8'));

async function train(path, steps = 800) {
  const { columns, rows } = read(path);
  const stats = analyzeColumns(columns, rows);
  return trainModel(rows, stats, { roles: suggestRoles(columns, stats), steps, learningRate: 0.05, optimizer: 'adam' });
}

describe('batch prediction (synthetic)', () => {
  // Hand-made model: y = 100 + 2·a + {x: 0, y: 50}[b]
  const model = {
    outcome: { name: 'y', min: 0, max: 1000 },
    intercept: 100,
    terms: [
      { name: 'a', kind: 'number', weight: 2, min: 0, max: 100, mean: 40 },
      { name: 'b', kind: 'enum', baseline: 'x', values: ['x', 'y'], weights: { x: 0, y: 50 }, freq: { x: 0.5, y: 0.5 } },
    ],
  };

  it('suggests same-named inputs, a price-like actual column and a label column', () => {
    const m = suggestMapping(model, ['a', 'kind', 'closePrice', 'mlsNumber']);
    expect(m).toEqual({ inputs: { a: 'a', b: null }, actual: 'closePrice', label: 'mlsNumber' });
    expect(suggestMapping(model, ['a', 'b', 'y']).actual).toBe('y');
  });

  it('estimates every row, fills gaps with training averages and measures the gap', () => {
    const rows = [
      { a: '10', kind: 'y', sold: '200', id: 'A' }, // 100 + 20 + 50 = 170, gap +30
      { a: '', kind: 'x', sold: '150', id: 'B' }, // a → mean 40: 100 + 80 = 180, gap −30
      { a: '50', kind: 'x', sold: '', id: 'C' }, // 200, no actual
    ];
    const { results, summary } = predictBatch(model, rows, { inputs: { a: 'a', b: 'kind' }, actual: 'sold', label: 'id' });
    expect(results.map((r) => r.estimate)).toEqual([170, 180, 200]);
    expect(results.map((r) => r.gap)).toEqual([30, -30, null]);
    expect(results[0].gapPct).toBeCloseTo(0.15);
    expect(results[1].imputed).toEqual(['a']);
    expect(results.map((r) => r.label)).toEqual(['A', 'B', 'C']);
    expect(summary).toMatchObject({ rows: 3, withActual: 2, imputedRows: 1, meanGap: 0, mae: 30 });
    expect(summary.within10).toBe(0); // 15% and 20%

    const csv = resultsToCsv(['a', 'kind', 'sold', 'id'], rows, results, 'y');
    expect(csv.split(/\r?\n/)[0]).toBe('a,kind,sold,id,est_y,gap,gap_pct,imputed_fields,note');
    expect(csv.split(/\r?\n/)[2]).toContain('180.00,-30.00,-20.00,a');
  });

  it('matches single-record predict for complete rows', () => {
    const { results } = predictBatch(model, [{ a: '33', b: 'y' }], suggestMapping(model, ['a', 'b']));
    expect(results[0].estimate).toBe(predict(model, { a: '33', b: 'y' }).value);
  });
});

describe('Austin: train on for-sale, verify on sold', () => {
  it('estimates all 19 sold listings and compares the 3 with a closePrice', async () => {
    const model = await train('./test-austin-training-data.csv');
    expect(model.outcome.name).toBe('listPrice');

    const sold = read('./test-austin-verify-data.csv');
    const mapping = suggestMapping(model, sold.columns);
    expect(mapping.actual).toBe('closePrice');
    expect(mapping.label).toBe('mlsNumber');
    expect(mapping.inputs.listXdaysago).toBeNull(); // the sold file has soldXdaysago instead

    const { results, summary } = predictBatch(model, sold.rows, mapping);
    expect(results).toHaveLength(19);
    expect(results.every((r) => Number.isFinite(r.estimate))).toBe(true);
    expect(summary.withActual).toBe(3);
    // LAND rows have no beds/baths/sqft and get training averages.
    expect(results.find((r) => r.label === '20983408').imputed).toEqual(expect.arrayContaining(['bedrooms', 'sqftTotal']));
  }, 60000);
});

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

  it('suggests same-named inputs, and an actual and a label column by their values', () => {
    // Column names say nothing: 'v2' holds values in the outcome range (0–1000),
    // 'v1' far outside it, 'tag' a different text on every row.
    const rows = Array.from({ length: 12 }, (_, i) => ({ a: String(i), kind: i % 2 ? 'p' : 'q', v1: String(5000 + i), v2: String(200 + 30 * i), tag: `R-${i}` }));
    const m = suggestMapping(model, ['a', 'kind', 'v1', 'v2', 'tag'], rows);
    expect(m).toEqual({ inputs: { a: 'a', b: null }, actual: 'v2', label: 'tag' });
    expect(suggestMapping(model, ['a', 'b', 'y']).actual).toBe('y');
    // A field marked Label in training wins.
    expect(suggestMapping({ ...model, labels: ['kind'] }, ['a', 'kind', 'v2', 'tag'], rows).label).toBe('kind');
  });

  it('does not take a code that sits inside the outcome range for the actual value', () => {
    // A ZIP-like column lies wholly inside a price range but barely varies;
    // the price column varies like the training prices, with a few outside.
    const priced = { ...model, outcome: { name: 'closePrice', min: 50000, max: 9000000 } };
    const rows = Array.from({ length: 200 }, (_, i) => ({ zip: String(95002 + (i % 140)), listPrice: String(i < 3 ? 12000000 : 300000 + 25000 * i) }));
    expect(suggestMapping(priced, ['zip', 'listPrice'], rows).actual).toBe('listPrice');
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

// Real data, thousands of rows: SKIP_HEAVY=1 (the deploy pipeline) skips it.
describe.skipIf(!!process.env.SKIP_HEAVY)('San Jose: train on sold, verify on for-sale', () => {
  it('estimates every for-sale listing and compares each listPrice', async () => {
    const model = await train('../public/test-data/test-san-jose-training-data.csv');
    expect(model.outcome.name).toBe('closePrice');
    expect(model.labels).toContain('mlsNumber'); // text ids are labels, not inputs

    const forSale = read('../public/test-data/test-san-jose-verify-data.csv');
    const mapping = suggestMapping(model, forSale.columns, forSale.rows);
    expect(mapping.actual).toBe('listPrice');
    expect(mapping.label).toBe('mlsNumber');
    expect(mapping.inputs.soldXdaysago).toBeNull(); // the for-sale file has listXdaysago instead

    const { results, summary } = predictBatch(model, forSale.rows, mapping);
    expect(results).toHaveLength(forSale.rows.length);
    expect(results.every((r) => Number.isFinite(r.estimate))).toBe(true);
    expect(summary.withActual).toBeGreaterThan(0.99 * forSale.rows.length);
    expect(summary.medianApe).toBeLessThan(0.25);
  }, 120000);
});

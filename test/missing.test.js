import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel, predict, predictBatch } from '../src/calc/index.js';

// price = 100·sqft + 20000·beds (+ 50000 for CONDO); LAND has beds on only 1 in 4 records.
const rows = [];
for (let i = 0; i < 120; i++) {
  const type = ['HOUSE', 'CONDO', 'LAND'][i % 3];
  const sqft = 800 + ((i * 37) % 2000);
  const beds = 1 + (i % 5);
  const blank = type === 'LAND' ? i % 12 !== 2 : i % 10 === 0; // LAND 75% blank, others 10%
  rows.push({ type, sqft: String(sqft), beds: blank ? '' : String(beds), price: String(100 * sqft + (type === 'LAND' ? 0 : 20000 * beds) + (type === 'CONDO' ? 50000 : 0)) });
}
const columns = ['type', 'sqft', 'beds', 'price'];
const stats = analyzeColumns(columns, rows);
const roles = { type: 'input', sqft: 'input', beds: 'input', price: 'outcome' };
const base = { roles, types: {}, steps: 400, learningRate: 0.05, optimizer: 'adam' };

describe('missing values', () => {
  it('drops records with a blank input by default', async () => {
    const m = await trainModel(rows, stats, base);
    expect(m.training.rowsUsed).toBe(rows.filter((r) => r.beds !== '').length);
    expect(m.training.rowsFilled).toBe(0);
    expect(() => predict(m, { type: 'HOUSE', sqft: 1000, beds: '' })).toThrow(/required/);
  });

  it('fills blanks with the median and keeps every record', async () => {
    const m = await trainModel(rows, stats, { ...base, missing: 'median' });
    expect(m.training.rowsUsed).toBe(rows.length);
    expect(m.training.rowsFilled).toBe(rows.filter((r) => r.beds === '').length);
    const beds = m.terms.find((t) => t.name === 'beds');
    expect(beds.fill).toBe(stats.beds.median);
    // A blank in prediction and in batch takes the same median.
    const p = predict(m, { type: 'HOUSE', sqft: 1000, beds: '' });
    expect(p.filled).toEqual(['beds']);
    expect(p.value).toBeCloseTo(predict(m, { type: 'HOUSE', sqft: 1000, beds: beds.fill }).value, 6);
    const { results } = predictBatch(m, [{ type: 'HOUSE', sqft: '1000', beds: '' }], { inputs: { type: 'type', sqft: 'sqft', beds: 'beds' }, actual: null, label: null });
    expect(results[0].estimate).toBeCloseTo(p.value, 6);
  });

  it('leaves a field out of a group where most records have it blank', async () => {
    for (const missing of ['drop', 'median']) {
      const m = await trainModel(rows, stats, { ...base, missing, groupBy: 'type' });
      expect(m.groups.LAND.droppedInputs).toEqual([{ name: 'beds', reason: '75% blank in this group' }]);
      expect(m.groups.LAND.terms.map((t) => t.name)).toEqual(['sqft']);
      // LAND keeps all its records: beds no longer drops them.
      expect(m.groups.LAND.training.rowsUsed).toBe(40);
      expect(m.groups.HOUSE.terms.map((t) => t.name)).toContain('beds');
    }
  });
});

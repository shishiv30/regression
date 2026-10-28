import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel, predict, validateConfig, suggestMapping, predictBatch, modelInputs, MIN_GROUP_ROWS } from '../src/calc/index.js';

// Three kinds with different formulas:
//   house: y = 100 + 5·x            (z is always 0 for houses → no weight to learn)
//   condo: y = 1000 − 3·x + 50·z
//   land:  only 5 records → too few for its own formula
let seed = 3;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const rows = [];
for (let i = 0; i < 120; i++) {
  const x = Math.round(rand() * 100), z = Math.round(rand() * 10);
  rows.push({ kind: 'house', x: String(x), z: '0', y: String(100 + 5 * x + (rand() - 0.5)) });
  rows.push({ kind: 'condo', x: String(x), z: String(z), y: String(1000 - 3 * x + 50 * z + (rand() - 0.5)) });
}
for (let i = 0; i < 5; i++) rows.push({ kind: 'land', x: String(i * 10), z: '1', y: String(50 + i) });

const columns = ['kind', 'x', 'z', 'y'];
const stats = analyzeColumns(columns, rows);
const config = { roles: { kind: 'input', x: 'input', z: 'input', y: 'outcome' }, groupBy: 'kind', steps: 2000, learningRate: 0.05, optimizer: 'adam' };

describe('group-by training', () => {
  it('validates the group-by field', () => {
    expect(validateConfig(stats, config)).toBeNull();
    expect(validateConfig(stats, { ...config, groupBy: 'x' })).toMatch(/must be an enum/);
    expect(validateConfig(stats, { ...config, roles: { kind: 'input', y: 'outcome' } })).toMatch(/at least one field as an input/);
  });

  it('trains one formula per group on that group only', async () => {
    const model = await trainModel(rows, stats, config);
    expect(Object.keys(model.groups).sort()).toEqual(['condo', 'house']);

    const house = model.groups.house;
    const condo = model.groups.condo;
    const w = (m, name) => m.terms.find((t) => t.name === name)?.weight;

    expect(house.terms.map((t) => t.name)).toEqual(['x']); // z dropped, kind is the group
    expect(house.droppedInputs).toEqual([{ name: 'z', reason: 'same value for every record' }]);
    expect(w(house, 'x')).toBeCloseTo(5, 1);
    expect(house.intercept).toBeCloseTo(100, 0);

    expect(w(condo, 'x')).toBeCloseTo(-3, 1);
    expect(w(condo, 'z')).toBeCloseTo(50, 0);
    expect(condo.intercept).toBeCloseTo(1000, -1);

    expect(model.skipped).toEqual([{ value: 'land', rows: 5, reason: `only 5 complete records (needs ${MIN_GROUP_ROWS})` }]);
    expect(model.metrics.r2).toBeGreaterThan(0.999);
    expect(model.training.rowsUsed).toBe(240);

    // Prediction routes by group.
    expect(predict(model, { kind: 'house', x: 10, z: 7 }).value).toBeCloseTo(150, 0);
    expect(predict(model, { kind: 'condo', x: 10, z: 2 }).value).toBeCloseTo(1070, 0);
    expect(predict(model, { kind: 'condo', x: 10, z: 2 }).group).toBe('condo');
    expect(() => predict(model, { kind: 'land', x: 1, z: 1 })).toThrow(/only 5 complete records/);
    expect(() => predict(model, { kind: 'boat', x: 1, z: 1 })).toThrow(/not in the training data/);

    // Inputs: group field first, then the union of every group's inputs.
    expect(modelInputs(model).map((t) => t.name)).toEqual(['kind', 'x', 'z']);

    // Batch: each row uses its group's formula; unknown groups get no estimate.
    const batch = [
      { type: 'condo', x: '20', z: '', sold: '1300' },
      { type: 'house', x: '20', z: '', sold: '' },
      { type: 'land', x: '20', z: '1', sold: '90' },
    ];
    const mapping = { ...suggestMapping(model, ['type', 'x', 'z', 'sold']), actual: 'sold' };
    expect(mapping.inputs).toEqual({ kind: null, x: 'x', z: 'z' });
    mapping.inputs.kind = 'type';
    const { results, summary } = predictBatch(model, batch, mapping);
    expect(results[0].group).toBe('condo');
    expect(results[0].imputed).toEqual(['z']); // condo's own average z is used
    expect(results[1].estimate).toBeCloseTo(200, 0);
    expect(results[1].imputed).toEqual([]); // house formula has no z input
    expect(results[2].estimate).toBeNull();
    expect(results[2].error).toMatch(/land/);
    expect(summary).toMatchObject({ rows: 3, withActual: 1, noEstimate: 1 });
  }, 60000);

  it('honours the minimum records per group and rejects groups with too few records per weight', async () => {
    const loose = await trainModel(rows, stats, { ...config, minGroupRows: 5 });
    expect(Object.keys(loose.groups).sort()).toEqual(['condo', 'house', 'land']);

    // 12 records, but an enum with 8 values: 9 weights need 18 records.
    const tag = (i) => String.fromCharCode(97 + (i % 8));
    const wide = [...rows, ...Array.from({ length: 12 }, (_, i) => ({ kind: 'lot', x: String(i), z: tag(i), y: String(10 + i) }))];
    const cols = ['kind', 'x', 'z', 'y'];
    const m = await trainModel(wide, analyzeColumns(cols, wide, { z: 'enum' }), { ...config, types: { z: 'enum' }, minGroupRows: 10 });
    expect(m.skipped.find((g) => g.value === 'lot').reason).toMatch(/12 complete records for 9 weights/);
  }, 60000);
});

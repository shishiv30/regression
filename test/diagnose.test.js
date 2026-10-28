import { describe, it, expect } from 'vitest';
import { analyzeColumns, trainModel, predictBatch, diagnoseGaps } from '../src/calc/index.js';
import { signedRankZ } from '../src/calc/batch.js';

function rng(seed) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

// y = 100 + 3·size + noise. In training, `leak` follows y (as a field
// filled in after the fact would); in the verify file it is random, so a
// formula leaning on it misses there. `tag` is rare in training for 'z'.
function data(n, seed, leaky) {
  const r = rng(seed);
  return Array.from({ length: n }, (_, i) => {
    const size = 10 + r() * 90;
    const y = 100 + 3 * size + (r() - 0.5) * 60;
    const leak = leaky ? y / 10 + (r() - 0.5) * 2 : 10 + r() * 40;
    return { size: size.toFixed(2), leak: leak.toFixed(2), tag: leaky && i < 3 ? 'z' : ['a', 'b'][i % 2], y: y.toFixed(2) };
  });
}

describe('gap diagnosis', () => {
  it('signed-rank z leans with the differences', () => {
    const r = rng(1);
    const up = Array.from({ length: 60 }, () => r() * 0.1 + 0.02);
    expect(signedRankZ(up)).toBeGreaterThan(5);
    expect(signedRankZ(up.map((v) => -v))).toBeLessThan(-5);
    expect(Math.abs(signedRankZ(up.map((v, i) => (i % 2 ? v : -v))))).toBeLessThan(1.96);
  });

  it('finds a field that hurts on new records and one that helps', async () => {
    const train = data(300, 3, true), verify = data(200, 4, false);
    const columns = ['size', 'leak', 'tag', 'y'];
    const stats = analyzeColumns(columns, train);
    const model = await trainModel(train, stats, { roles: { size: 'input', leak: 'input', tag: 'input', y: 'outcome' }, steps: 1500, learningRate: 0.05, optimizer: 'adam', ridge: 0 });
    const mapping = { inputs: { size: 'size', leak: 'leak', tag: 'tag' }, actual: 'y', label: null };
    const { results } = predictBatch(model, verify, mapping);
    const d = diagnoseGaps(model, verify, mapping, results);
    const f = Object.fromEntries(d.fields.map((x) => [x.name, x]));
    expect(f.leak.verdict).toBe('hurts');
    expect(f.leak.suggest).toBe(true);
    expect(f.leak.without.medianApe).toBeLessThan(d.before.medianApe);
    // It drives nearly every miss, big or not, so it is no big-gap pattern;
    // its verify values outside the training range are.
    expect(f.leak.driver.big).toBeGreaterThan(0.9);
    expect(f.leak.unusualPattern).toBe(true);
    expect(f.size.verdict).toBe('helps');
    expect(d.fields[0].name).toBe('leak');
    expect(d.rows).toBe(200);
    expect(d.bigRows).toBe(20);
    expect(diagnoseGaps(model, verify.slice(0, 10), mapping, results.slice(0, 10))).toBeNull();
  });
});

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseCsv, analyzeColumns, suggestRoles, validateConfig, trainModel, predict, suggestMapping, predictBatch, modelInputs, resultsToCsv } from '../src/calc/index.js';
import { distanceKm } from '../src/calc/location.js';

const load = (f) => parseCsv(readFileSync(new URL(`../public/test-data/${f}`, import.meta.url), 'utf8'));
const train = load('location-train.csv');
const verify = load('location-verify.csv');
const truth = JSON.parse(readFileSync(new URL('../public/test-data/index.json', import.meta.url), 'utf8')).find((d) => d.file === 'location-train.csv').truth;
const stats = analyzeColumns(train.columns, train.rows);
const base = { types: {}, groupBy: null, steps: 1500, learningRate: 0.05, optimizer: 'adam', neighbours: 10 };

// The neighbourhood effect the generator added at a point.
const kmTo = (s, lat, lng) => distanceKm(lat, lng, s.lat, s.lng);
const effect = (lat, lng) => truth.location.reduce((a, s) => a + s.amount * Math.exp(-0.5 * (kmTo(s, lat, lng) / s.km) ** 2), 0);

describe('location adjustment', () => {
  it('measures distance in km', () => {
    // 0.01° of latitude ≈ 1.1 km; 0.01° of longitude at 30° N ≈ 0.96 km.
    expect(distanceKm(30, -97, 30.01, -97)).toBeCloseTo(1.106, 2);
    expect(distanceKm(30, -97, 30, -96.99)).toBeCloseTo(0.964, 2);
  });

  it('picks latitude/longitude columns as the location', () => {
    const roles = suggestRoles(train.columns, stats);
    expect(roles).toMatchObject({ latitude: 'lat', longitude: 'lng', price: 'outcome', sqft: 'input', beds: 'input' });
  });

  it('validates the location fields', () => {
    const roles = { sqft: 'input', beds: 'input', price: 'outcome' };
    expect(validateConfig(stats, { ...base, roles: { ...roles, latitude: 'lat', longitude: 'lng' } })).toBeNull();
    expect(validateConfig(stats, { ...base, roles: { ...roles, latitude: 'lat' } })).toMatch(/longitude field too/);
  });

  let model;
  it('keeps the formula and improves the held-out fit', async () => {
    model = await trainModel(train.rows, stats, { ...base, roles: suggestRoles(train.columns, stats) });
    // Lat/lng are not formula inputs.
    expect(model.terms.map((t) => t.name)).toEqual(['sqft', 'beds']);
    expect(model.location).toMatchObject({ lat: 'latitude', lng: 'longitude', neighbours: 10 });
    expect(model.location.points).toHaveLength(800);
    const m = model.location.metrics;
    expect(m.r2After).toBeGreaterThan(m.r2Before);
    expect(m.rmseAfter).toBeLessThan(m.rmseBefore * 0.8);
    // The formula is fitted on the formula fields only, so the sqft weight is
    // close to the truth even though location is left out of it.
    expect(model.terms[0].weight).toBeGreaterThan(truth.weights.sqft * 0.85);
    expect(model.terms[0].weight).toBeLessThan(truth.weights.sqft * 1.15);
  });

  it('shifts a record towards its neighbourhood', () => {
    const spot = truth.location[0]; // pricey spot
    const dip = truth.location[2]; // cheap spot
    const at = (s) => predict(model, { sqft: 2000, beds: 3, latitude: s.lat, longitude: s.lng });
    const hi = at(spot), lo = at(dip);
    expect(hi.location.adjustment).toBeGreaterThan(0.5 * (effect(spot.lat, spot.lng) - 30000));
    expect(lo.location.adjustment).toBeLessThan(0);
    // The adjustment is one more contribution, so the breakdown still adds up.
    const sum = hi.reference + hi.contributions.reduce((a, c) => a + c.contribution, 0);
    expect(sum).toBeCloseTo(hi.value, 6);
    expect(hi.contributions.at(-1)).toMatchObject({ name: 'location', location: true });
    expect(hi.location.neighbours).toHaveLength(10);
    expect(hi.location.neighbours[0].distanceKm).toBeLessThan(1);
  });

  it('fades out far from every training record', () => {
    const far = predict(model, { sqft: 2000, beds: 3, latitude: 31.5, longitude: -99 });
    expect(Math.abs(far.location.adjustment)).toBeLessThan(1);
  });

  it('skips the adjustment when coordinates are missing', () => {
    const p = predict(model, { sqft: 2000, beds: 3 });
    expect(p.location).toBeNull();
    expect(p.value).toBeCloseTo(predict(model, { sqft: 2000, beds: 3 }, { location: false }).value, 6);
    expect(p.warnings.join()).toMatch(/no location adjustment/);
  });

  it('adjusts batch rows that have coordinates and flags the rest', () => {
    expect(modelInputs(model).filter((t) => t.location).map((t) => t.name)).toEqual(['latitude', 'longitude']);
    const mapping = suggestMapping(model, verify.columns);
    expect(mapping).toMatchObject({ inputs: { latitude: 'latitude', longitude: 'longitude' }, actual: 'closePrice', label: 'mlsNumber' });
    const { results, summary } = predictBatch(model, verify.rows, mapping);
    const without = results.filter((r) => r.missingLocation.length);
    expect(without).toHaveLength(10);
    expect(without[0]).toMatchObject({ missingLocation: ['latitude', 'longitude'], locationAdj: null, imputed: [] });
    expect(summary.imputedRows).toBe(10);

    // Held-out sold houses: the location adjustment cuts the error.
    const located = results.filter((r) => r.locationAdj !== null);
    const formulaOnly = predictBatch({ ...model, location: undefined }, verify.rows, mapping).results;
    const mae = (rs) => rs.reduce((a, r) => a + Math.abs(r.gap), 0) / rs.length;
    expect(mae(located)).toBeLessThan(0.8 * mae(formulaOnly.filter((r) => located.some((l) => l.index === r.index))));

    const csv = resultsToCsv(verify.columns, verify.rows, results, 'price');
    expect(csv.split('\n')[0]).toContain('est_price,location_adj,gap');
  });

  it('builds a location per group from that group only', async () => {
    // Two kinds sharing the same map, each with its own formula and residuals.
    const rows = train.rows.map((r, i) => ({ ...r, kind: i % 2 ? 'a' : 'b' }));
    const st = analyzeColumns([...train.columns, 'kind'], rows);
    const grouped = await trainModel(rows, st, { ...base, roles: { ...suggestRoles(train.columns, stats), kind: 'input' }, groupBy: 'kind' });
    for (const g of Object.values(grouped.groups)) expect(g.location.points).toHaveLength(400);
    expect(grouped.locationMetrics.n).toBe(800);
    expect(grouped.locationMetrics.r2After).toBeGreaterThan(grouped.locationMetrics.r2Before);
    const p = predict(grouped, { kind: 'a', sqft: 2000, beds: 3, latitude: 30.29, longitude: -97.75 });
    expect(p.group).toBe('a');
    expect(p.location.adjustment).toBeGreaterThan(0);
  });

  it('judges outliers against their neighbourhood, not the whole map', async () => {
    const m = await trainModel(train.rows, stats, { ...base, steps: 300, outlierCutoff: 3.5, roles: suggestRoles(train.columns, stats) });
    // Houses in the pricey and cheap spots are explained by their neighbours.
    expect(m.training.outliers.length).toBeLessThan(train.rows.length * 0.02);
    // A wrong price among normal neighbours still stands out.
    const rows = train.rows.map((r, i) => (i === 5 ? { ...r, price: String(Number(r.price) * 10) } : r));
    const poisoned = await trainModel(rows, analyzeColumns(train.columns, rows), { ...base, steps: 300, outlierCutoff: 3.5, roles: suggestRoles(train.columns, stats) });
    expect(poisoned.training.outliers[0].row).toBe(5);
  });

  it('trains without a location when no lat/lng is marked', async () => {
    const m = await trainModel(train.rows, stats, { ...base, steps: 200, roles: { sqft: 'input', beds: 'input', price: 'outcome' } });
    expect(m.location).toBeUndefined();
    expect(predict(m, { sqft: 2000, beds: 3 }).location).toBeNull();
  });
});

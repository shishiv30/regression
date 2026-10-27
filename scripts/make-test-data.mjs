// Generates the test data collection in public/test-data/.
// Every dataset comes from a known formula (plus noise), recorded in index.json
// as `truth`, so trained weights can be checked against it.
//   node scripts/make-test-data.mjs
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const OUT = new URL('../public/test-data/', import.meta.url);
mkdirSync(OUT, { recursive: true });

let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const pick = (arr) => arr[Math.floor(rand() * arr.length)];
const uniform = (a, b) => a + rand() * (b - a);
const round = (v, d = 0) => Math.round(v * 10 ** d) / 10 ** d;
const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

const index = [];
function save(file, { title, description, columns, rows, truth, eol = '\n' }) {
  const text = [columns.join(','), ...rows.map((r) => r.map(csvCell).join(','))].join(eol) + eol;
  writeFileSync(new URL(file, OUT), text);
  index.push({ file, title, description, rows: rows.length, truth });
  console.log(`${file.padEnd(26)} ${String(rows.length).padStart(7)} rows  ${(text.length / 1024).toFixed(0).padStart(6)} KB`);
}

// 1. Salary: several enums, a text column, collinear age/experience,
//    thousands separators in the outcome, ~3% missing cells.
{
  const edu = { 'High School': 0, Bachelor: 15000, Master: 25000, PhD: 35000 };
  const dept = { Engineering: 20000, Sales: 5000, Marketing: 0, Support: -8000, HR: -5000 };
  const first = ['Ana', 'Ben', 'Chen', 'Dana', 'Eli', 'Fatima', 'Goran', 'Hana', 'Ivan', 'Jia', 'Kofi', 'Lena'];
  const last = ['Smith', 'Lee', 'Garcia', 'Okafor', 'Novak', 'Tanaka', 'Silva', 'Khan', 'Muller', 'Rossi'];
  const rows = [];
  for (let i = 0; i < 500; i++) {
    const exp = Math.floor(uniform(0, 31));
    const age = Math.round(22 + exp + uniform(0, 12));
    const e = pick(Object.keys(edu)), d = pick(Object.keys(dept)), remote = rand() < 0.35 ? 'yes' : 'no';
    const salary = 40000 + 2500 * exp + edu[e] + dept[d] + (remote === 'yes' ? -3000 : 0) + gauss() * 6000;
    const row = [`${pick(first)} ${pick(last)}`, exp, age, e, d, remote, Math.round(salary).toLocaleString('en-US')];
    for (const k of [1, 3, 4]) if (rand() < 0.01) row[k] = '';
    rows.push(row);
  }
  save('salary.csv', {
    title: 'Salary (enums + missing values)',
    description: '500 employees. 3 enum inputs, a free-text name column, age collinear with experience, "85,000"-style outcome, ~3% missing cells.',
    columns: ['employee_name', 'years_experience', 'age', 'education', 'department', 'remote', 'salary'],
    rows,
    truth: { outcome: 'salary', intercept: 40000, weights: { years_experience: 2500, age: 0 }, enums: { education: edu, department: dept, remote: { no: 0, yes: -3000 } } },
  });
}

// 2. Fuel efficiency: 8 inputs, correlated, numeric-looking cylinders.
{
  const origin = { USA: 0, Europe: 1.5, Japan: 3 };
  const trans = { automatic: 0, manual: 1.2 };
  const drive = { FWD: 0, RWD: -1, AWD: -1.8 };
  const rows = [];
  for (let i = 0; i < 400; i++) {
    const cyl = pick([4, 4, 4, 6, 6, 8]);
    const disp = round(cyl * uniform(25, 45));
    const hp = round(disp * uniform(0.45, 0.75) + 20);
    const weight = round(1600 + cyl * 250 + uniform(-300, 400));
    const accel = round(uniform(8, 20), 1);
    const year = Math.floor(uniform(2000, 2025));
    const o = pick(Object.keys(origin)), t = pick(Object.keys(trans)), dr = pick(Object.keys(drive));
    const mpg = 20 - 0.004 * weight - 0.03 * hp + 0.2 * accel + 0.35 * (year - 2000) + origin[o] + trans[t] + drive[dr] + 15 + gauss() * 1.5;
    rows.push([weight, hp, cyl, disp, accel, year, o, t, dr, round(mpg, 1)]);
  }
  save('fuel-efficiency.csv', {
    title: 'Fuel efficiency (8 inputs)',
    description: '400 cars, 8 inputs (5 numeric, 3 enum), strongly correlated engine specs. Checks how the graph handles many nodes.',
    columns: ['weight_lbs', 'horsepower', 'cylinders', 'displacement', 'acceleration', 'model_year', 'origin', 'transmission', 'drive', 'mpg'],
    rows,
    truth: { outcome: 'mpg', weights: { weight_lbs: -0.004, horsepower: -0.03, cylinders: 0, displacement: 0, acceleration: 0.2, model_year: 0.35 }, enums: { origin, transmission: trans, drive } },
  });
}

// 3. Ad spend: tiny dataset; newspaper has no real effect.
{
  const rows = [];
  for (let i = 0; i < 30; i++) {
    const tv = round(uniform(0, 300), 1), radio = round(uniform(0, 50), 1), paper = round(uniform(0, 100), 1);
    rows.push([tv, radio, paper, round(3 + 0.045 * tv + 0.19 * radio + gauss() * 0.8, 1)]);
  }
  save('ad-spend.csv', {
    title: 'Ad spend (tiny, one useless input)',
    description: '30 rows only. newspaper has a true weight of 0, so its edge should come out thin.',
    columns: ['tv', 'radio', 'newspaper', 'sales'],
    rows,
    truth: { outcome: 'sales', intercept: 3, weights: { tv: 0.045, radio: 0.19, newspaper: 0 } },
  });
}

// 4. Messy: CRLF, padded whitespace, quoted commas, blank lines, N/A text in
//    a numeric column, a constant column, booleans, an empty column.
{
  const rows = [];
  for (let i = 0; i < 120; i++) {
    const rooms = Math.floor(uniform(1, 6));
    const dist = round(uniform(0.5, 25), 2);
    const pets = rand() < 0.5 ? 'TRUE' : 'FALSE';
    const city = pick(['Austin, TX', 'Portland, OR', 'Denver, CO']);
    const cityAdj = { 'Austin, TX': 150, 'Portland, OR': 300, 'Denver, CO': 0 }[city];
    const rent = 800 + 450 * rooms - 20 * dist + (pets === 'TRUE' ? 75 : 0) + cityAdj + gauss() * 60;
    const zip = rand() < 0.05 ? 'N/A' : String(Math.floor(uniform(10000, 99999)));
    rows.push([`  ${rooms} `, dist, ` ${pets}`, city, zip, 'US', '', rand() < 0.04 ? '' : Math.round(rent)]);
  }
  rows.splice(40, 0, ['', '', '', '', '', '', '', '']); // blank record
  save('messy.csv', {
    title: 'Messy formatting',
    description: 'CRLF line endings, padded spaces, quoted "City, ST" values, "N/A" in a numeric-looking column, a constant column, an empty column, blank lines and missing outcomes.',
    columns: ['rooms', 'distance_km', 'pets_allowed', 'city', 'zip', 'country', 'notes', 'monthly_rent'],
    rows,
    eol: '\r\n',
    truth: { outcome: 'monthly_rent', intercept: 800, weights: { rooms: 450, distance_km: -20 }, enums: { pets_allowed: { FALSE: 0, TRUE: 75 }, city: { 'Denver, CO': 0, 'Austin, TX': 150, 'Portland, OR': 300 } } },
  });
}

// 5. Scale: inputs spanning 1e-3 … 1e6 and an outcome in the billions.
{
  const rows = [];
  for (let i = 0; i < 300; i++) {
    const rate = round(uniform(0.001, 0.009), 5);
    const employees = Math.floor(uniform(10, 50000));
    const assets = Math.round(uniform(1e5, 5e6));
    const sector = pick(['Tech', 'Retail', 'Energy', 'Health']);
    const adj = { Tech: 2e8, Retail: 0, Energy: 5e8, Health: 1e8 }[sector];
    const value = 1e8 - 4e10 * rate + 30000 * employees + 150 * assets + adj + gauss() * 5e7;
    rows.push([rate, employees, assets, sector, Math.round(value)]);
  }
  save('company-value.csv', {
    title: 'Extreme scales',
    description: 'Interest rate ~0.005, employees ~10k, assets ~1M, outcome ~1e9. Checks normalisation and number formatting.',
    columns: ['interest_rate', 'employees', 'assets_usd', 'sector', 'valuation_usd'],
    rows,
    truth: { outcome: 'valuation_usd', intercept: 1e8, weights: { interest_rate: -4e10, employees: 30000, assets_usd: 150 }, enums: { sector: { Retail: 0, Tech: 2e8, Energy: 5e8, Health: 1e8 } } },
  });
}

// 6. Large: sized to exceed the ~5 MB localStorage quota and stress training.
{
  const sites = Array.from({ length: 12 }, (_, i) => `site-${String.fromCharCode(65 + i)}`);
  const siteAdj = Object.fromEntries(sites.map((s, i) => [s, (i - 6) * 0.8]));
  const rows = [];
  for (let i = 0; i < 120000; i++) {
    const temp = round(uniform(-10, 40), 2), hum = round(uniform(10, 95), 2), wind = round(uniform(0, 30), 2);
    const pressure = round(uniform(980, 1040), 1), hour = Math.floor(uniform(0, 24)), site = pick(sites);
    const load = 50 + 1.8 * temp - 0.25 * hum + 0.6 * wind + 0.05 * (pressure - 1000) + 0.4 * hour + siteAdj[site] + gauss() * 3;
    rows.push([i + 1, temp, hum, wind, pressure, hour, site, round(load, 2)]);
  }
  save('sensor-large.csv', {
    title: 'Large (120k rows)',
    description: '120,000 sensor readings with a 12-value enum. Bigger than the localStorage quota; checks the warning and training speed.',
    columns: ['reading_id', 'temperature_c', 'humidity_pct', 'wind_kmh', 'pressure_hpa', 'hour', 'site', 'power_load_kw'],
    rows,
    truth: { outcome: 'power_load_kw', weights: { temperature_c: 1.8, humidity_pct: -0.25, wind_kmh: 0.6, pressure_hpa: 0.05, hour: 0.4 }, enums: { site: siteAdj } },
  });
}

// 7. Location: house prices with a neighbourhood effect a linear formula of
//    lat/lng cannot express (two pricey spots and a cheap one). A training
//    file and a verify file with sold prices; some verify rows lack coordinates.
{
  const spots = [
    { lat: 30.29, lng: -97.75, km: 1.5, amount: 150000 },
    { lat: 30.22, lng: -97.70, km: 2, amount: 90000 },
    { lat: 30.25, lng: -97.80, km: 1.8, amount: -80000 },
  ];
  const kmTo = (s, lat, lng) => Math.hypot((lat - s.lat) * 110.57, (lng - s.lng) * 111.32 * Math.cos((lat * Math.PI) / 180));
  const locEffect = (lat, lng) => spots.reduce((a, s) => a + s.amount * Math.exp(-0.5 * (kmTo(s, lat, lng) / s.km) ** 2), 0);
  const house = () => {
    const sqft = Math.round(uniform(700, 3500)), beds = Math.floor(uniform(1, 6));
    const lat = round(uniform(30.18, 30.34), 6), lng = round(uniform(-97.84, -97.66), 6);
    const price = 120000 + 210 * sqft + 15000 * beds + locEffect(lat, lng) + gauss() * 15000;
    return { sqft, beds, lat, lng, price: Math.round(price) };
  };
  const truth = { outcome: 'price', intercept: 120000, weights: { sqft: 210, beds: 15000 }, location: spots };
  save('location-train.csv', {
    title: 'Neighbourhood prices (lat/lng, training)',
    description: '800 houses. Price = formula of sqft and beds + a neighbourhood effect (two pricey spots, one cheap). latitude/longitude are picked up as the location automatically.',
    columns: ['sqft', 'beds', 'latitude', 'longitude', 'price'],
    rows: Array.from({ length: 800 }, () => house()).map((h) => [h.sqft, h.beds, h.lat, h.lng, h.price]),
    truth,
  });
  save('location-verify.csv', {
    title: 'Neighbourhood prices (lat/lng, verify)',
    description: '150 sold houses from the same area to check the location adjustment; 10 have no coordinates.',
    columns: ['mlsNumber', 'sqft', 'beds', 'latitude', 'longitude', 'closePrice'],
    rows: Array.from({ length: 150 }, (_, i) => {
      const h = house();
      const noCoords = i % 15 === 0;
      return [`LOC-${1000 + i}`, h.sqft, h.beds, noCoords ? '' : h.lat, noCoords ? '' : h.lng, h.price];
    }),
    truth: null, // a verify file: its outcome column is closePrice
  });
}

// 8. Real Austin listings (checked in under test/): train on the for-sale
//    file, then batch-predict the sold file and compare with closePrice.
//    No known formula, so no `truth`.
for (const [src, title, description] of [
  ['test-austin-training-data.csv', 'Austin for-sale listings (training)', 'Real for-sale listings. Train with listPrice as the outcome.'],
  ['test-austin-verify-data.csv', 'Austin sold listings (verify)', 'Real sold listings. Use under "Predict many records" with closePrice as the actual value.'],
]) {
  const text = readFileSync(new URL(`../test/${src}`, import.meta.url), 'utf8');
  writeFileSync(new URL(src, OUT), text);
  const rows = text.split(/\r?\n/).filter((l) => l.trim()).length - 1;
  index.push({ file: src, title, description, rows, truth: null });
  console.log(`${src.padEnd(26)} ${String(rows).padStart(7)} rows  (copied)`);
}

writeFileSync(new URL('index.json', OUT), JSON.stringify(index, null, 2) + '\n');

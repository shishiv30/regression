// Model space ↔ outcome units. On a log scale the formula estimates
// log(outcome); otherwise the two are the same.
export const toValue = (outcome, v) => (outcome.log ? Math.exp(v) : v);
export const toModel = (outcome, v) => (outcome.log ? Math.log(v) : v);

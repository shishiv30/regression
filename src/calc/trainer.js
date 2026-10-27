// Gradient-descent linear regression on TensorFlow.js (CPU backend, runs
// entirely in the browser): minimise MSE of  X·w + b  against y.
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/register_all_gradients.js';
import '@tensorflow/tfjs-backend-cpu';

let backendReady;
function ensureBackend() {
  backendReady ??= tf.setBackend('cpu').then(() => tf.ready());
  return backendReady;
}

const yieldToUi = () => new Promise((r) => setTimeout(r, 0));

export async function trainLinear(X, y, { steps = 500, learningRate = 0.05, optimizer = 'adam', onProgress, shouldStop } = {}) {
  await ensureBackend();
  const n = X.length;
  const d = X[0].length;
  const xs = tf.tensor2d(X, [n, d]);
  const ys = tf.tensor2d(y, [n, 1]);
  const w = tf.variable(tf.zeros([d, 1]));
  const b = tf.variable(tf.scalar(0));
  const opt = optimizer === 'sgd' ? tf.train.sgd(learningRate) : tf.train.adam(learningRate);
  const loss = () => tf.mean(tf.square(tf.sub(tf.add(tf.matMul(xs, w), b), ys)));

  const history = [];
  let stopped = false;
  try {
    for (let step = 1; step <= steps; step++) {
      const cost = opt.minimize(loss, true, [w, b]);
      const value = cost.dataSync()[0];
      cost.dispose();
      if (!Number.isFinite(value)) throw new Error('Training diverged (loss is not finite). Try a smaller learning rate.');
      history.push(value);
      if (step % 20 === 0 || step === steps) {
        onProgress?.({ step, steps, loss: value });
        await yieldToUi();
        if (shouldStop?.()) {
          stopped = true;
          break;
        }
      }
    }
    return { weights: Array.from(w.dataSync()), bias: b.dataSync()[0], history, stopped };
  } finally {
    tf.dispose([xs, ys, w, b]);
    opt.dispose();
  }
}

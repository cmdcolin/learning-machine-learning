/**
 * Browser-based MNIST training — pure TypeScript, no dependencies.
 *
 * Implements a 2-layer MLP (Multi-Layer Perceptron):
 *   Input (784 pixels) → Hidden Layer (128 neurons, ReLU) → Output (10 classes, Softmax)
 *
 * The same math as train.py, but written step-by-step so you can see exactly
 * what's happening without any framework magic.
 *
 * Key concepts covered:
 *   - Xavier weight initialization
 *   - Forward pass (matrix multiply → activation → softmax)
 *   - Cross-entropy loss
 *   - Backpropagation (chain rule)
 *   - Adam optimizer
 */

export const HIDDEN = 128; // neurons in the hidden layer
export const INPUT = 784;  // 28×28 pixels, flattened
export const OUTPUT = 10;  // digits 0–9

export interface TrainSample {
  image: number[]; // 784 pixel values, 0–255 (normalized to 0–1 before use)
  label: number;   // true digit 0–9
}

export interface MLPWeights {
  W1: number[][]; // [128][784] — hidden layer weight matrix
  b1: number[];   // [128]     — hidden layer biases
  W2: number[][]; // [10][128] — output layer weight matrix
  b2: number[];   // [10]      — output layer biases
}

// Adam keeps two running averages per parameter to adapt the learning rate.
// Mutated in-place during training to avoid GC pressure.
export interface AdamState {
  t: number; // global timestep (incremented each batch)
  mW1: number[][]; vW1: number[][];
  mb1: number[];   vb1: number[];
  mW2: number[][]; vW2: number[][];
  mb2: number[];   vb2: number[];
}

export interface EpochResult {
  epoch: number;
  loss: number;
  accuracy: number;
}

// ── Initialization ──────────────────────────────────────────────────────────

// Box-Muller: generates a standard-normal random number from two uniforms.
// Needed because Math.random() only gives a uniform [0,1] distribution.
function randn(): number {
  const u1 = Math.random() || 1e-10; // avoid log(0)
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Xavier init: scales weights by sqrt(2 / (fan_in + fan_out)).
// Keeps activation variances roughly constant across layers so gradients
// neither explode nor vanish at the start of training.
function makeMatrix(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(randn() * scale);
    }
    m.push(row);
  }
  return m;
}

export function initWeights(): MLPWeights {
  return {
    W1: makeMatrix(HIDDEN, INPUT),
    b1: new Array(HIDDEN).fill(0),
    W2: makeMatrix(OUTPUT, HIDDEN),
    b2: new Array(OUTPUT).fill(0),
  };
}

export function initAdamState(): AdamState {
  const zeros2 = (r: number, c: number) =>
    Array.from({ length: r }, () => new Array(c).fill(0));
  return {
    t: 0,
    mW1: zeros2(HIDDEN, INPUT), vW1: zeros2(HIDDEN, INPUT),
    mb1: new Array(HIDDEN).fill(0), vb1: new Array(HIDDEN).fill(0),
    mW2: zeros2(OUTPUT, HIDDEN), vW2: zeros2(OUTPUT, HIDDEN),
    mb2: new Array(OUTPUT).fill(0), vb2: new Array(OUTPUT).fill(0),
  };
}

// ── Forward Pass ─────────────────────────────────────────────────────────────

// We cache intermediate values (z1, a1) because backprop needs them.
export interface ForwardCache {
  x: number[];     // normalized input (784 values, 0–1)
  z1: number[];    // pre-ReLU activations for hidden layer (128 values)
  a1: number[];    // post-ReLU activations (128 values, ≥0)
  probs: number[]; // final softmax probabilities (10 values, sum=1)
}

// ReLU(x) = max(0, x) — the simplest useful nonlinearity.
// Lets the network learn nonlinear functions; without it, stacking layers
// would collapse to a single matrix multiply.
function relu(x: number[]): number[] {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] > 0 ? x[i] : 0;
  }
  return out;
}

// Softmax converts raw scores (logits) into probabilities.
// We subtract max(z) first to prevent exp() overflow (numerically equivalent).
function softmax(z: number[]): number[] {
  let max = z[0];
  for (let i = 1; i < z.length; i++) {
    if (z[i] > max) max = z[i];
  }
  let sum = 0;
  const exps = new Array(z.length);
  for (let i = 0; i < z.length; i++) {
    exps[i] = Math.exp(z[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < z.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

// matrix-vector multiply: result[i] = Σ_j M[i][j] * v[j] + bias[i]
function matVecAdd(M: number[][], v: number[], bias: number[]): number[] {
  const out = new Array(M.length);
  for (let i = 0; i < M.length; i++) {
    let sum = bias[i];
    const row = M[i];
    for (let j = 0; j < v.length; j++) {
      sum += row[j] * v[j];
    }
    out[i] = sum;
  }
  return out;
}

export function forward(imageUint8: number[], W: MLPWeights): ForwardCache {
  // Normalize pixels from [0,255] to [0,1]
  const x = new Array(INPUT);
  for (let i = 0; i < INPUT; i++) x[i] = imageUint8[i] / 255;

  const z1 = matVecAdd(W.W1, x, W.b1); // hidden pre-activation
  const a1 = relu(z1);                  // hidden post-activation
  const z2 = matVecAdd(W.W2, a1, W.b2); // output logits
  const probs = softmax(z2);

  return { x, z1, a1, probs };
}

// ── Loss ─────────────────────────────────────────────────────────────────────

// Cross-entropy loss: -log(probability assigned to the correct class).
// If probs[label] = 1.0 (perfectly confident), loss = 0.
// If probs[label] = 0.01 (very confused), loss = 4.6.
function crossEntropy(probs: number[], label: number): number {
  return -Math.log(Math.max(probs[label], 1e-10));
}

// ── Backward Pass (Backpropagation) ──────────────────────────────────────────

interface Gradients {
  dW1: number[][]; db1: number[];
  dW2: number[][]; db2: number[];
}

// Backprop computes ∂Loss/∂W for every weight using the chain rule.
// We reuse the cached activations from forward() to avoid recomputing them.
function backward(cache: ForwardCache, label: number, W: MLPWeights): Gradients {
  const { x, z1, a1, probs } = cache;

  // --- Output layer ---
  // d(Loss)/d(z2) = probs - one_hot(label)
  // This is the elegant combined derivative of softmax + cross-entropy.
  const dz2 = [...probs];
  dz2[label] -= 1; // subtract 1 at the true class position

  // d(Loss)/d(W2[i][j]) = dz2[i] * a1[j]
  const dW2: number[][] = [];
  for (let i = 0; i < OUTPUT; i++) {
    const row: number[] = new Array(HIDDEN);
    for (let j = 0; j < HIDDEN; j++) {
      row[j] = dz2[i] * a1[j];
    }
    dW2.push(row);
  }
  const db2 = [...dz2];

  // --- Hidden layer ---
  // Propagate gradient back through W2: da1 = W2ᵀ · dz2
  const da1 = new Array(HIDDEN).fill(0);
  for (let j = 0; j < HIDDEN; j++) {
    for (let i = 0; i < OUTPUT; i++) {
      da1[j] += W.W2[i][j] * dz2[i];
    }
  }

  // ReLU derivative: pass gradient through only where z1 was positive
  // (ReLU "gates" the gradient — zero where it was inactive)
  const dz1 = new Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) {
    dz1[i] = z1[i] > 0 ? da1[i] : 0;
  }

  // d(Loss)/d(W1[i][j]) = dz1[i] * x[j]
  const dW1: number[][] = [];
  for (let i = 0; i < HIDDEN; i++) {
    const row: number[] = new Array(INPUT);
    for (let j = 0; j < INPUT; j++) {
      row[j] = dz1[i] * x[j];
    }
    dW1.push(row);
  }
  const db1 = [...dz1];

  return { dW1, db1, dW2, db2 };
}

// ── Adam Optimizer ────────────────────────────────────────────────────────────

// Adam ("Adaptive Moment Estimation") keeps a moving average of gradients (m)
// and their squared values (v) to give each weight its own effective learning rate.
//
// Update rule:
//   m = β₁·m + (1-β₁)·g      ← smoothed gradient direction
//   v = β₂·v + (1-β₂)·g²     ← smoothed gradient magnitude
//   w -= lr · (m̂) / (√v̂ + ε)  ← bias-corrected update
//
// Weights and state are mutated in-place to avoid creating millions of
// temporary arrays on every update step.
function adamStep(
  W: MLPWeights,
  grads: Gradients,
  state: AdamState,
  lr: number,
): void {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  state.t++;
  const t = state.t;
  // Bias correction factors: early in training m and v are biased toward zero
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);

  for (let i = 0; i < HIDDEN; i++) {
    for (let j = 0; j < INPUT; j++) {
      const g = grads.dW1[i][j];
      state.mW1[i][j] = beta1 * state.mW1[i][j] + (1 - beta1) * g;
      state.vW1[i][j] = beta2 * state.vW1[i][j] + (1 - beta2) * g * g;
      W.W1[i][j] -= lr * (state.mW1[i][j] / bc1) / (Math.sqrt(state.vW1[i][j] / bc2) + eps);
    }
    const gb = grads.db1[i];
    state.mb1[i] = beta1 * state.mb1[i] + (1 - beta1) * gb;
    state.vb1[i] = beta2 * state.vb1[i] + (1 - beta2) * gb * gb;
    W.b1[i] -= lr * (state.mb1[i] / bc1) / (Math.sqrt(state.vb1[i] / bc2) + eps);
  }

  for (let i = 0; i < OUTPUT; i++) {
    for (let j = 0; j < HIDDEN; j++) {
      const g = grads.dW2[i][j];
      state.mW2[i][j] = beta1 * state.mW2[i][j] + (1 - beta1) * g;
      state.vW2[i][j] = beta2 * state.vW2[i][j] + (1 - beta2) * g * g;
      W.W2[i][j] -= lr * (state.mW2[i][j] / bc1) / (Math.sqrt(state.vW2[i][j] / bc2) + eps);
    }
    const gb = grads.db2[i];
    state.mb2[i] = beta1 * state.mb2[i] + (1 - beta1) * gb;
    state.vb2[i] = beta2 * state.vb2[i] + (1 - beta2) * gb * gb;
    W.b2[i] -= lr * (state.mb2[i] / bc1) / (Math.sqrt(state.vb2[i] / bc2) + eps);
  }
}

// ── Training Loop ─────────────────────────────────────────────────────────────

// Fisher-Yates shuffle — unbiased random permutation
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Accumulate gradients for one batch, then do a single Adam step.
// Called once per batch inside trainEpoch.
function processBatch(
  batch: TrainSample[],
  W: MLPWeights,
  adamState: AdamState,
  lr: number,
): { batchLoss: number; batchCorrect: number } {
  // Accumulate gradients over every sample in the batch
  const accW1: number[][] = Array.from({ length: HIDDEN }, () => new Array(INPUT).fill(0));
  const accb1: number[] = new Array(HIDDEN).fill(0);
  const accW2: number[][] = Array.from({ length: OUTPUT }, () => new Array(HIDDEN).fill(0));
  const accb2: number[] = new Array(OUTPUT).fill(0);

  let batchLoss = 0;
  let batchCorrect = 0;

  for (const sample of batch) {
    const cache = forward(sample.image, W);
    batchLoss += crossEntropy(cache.probs, sample.label);

    let maxP = cache.probs[0], maxI = 0;
    for (let k = 1; k < OUTPUT; k++) {
      if (cache.probs[k] > maxP) { maxP = cache.probs[k]; maxI = k; }
    }
    if (maxI === sample.label) batchCorrect++;

    const g = backward(cache, sample.label, W);

    for (let i = 0; i < HIDDEN; i++) {
      for (let j = 0; j < INPUT; j++) accW1[i][j] += g.dW1[i][j];
      accb1[i] += g.db1[i];
    }
    for (let i = 0; i < OUTPUT; i++) {
      for (let j = 0; j < HIDDEN; j++) accW2[i][j] += g.dW2[i][j];
      accb2[i] += g.db2[i];
    }
  }

  // Average gradients
  const n = batch.length;
  for (let i = 0; i < HIDDEN; i++) {
    for (let j = 0; j < INPUT; j++) accW1[i][j] /= n;
    accb1[i] /= n;
  }
  for (let i = 0; i < OUTPUT; i++) {
    for (let j = 0; j < HIDDEN; j++) accW2[i][j] /= n;
    accb2[i] /= n;
  }

  adamStep(W, { dW1: accW1, db1: accb1, dW2: accW2, db2: accb2 }, adamState, lr);

  return { batchLoss: batchLoss / n, batchCorrect };
}

// Train one full epoch. Shuffles data, iterates mini-batches, yields to the
// browser event loop between batches so the UI stays responsive.
export async function trainEpoch(
  data: TrainSample[],
  W: MLPWeights,
  adamState: AdamState,
  lr: number,
  batchSize: number,
  onBatch: (batchLoss: number, batchIdx: number, totalBatches: number) => void,
  signal: { cancelled: boolean },
): Promise<{ avgLoss: number; accuracy: number }> {
  const shuffled = shuffle(data);
  const totalBatches = Math.ceil(shuffled.length / batchSize);
  let totalLoss = 0;
  let totalCorrect = 0;
  let batchIdx = 0;

  for (let start = 0; start < shuffled.length; start += batchSize) {
    if (signal.cancelled) break;
    const batch = shuffled.slice(start, start + batchSize);
    const { batchLoss, batchCorrect } = processBatch(batch, W, adamState, lr);
    totalLoss += batchLoss;
    totalCorrect += batchCorrect;
    batchIdx++;
    onBatch(batchLoss, batchIdx, totalBatches);
    // Yield to browser so UI can update and user can cancel
    await new Promise<void>((r) => setTimeout(r, 0));
  }

  return {
    avgLoss: totalLoss / batchIdx,
    accuracy: totalCorrect / shuffled.length,
  };
}

// Run inference on a single image with the browser-trained weights
export function inferBrowser(
  imageUint8: number[],
  W: MLPWeights,
): { prediction: number; probs: number[] } {
  const cache = forward(imageUint8, W);
  let maxP = cache.probs[0], prediction = 0;
  for (let k = 1; k < OUTPUT; k++) {
    if (cache.probs[k] > maxP) { maxP = cache.probs[k]; prediction = k; }
  }
  return { prediction, probs: cache.probs };
}

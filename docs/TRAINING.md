# Training MNIST with JAX

This document explains the training process in `train.py`.

## 1. The Model Architecture (The "Brain")

The model is a sequential pipeline of layers. Each layer's job is to extract increasingly abstract features from the image.

```text
[Input: 784 pixels] 
       ↓ 
[Layer 1: 1024 Neurons]  ← Learns simple edges and curves
       ↓ (ReLU)
[Layer 2: 512 Neurons]   ← Learns parts of digits (loops, bars)
       ↓ (ReLU)
[Output: 10 Neurons]     ← Decides which digit matches best
```

## 2. Using JAX for Performance

JAX isn't just a library; it's a **compiler** for math. 

- **XLA (Accelerated Linear Algebra)**: When we use `@jit`, JAX looks at our `update` function, fuses all the operations together (like `add`, `multiply`, `relu`), and compiles them into a single, high-speed binary. This is why our training loop is so fast despite being written in Python.
- **Vmap (Vectorization)**: Instead of writing a `for` loop to process 128 images, `vmap` allows us to write the logic for *one* image and automatically transforms it to run on the whole batch in parallel.

## 3. The "Training" Loop Flow

Training is an iterative "Game of Hot and Cold":
1. **Forward**: The model guesses (e.g., "This is a 3").
2. **Loss**: We compare the guess to the truth (e.g., "Actually, it's a 7").
3. **Gradients**: JAX calculates the **exact mathematical blame** for every one of the 1.3 million parameters.
4. **Update**: The optimizer (AdamW) "nudges" the weights.

## 4. Generalization & Augmentation

To help the model recognize freehand drawings (which often look messier than the official MNIST set), we use **Data Augmentation**:
- **Gaussian Noise**: Small random fluctuations are added to each pixel.
- **Weight Decay**: AdamW uses L2 regularization to keep weights small, preventing the model from over-fitting to specific training images.

## 5. Serialization & Export

Once trained, the JAX model's parameters (the learned $W$ and $b$ matrices) are converted from JAX arrays to nested Python lists and saved as a JSON file. This allows our TypeScript code to load them without needing any Python runtime.

## 6. Training in the Browser (TypeScript Backprop)

The **"Train in Browser"** tab in the web app re-implements the same algorithm in pure TypeScript — no Python, no JAX, no external libraries. This is intentionally smaller and slower than the JAX version, but every step is visible and readable.

### Architecture comparison

| | JAX (train.py) | Browser (train.ts) |
|---|---|---|
| Layers | 784 → 1024 → 512 → 10 | 784 → 128 → 10 |
| Parameters | ~1.3M | ~102K |
| Training data | 60K images | 500 images |
| Optimizer | AdamW | Adam |
| Speed | Seconds/epoch (GPU) | ~2–5s/epoch (JS) |
| Expected accuracy | ~98% | ~85–92% |

### How browser training works

The file `packages/web/src/train.ts` implements:

1. **Xavier init** (`initWeights`): sets random starting weights scaled by `sqrt(2/(fan_in+fan_out))` to avoid vanishing/exploding gradients early in training.

2. **Forward pass** (`forward`): runs an image through `W1·x+b1 → ReLU → W2·a1+b2 → Softmax`, caching intermediate values needed for backprop.

3. **Cross-entropy loss**: `-log(probability of correct class)`. The browser shows this live per batch.

4. **Backpropagation** (`backward`): applies the chain rule layer by layer. The key insight is that the combined gradient of softmax + cross-entropy is simply `probs - one_hot(label)` — very clean to implement.

5. **Adam optimizer** (`adamStep`): updates weights in-place using per-parameter adaptive learning rates. Avoids the allocation overhead of creating new arrays on every step.

6. **Async batching** (`trainEpoch`): uses `setTimeout(resolve, 0)` between batches to yield to the browser event loop, keeping the UI responsive.

---

### External Resources
- [JAX Quickstart Guide](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Understanding ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
- [Cross-Entropy Loss Explained](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)

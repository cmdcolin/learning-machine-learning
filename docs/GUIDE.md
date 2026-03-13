# Understanding This Project — A Plain-English Guide

This project trains a computer to recognize handwritten digits. You can watch it happen live in your browser. This guide explains what's actually going on, using TypeScript analogies throughout — no math background needed.

---

## The one-sentence summary

A neural network is just a function with a lot of adjustable numbers inside it. **Training** means finding the right values for those numbers. **Inference** means using those numbers to make a prediction.

---

## What the 28×28 image actually is

Every MNIST image is 28 pixels wide and 28 pixels tall. Each pixel is a number from 0 (black) to 255 (white).

Before feeding it to the network we flatten the 2D grid into a simple array:

```
[0, 0, 12, 255, 200, 0, 0, ...]   ← 784 numbers total (28 × 28)
```

Then we divide each by 255 so everything is between 0.0 and 1.0. That's the only image processing. The network never "sees" a 2D picture — it just gets 784 floats.

---

## What a layer actually does

A layer is a function that takes an array and produces a smaller (or larger) array. Concretely, each output value is a weighted sum of every input value:

```typescript
// For one output neuron:
output = (input[0] * weight[0]) + (input[1] * weight[1]) + ... + bias
```

The **weights** are just numbers that decide how much each input matters. A weight of `0` means "ignore this pixel completely." A large positive weight means "this pixel is very important evidence." A negative weight means "this pixel is evidence *against* this digit."

The network in `train.ts` has two layers:

| Layer | Input size | Output size | What it learns |
|---|---|---|---|
| Hidden (W1, b1) | 784 pixels | 128 values | Low-level patterns — edges, curves, strokes |
| Output (W2, b2) | 128 values | 10 values | Which digit matches those patterns |

The final 10 output values are one score per digit (0 through 9). The digit with the highest score wins.

---

## What ReLU does

After the first layer, we apply ReLU to every value:

```typescript
relu(x) = x > 0 ? x : 0
```

This seems almost too simple to matter. But without it, stacking two layers would be *mathematically identical* to having one layer — all the matrix multiplications would collapse into a single one. ReLU breaks that by making the network nonlinear, which is what lets it learn shapes and patterns rather than just linear combinations of pixels.

Think of it as a switch: each neuron is either "off" (the pattern wasn't found) or "on by some amount" (the pattern was found to some degree).

---

## What softmax does

After the output layer we have 10 raw scores, like `[-1.2, 0.4, 3.1, -0.8, ...]`. These are on no particular scale and don't sum to anything meaningful. Softmax squashes them into proper probabilities that add up to 1.0:

```
Before softmax: [-1.2,  0.4,  3.1, -0.8, ...]
After softmax:  [ 0.01, 0.06, 0.82, 0.02, ...]   ← sums to 1.0
```

So the final output is "I'm 82% confident this is a 2, 6% it's a 1, 2% it's a 3..." etc.

---

## What loss is

Loss is a single number that answers: *how wrong was the prediction?*

We use **cross-entropy loss**, which is simply:

```
loss = -log( probability assigned to the correct class )
```

Examples:
- Network says 95% chance of correct digit → `loss = -log(0.95) ≈ 0.05` (tiny loss, good job)
- Network says 10% chance of correct digit → `loss = -log(0.10) ≈ 2.30` (big loss, bad job)
- Network says 1% chance of correct digit → `loss = -log(0.01) ≈ 4.60` (huge loss)

When training starts, the network is random and loss is usually around 2.3 (which is `-log(1/10)` — basically random guessing). As training progresses, you should see it drop toward 0.3 or lower.

**Where to see this:** The loss number in the "Train in Browser" tab updates live every batch. The chart shows it falling over epochs.

---

## How training works, step by step

This is the core loop. It repeats thousands of times.

### Step 1 — Forward pass
Run a batch of images through the network and collect the predicted probabilities. This is literally just the math above: matrix multiply, ReLU, matrix multiply, softmax.

**In code:** `forward()` in `packages/web/src/train.ts`

### Step 2 — Compute loss
Compare predictions to the true labels. How wrong were we, on average across the batch?

**In code:** `crossEntropy()` in `train.ts`

### Step 3 — Backpropagation
This is the important one. Working backwards through the network, figure out: for each weight, did increasing it make the loss go up or down? And by how much?

This is calculus (the chain rule), but the code in `backward()` in `train.ts` does it explicitly — you can read exactly what's happening for each layer without any framework.

The key intuition: the gradient for the output layer is just `predicted - truth`. If the network said 82% for digit 2 but the answer was 7, the gradient pushes the "2" score down and the "7" score up. That error signal flows backward through W2, through the ReLU, through W1.

**In code:** `backward()` in `packages/web/src/train.ts`

### Step 4 — Update weights (Adam optimizer)
Nudge every weight slightly in the direction that reduces loss. The amount to nudge is the **learning rate** — a small number like `0.001`.

We use **Adam** (not plain gradient descent) because it's much better in practice. Adam keeps a running average of recent gradients, so it moves confidently in consistent directions and slows down when gradients are noisy. Think of it as gradient descent with memory.

**In code:** `adamStep()` in `train.ts`

### Then repeat
Shuffle the training data, grab the next batch of 32 images, and do it all again. One full pass through all the training data is called an **epoch**.

---

## The difference between the two networks

This project has two trained networks:

**Browser network** (`train.ts`): 784 → 128 → 10 = **102,410 parameters**, trained on 500 images in your browser. Reaches roughly 85–92% accuracy. Trains in about 30–60 seconds.

**JAX network** (`train.py`): 784 → 1024 → 512 → 10 = **1.3 million parameters**, trained on 60,000 images using Python. Reaches ~98% accuracy. The weights are already exported to `packages/web/src/assets/weights.json` — the Inference tab uses these.

The browser network is intentionally smaller so you can watch it train. The JAX network is what you draw against in the Inference tab.

---

## CPU vs WebGPU inference

On the Inference tab there's a toggle between CPU and WebGPU. Both run the same math — the difference is where.

**CPU mode** (`packages/core/src/index.ts`): Plain TypeScript for-loops. Works everywhere. Surprisingly fast for a single image because modern CPUs are very fast at sequential arithmetic, and our network is small.

**WebGPU mode** (`packages/webgpu/src/index.ts`): Sends the weight matrices to your graphics card and runs everything in parallel using compute shaders (written in WGSL, a language similar to GLSL). Each output neuron is computed simultaneously. The overhead of sending data to the GPU means it's often *not faster* for a single image — but for a video stream or a much larger network, the GPU wins decisively.

---

## Things to change to learn more

Everything below is a safe experiment — nothing will break, and you can always undo it.

---

### Change the hidden layer size

**File:** `packages/web/src/train.ts`, line 11

```typescript
export const HIDDEN = 128;
```

Try `32` — trains faster, lower accuracy (~80%). Try `256` — trains slower, slightly higher accuracy. Try `16` — watch how low accuracy goes with very few neurons. Notice that going from 16 to 128 helps a lot, but going from 128 to 256 barely helps — there's a point of diminishing returns.

---

### Change the number of training epochs

**File:** `packages/web/src/TrainView.tsx`, line 15

```typescript
const TOTAL_EPOCHS = 20;
```

Try `5` — fast, you'll see loss plateau. Try `50` — watch what happens after the loss stops improving much. After a certain point, more epochs on the same 500 images can cause *overfitting*: the network memorizes the training data instead of learning general patterns, and accuracy on new data stops improving or gets worse.

---

### Change the learning rate

**File:** `packages/web/src/TrainView.tsx`, line 17

```typescript
const LEARNING_RATE = 0.001;
```

Try `0.01` — loss will drop faster at first but might bounce around or diverge (go up instead of down). Try `0.0001` — loss will drop very slowly, smoothly. Too low and you need many more epochs to get anywhere. The ideal learning rate is a classic hyperparameter tuning problem in ML.

---

### Change the batch size

**File:** `packages/web/src/TrainView.tsx`, line 16

```typescript
const BATCH_SIZE = 32;
```

Try `1` — each weight update is based on a single image. This is very noisy (the loss chart will be jagged) but each update is fast and the model can sometimes find good solutions. Try `128` — smoother updates, each batch gives a better estimate of the true gradient, but fewer weight updates per epoch. Try `500` (the full dataset) — this is called "batch gradient descent." Very stable but very slow progress.

---

### Change the training data size

**File:** `train.py`, near the export section — change `50` per class to a different number, then re-run `make train` to regenerate `train_data.json`. Alternatively, modify the inline Python that generated it originally.

Try 10 per class (100 total) — accuracy will be noticeably worse, the network barely has enough variety to learn from. Try 200 per class (2000 total) — accuracy will climb and you'll see the value of more data clearly.

---

### Add a second hidden layer (harder)

**File:** `packages/web/src/train.ts` — the current network is 784 → 128 → 10. To add a second hidden layer (784 → 128 → 64 → 10), you'd need to:
1. Add `W3` and `b3` to `MLPWeights`
2. Add another matVecAdd + relu step in `forward()`
3. Add another layer's backprop in `backward()`
4. Add the new weights to `initWeights()` and `initAdamState()`

This is a meaningful exercise because you'll see exactly how backprop generalizes to more layers — the same pattern repeats.

---

## The big picture in one diagram

```
Training (browser, train.ts)
─────────────────────────────────────────────────────────────────
 500 labeled images
        ↓
 [Forward pass] → predicted probabilities
        ↓
 [Loss] → how wrong were we?
        ↓
 [Backward pass] → which weights caused the error?
        ↓
 [Adam update] → nudge those weights
        ↓
 repeat 20 epochs
        ↓
 102,410 weights with good values


Inference (web app, core/index.ts or webgpu/index.ts)
─────────────────────────────────────────────────────────────────
 Your drawing (280×280 canvas)
        ↓
 Crop + scale to 28×28
        ↓
 Flatten to 784 numbers [0–1]
        ↓
 [Forward pass only — no training, just math]
        ↓
 10 probabilities → pick highest → "I think that's a 7"
```

---

## What to read next in this repo

- `packages/web/src/train.ts` — the entire training algorithm in ~200 lines of plain TypeScript. Every function has a comment explaining what it does and why.
- `packages/core/src/index.ts` — the inference-only version (even simpler, no backprop needed).
- `packages/webgpu/src/index.ts` — the GPU version, interesting if you're curious about compute shaders.
- `train.py` — the Python/JAX version; the same algorithm, but JAX handles backprop automatically so you only write the forward pass.

---

## Further reading

Ordered roughly from "start here" to "go deeper."

### Videos — watch these first

**[3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** (4 videos, ~1 hour total)
The single best introduction to neural networks that exists. Grant Sanderson animates exactly what weights, layers, and backpropagation mean geometrically. Watch these before reading anything else. The third video on backpropagation is especially relevant — it's exactly what `backward()` in `train.ts` implements.

**[Andrej Karpathy: Building micrograd from scratch](https://www.youtube.com/watch?v=VMj-3S1tku0)** (~2.5 hours)
Karpathy (formerly of OpenAI/Tesla) builds a tiny neural network training library from absolute zero in Python, explaining every line. This is the best "how does backprop actually work in code" resource available. After watching this, `train.ts` will feel very familiar — micrograd does the same thing.

**[Andrej Karpathy: The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=PaCmpygFfXo)**
A shorter companion to the above, focused specifically on the math of backpropagation without any library abstraction.

---

### Books — approachable ones first

**[Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com)** (free online)
The best book for someone coming from programming rather than math. Nielsen builds a digit classifier from scratch (almost identical to this project) and explains every concept in plain English with interactive diagrams. Chapters 1 and 2 cover exactly what this codebase does. Highly recommended as your first book.

**[Make Your Own Neural Network — Tariq Rashid](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608)** (~$10, also on Amazon)
Written for complete beginners, intentionally avoids heavy math. Rashid builds an MNIST classifier step by step and explains the intuition behind each part. Very short and readable. Good if Nielsen feels too dense.

**[Dive into Deep Learning — d2l.ai](https://d2l.ai)** (free online)
A full university-level textbook that's unusually practical — every concept has runnable code alongside it (in PyTorch, JAX, and TensorFlow). Has excellent chapters on the math foundations (linear algebra, calculus) if you want to understand the equations more formally. The [MLP chapter](https://d2l.ai/chapter_multilayer-perceptrons/index.html) covers exactly what this project builds.

**[Deep Learning — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org)** (free online)
The standard academic textbook. Dense, rigorous, and thorough. Not a beginner book — save it for after you're comfortable with the basics. Part 1 (Applied Math) is worth reading once you want to understand *why* the math works. Part 2 (Deep Networks) goes much deeper than this project.

---

### Interactive tools

**[TensorFlow Playground](https://playground.tensorflow.org)**
A browser toy where you can visually watch a neural network train on simple 2D datasets. You drag sliders to change the number of layers, neurons, learning rate, and activation function, and you see the decision boundary update in real time. The best tool for building intuition about what hidden layers and activation functions actually do.

**[CNN Explainer](https://poloclub.github.io/cnn-explainer/)**
An interactive visualization of a convolutional neural network (the kind used in most modern image classifiers). Shows you what each filter and activation looks like as an image passes through. Good next step after understanding the MLP in this project — CNNs are how you'd get from ~92% to ~99.7% on MNIST.

**[Distill.pub](https://distill.pub)**
A research journal that publishes ML papers as interactive web articles instead of PDFs. Everything is beautifully visualized and written to be understood. The articles on [attention](https://distill.pub/2016/augmented-rnns/), [feature visualization](https://distill.pub/2017/feature-visualization/), and [the building blocks of interpretability](https://distill.pub/2018/building-blocks/) are particularly good.

---

### Going deeper into the specific concepts here

**Backpropagation:**
The original 1986 paper is surprisingly readable: [Learning representations by back-propagating errors — Rumelhart, Hinton, Williams](https://www.nature.com/articles/323533a0). Nielsen's [chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) is a much gentler version of the same ideas.

**Adam optimizer:**
The original paper: [Adam: A Method for Stochastic Optimization — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980). Only 9 pages, well-written, and the algorithm in the paper is almost exactly the code in `adamStep()` in `train.ts`.

**WebGPU / compute shaders:**
[WebGPU Fundamentals](https://webgpufundamentals.org) — the best practical introduction to WebGPU written for web developers. The [compute shader tutorial](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) explains exactly what `packages/webgpu/src/index.ts` is doing.

**JAX (the Python training framework):**
[JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) — official intro. Short and practical. [You Don't Know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html) — a good blog post explaining what makes JAX different from NumPy and why `@jit`, `vmap`, and `grad` are powerful.

# Training and Inference — Going Deeper

Things that come up constantly but weren't covered in the basics. Each one answers a "wait, why does it do that?" question you'd hit quickly going further.

---

## The three buckets of data: train, validation, and test

When you train a model, you split your data into three groups and never mix them.

**Training set** — the data the model actually learns from. Weights are updated based on this data.

**Validation set** — data the model never trains on, but you check during training. After each epoch you run the model on the validation set and measure accuracy. This tells you how well it's generalizing to data it hasn't seen.

**Test set** — data you touch exactly once, at the very end, after all decisions are made. This is your honest final score.

Why three instead of two? Because the validation set gets "used up."

Here's what happens without a test set: you train a model, check validation accuracy, tune your hyperparameters to improve validation accuracy, train again, check validation accuracy again, tune again... After enough iterations, you've inadvertently optimized for the validation set. Your own choices — which hyperparameters to try, when to stop training — were guided by validation performance. The validation set has leaked into your decisions.

The test set is the data you've never made any decision based on. You only look at it once, when you're done, to get an honest measure of real-world performance.

**Data leakage** is when information from the validation or test set accidentally influences training. It can be subtle — normalizing your data using statistics computed from the full dataset (including test) instead of just the training set, for example. Leaked models look great in evaluation and fail in production. It's one of the most common mistakes in ML.

In this project: MNIST comes pre-split into 60K training and 10K test images. The 10K test set is what `train.py` reports accuracy on after each epoch. In a more rigorous setup, you'd hold out some of those 60K training images as a validation set too, and only touch the 10K test set at the very end.

---

## What the weights actually learn to represent

After training, the weights in a neural network encode something real. It's not random noise — you can visualize what early layers have learned and it's recognizable.

For image networks, the first layer's weights typically look like **edge detectors** — small patterns that respond to horizontal edges, vertical edges, diagonal edges, and color gradients. The network discovered these on its own from data, but they match what vision scientists had previously found by studying actual neurons in animal visual cortex.

The second layer combines those edges into **curves and corners**. The third layer combines those into **object parts** — something like "eye-shaped region" or "wheel-shaped region." Later layers respond to whole objects.

This hierarchy — pixels → edges → parts → objects — is not designed in. It emerges from training with backpropagation on enough data. The same general hierarchy appears across different architectures trained on different datasets. It seems to be something real about how visual information is structured, not an artifact of the training approach.

For language models, you can do similar visualizations of what different attention heads learned. Some heads reliably attend to grammatical subjects. Some track coreference (which pronoun refers to which noun). Some track positional relationships. Again, not designed — emerged from predicting the next word.

This is one of the most striking things about neural networks: they discover interpretable structure without being told to look for it. The structure was always in the data; the network learns to represent it.

---

## Vanishing and exploding gradients — why deep networks were hard

This is the problem that made "deep learning" difficult for 25 years and explains why many modern techniques exist.

Remember that training works by sending an error signal backwards through the network (backpropagation). Each layer passes the signal to the layer before it. But at each layer, the signal gets multiplied by the weights.

If your weights are slightly larger than 1 — say, 1.1 — then after 50 layers, the gradient has been multiplied by 1.1 fifty times: 1.1^50 ≈ 117. The gradient has **exploded**. The weight updates become enormous and the training process blows up.

If your weights are slightly smaller than 1 — say, 0.9 — then after 50 layers: 0.9^50 ≈ 0.005. The gradient has **vanished**. By the time the error signal reaches the early layers, it's essentially zero. Those layers stop learning entirely.

This is why deep networks were nearly impossible to train before the right tools existed. The gradients either exploded or vanished, and there was no obvious fix.

Several innovations specifically address this problem:

**Xavier/Glorot initialization** (what this project uses) — carefully sets initial weight values so that gradients start in a healthy range. Not too big, not too small. This is why the initialization formula involves the size of the layer — bigger layers need smaller initial weights to keep the signal stable.

**ReLU activation** — the old activation functions (sigmoid, tanh) had gradients that got very small when inputs were large or small. ReLU (just max(0, x)) has a gradient of exactly 1 for positive inputs and 0 for negative inputs. No squashing. This keeps gradients from vanishing as easily.

**ResNets / skip connections** (2015) — instead of forcing information to flow through every layer sequentially, add direct "shortcuts" that jump over layers. The gradient can flow back through the shortcut, bypassing any problematic layers. This is why 100-layer networks became trainable while 20-layer networks had been the practical limit.

**Batch normalization / layer normalization** — see below.

When you encounter any of these techniques, they exist because of vanishing/exploding gradients. The problem shaped the entire history of deep learning.

---

## Normalization layers — stabilizing training

**Batch normalization** was introduced in 2015 and immediately made deep networks much easier to train. The idea: after computing the activations in a layer, force them to have a mean of zero and a standard deviation of one, then let the network scale and shift them as it sees fit.

Why does this help? Because without it, the distribution of values flowing through the network changes constantly as training progresses. Layer 5's weights were set based on what it was receiving from layer 4 yesterday. But today layer 4's weights have changed, so layer 5 is receiving different-looking inputs even though it hasn't changed. It's like trying to hit a moving target. Normalization stabilizes the target.

**Layer normalization** is a variant used in transformers and language models. Instead of normalizing across the batch, it normalizes across the features within a single example. Transformers use layer norm because they process variable-length sequences, which makes batch normalization awkward.

You don't need to know the math. The practical effect: training is faster, more stable, and you can use higher learning rates. Almost every serious model built after 2015 uses some form of normalization.

---

## Dropout — randomly turning neurons off during training

Dropout is one of the most counterintuitive things in deep learning. During training, you randomly set some fraction of neurons to zero — typically 20–50% of them — on every forward pass. The network has to learn to make correct predictions even though random parts of it are disabled.

Why would breaking your network on purpose help it learn?

Because it forces redundancy. If any single neuron knows it can be turned off at any moment, it can't become the sole holder of important information. Multiple neurons have to learn to represent the same concept in different ways. The network can't over-rely on any one feature.

The result is a model that generalizes better. Dropout acts as a form of regularization — a technique to prevent overfitting.

**The important practical detail:** dropout is only active during training. During inference, all neurons are on. But their outputs get scaled to compensate — if a neuron was dropped 50% of the time during training, its output gets multiplied by 0.5 during inference so the total signal magnitude is similar. Modern frameworks handle this automatically.

This is why there's a distinction between "training mode" and "evaluation mode" in most ML frameworks — and why forgetting to switch to evaluation mode is a common bug that makes your production model subtly worse than your validation scores suggested.

---

## Training mode vs inference mode

In code you'll often see something like:

```python
model.train()   # before training
model.eval()    # before running inference
```

Some layers behave differently depending on which mode you're in:

**Dropout** — active during training, disabled during inference (as described above).

**Batch normalization** — during training, it normalizes using the statistics of the current batch (mean and variance of that batch). During inference, it uses running averages accumulated over the entire training process. You want inference to be consistent and deterministic, not dependent on whatever batch happened to come in.

Forgetting to call `model.eval()` before running inference is a real bug. Your model will behave randomly because dropout is still firing, and batch norm will be using batch statistics instead of the stable running averages. Predictions will be wrong and inconsistent. The model might look fine on average but be unreliable on individual predictions.

In this project's TypeScript inference code, there's no dropout or batch norm, so this distinction doesn't apply. But in any serious model you use (including DNABERT, ESM-2), this matters.

---

## Embeddings — what it means for things to be "close"

An embedding is a way of placing things in space such that similar things end up near each other.

Take words. You could represent each word as an integer: "cat" = 4271, "dog" = 4272. But those numbers don't mean anything — "cat" is not slightly more than "dog." The integers are just arbitrary IDs.

An embedding gives each word a vector — a list of, say, 512 numbers. After training, words with similar meanings end up with similar vectors. "Cat" and "dog" are close in embedding space. "Cat" and "democracy" are far apart.

What makes this powerful: you can do arithmetic. The famous example is that `king - man + woman ≈ queen` in a well-trained word embedding. "King" minus the "maleness" direction plus "femaleness" lands near "queen." The geometric relationships encode semantic relationships.

For DNA: embedding a DNA sequence means learning a vector that captures what role that sequence plays. Two promoter sequences from different genes end up near each other. A coding region and a regulatory region end up in different parts of space. The model learns this structure just from predicting what comes next in genomic sequences.

For proteins: ESM-2 produces embeddings where proteins with similar functions cluster together, proteins from the same evolutionary family cluster together, and proteins that interact with each other tend to be near each other — without any of those relationships being explicitly labeled in training data.

The embedding space is the network's internal representation of what things mean. Everything the network knows is encoded in where it placed things in this space.

---

## The learning rate schedule — you don't keep the same rate throughout

In practice, the learning rate is not constant during training. It's changed according to a schedule.

**Warmup:** start with a very small learning rate and increase it over the first few epochs. Reason: at the beginning, your weights are random and the gradients can be large and inconsistent. A big learning rate early on causes chaotic updates. Warming up slowly lets the weights get into a reasonable range before taking big steps.

**Decay:** reduce the learning rate as training progresses. Later in training, you're close to a good solution and want to fine-tune rather than take big steps that might overshoot. Common approaches: reduce it by half every N epochs, or reduce it smoothly following a cosine curve.

**Cosine schedule** (most popular): the learning rate follows the top half of a cosine wave — starting at max, falling smoothly to near zero. Popular because it's smooth and the slow decay at the end gives the model time to settle into a good final state.

Large models (like GPT-style language models) essentially always use warmup + cosine decay. The specific schedule affects final performance noticeably.

In this project's browser training, the learning rate is fixed at 0.001 for simplicity. For 20 epochs on 500 images, it doesn't matter much. For a real training run, you'd want a schedule.

---

## Temperature — controlling how confident the model sounds

This one is specific to language models but you'll encounter it quickly.

When an LLM generates text, it picks the next token by sampling from the probability distribution over all possible tokens. The probabilities might be: "the" = 40%, "a" = 20%, "my" = 15%, ...

**Temperature** is a number that controls how peaked or flat this distribution is:

- **Temperature = 1.0** — use the probabilities as-is
- **Temperature < 1.0** (e.g. 0.2)— flatten the probabilities toward the most likely token. At temperature 0, you always pick the most likely token. Output is deterministic and repetitive but "safe."
- **Temperature > 1.0** (e.g. 1.5) — spread the probabilities more evenly. The model becomes more random, sometimes picking unlikely tokens. Output is more creative and surprising but less coherent.

When you use ChatGPT for creative writing vs. factual answers, it's likely using different temperature settings. Code generation often uses low temperature (you want correctness, not creativity). Story generation uses higher temperature.

For DNA and protein generation, temperature controls how closely the generated sequences follow patterns in the training data. Low temperature = sequences that look very similar to known real sequences. High temperature = more novel, potentially more diverse, but also more likely to produce non-functional sequences.

---

## What inference actually costs

Training is expensive once. Inference is expensive forever — every time a user asks a question, every time you run a prediction, every time you process a new DNA sample.

A few practical concepts:

**Latency vs throughput:** Latency is how long a single prediction takes. Throughput is how many predictions you can do per second. These are often in tension. Processing one sample at a time minimizes latency. Batching many samples together increases throughput but each individual sample waits longer.

**Quantization:** Model weights are usually stored as 32-bit floating point numbers. Quantization reduces them to 8-bit integers (or even 4-bit). This makes the model 4× smaller and often 2–4× faster, with a small accuracy cost. Most deployed models are quantized. When you see "Q4" or "Q8" in a model name, that's quantization.

**KV cache (for transformers):** When generating text token by token, the transformer has to compute attention over all previous tokens for each new token. This is wasteful — the attention computations for earlier tokens don't change. The KV cache stores those computations so they don't have to be redone. This makes generation dramatically faster but requires memory proportional to the context length.

**Model size vs context length tradeoff:** Larger models understand better but are slower and more expensive. Longer context windows let you feed in more information but increase memory usage quadratically (attention is O(n²) in sequence length without special techniques). These tradeoffs govern almost every deployment decision in production ML.

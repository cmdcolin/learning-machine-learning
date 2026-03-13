# Glossary

Plain English definitions. No formalism, no assumed knowledge. Written to build intuition, not to be technically exhaustive.

---

## Loss

You're learning to throw darts. You throw one, it lands 8 inches from the bullseye. That distance — 8 inches — is the loss. It's just a measure of how wrong you were.

In a neural network, after every prediction, we compute a single number that says "here's how wrong that was." Training is the process of making that number go down over time. When loss is near zero, the network is making good predictions. When loss is high, it's confused.

The specific formula changes depending on what kind of problem you're solving, but the concept is always the same: one number, measuring wrongness.

---

## Loss function

The specific recipe for computing loss. Different problems use different recipes.

**Cross-entropy** — used when you're picking one answer from a list (like which digit is this). Measures how much probability you gave the correct answer. If you said "90% chance it's a 7" and it was a 7, loss is low. If you said "2% chance it's a 7" and it was a 7, loss is high.

**Mean squared error (MSE)** — used when you're predicting a number rather than a category. House price, temperature, patient age. You take your prediction, subtract the real answer, square it. Squaring means big mistakes are punished much harder than small ones — being off by 10 is 25 times worse than being off by 2, not just 5 times worse.

**Binary cross-entropy** — same as cross-entropy but for yes/no questions. Is this spam or not? Does this patient have the disease or not? Just two options instead of many.

**Contrastive loss** — used when you're not predicting a label but rather trying to learn that two things are similar or different. "These two photos are the same person, those two are different people." Used a lot in biology — comparing protein sequences, finding similar DNA regions.

---

## Gradient

The gradient is the answer to: "if I nudge this weight slightly, does the loss go up or down, and by how much?"

Imagine you're blindfolded on a hilly landscape and you want to find the lowest point. You can't see anything, but you can feel the ground under your feet. If it slopes down to your left, you step left. If it slopes up in all directions, you're at the bottom.

The gradient is that slope — for every weight in the network simultaneously. It tells you which direction to step to reduce loss.

---

## Gradient descent

The process of repeatedly following the gradient downhill. Compute which direction reduces loss, take a small step that way, compute again, step again. Keep going until you're at a low point.

"Descent" means going downhill. The "hill" is the loss landscape — a surface in very high-dimensional space where the height at any point represents how wrong the network is with those particular weight values. Training is finding a low point on that surface.

---

## Backpropagation (backprop)

The algorithm for computing the gradient efficiently.

You have a million weights. After a wrong prediction, which ones were responsible? Backprop works backwards through the network using the chain rule from calculus — starting at the output (where the error is measured) and working back through each layer, figuring out how much each weight contributed to the mistake.

The result is a gradient for every single weight: "this weight should go up a little, that one should go down a lot, this other one barely matters." This is what makes training possible — without backprop, you'd have no efficient way to figure out which of your million weights to change.

In code: `backward()` in `packages/web/src/train.ts`. About 40 lines of explicit math, no magic.

---

## Weight

A single adjustable number inside the network. The network's "knowledge" is entirely stored in its weights — nothing else persists between predictions.

Before training, weights are random. After training, they encode everything the network learned. The entire point of training is to find good values for the weights.

In the browser demo, there are 102,410 weights. In the JAX model, 1.3 million. In GPT-4, roughly 1.8 trillion.

---

## Bias

A weight that doesn't connect to any input — it's just added to the output of a neuron unconditionally. Every neuron has one.

If weights are the volume knobs on a mixing board (controlling how much each input matters), biases are the baseline setting when everything is turned to zero. They let neurons activate even when all their inputs are zero, which turns out to be important for learning.

---

## Neuron

A single unit in the network that takes some inputs, multiplies each by a weight, adds them all up, adds its bias, and passes the result through an activation function.

The word "neuron" is borrowed from biology but is misleading. It's just a number being computed. There's no intelligence in a single neuron, the same way there's no intelligence in a single transistor.

---

## Layer

A group of neurons that all process the same input and produce outputs that become the next layer's input. The network is just layers stacked on top of each other.

The first layer processes raw pixels (or raw tokens). The last layer produces the final prediction. Everything in between learns progressively more abstract representations of the input.

---

## Activation function

After summing up the weighted inputs, a neuron passes the result through an activation function before sending it to the next layer.

Without activation functions, stacking a hundred layers would be mathematically identical to having one layer — all the matrix multiplications would collapse into a single one. Activation functions break this by introducing nonlinearity, which is what allows the network to learn complex patterns instead of just linear combinations.

**ReLU** (most common): if the value is negative, replace it with zero. If positive, leave it alone. `max(0, x)`. Simple, but surprisingly effective.

**Softmax** (used at the output): takes a list of numbers and squashes them into probabilities that sum to 1.0. Converts raw scores into "I'm 82% confident this is a 2."

---

## Forward pass

Running an input through the network from the first layer to the last to get a prediction. Just doing the math in the forward direction: input → weights → activation → next layer → ... → output.

No learning happens during the forward pass. It's just computing a prediction.

---

## Backward pass

Running the error signal backwards through the network to compute the gradient (see: backpropagation). This is where the learning information is generated — figuring out how each weight contributed to the mistake.

---

## Epoch

One complete pass through the entire training dataset. If you have 500 training images and a batch size of 32, one epoch is about 16 batches.

Training typically runs for many epochs — the same data shown to the network many times. Each time through, the network is slightly better at the task, so it extracts slightly different information from the same examples.

---

## Batch

A small subset of the training data processed together before updating the weights. Instead of updating after every single example (too noisy) or after seeing all the data (too slow and memory-intensive), you process a batch of e.g. 32 examples, average the gradients, then update.

Batch size is a hyperparameter. Small batches = noisy but fast updates. Large batches = smooth but slow updates.

---

## Learning rate

How big a step to take when updating weights. The single most important hyperparameter.

Too high: the network overshoots the good values, bounces around, and might get worse instead of better. Loss curve is jagged or diverges upward.

Too low: the network moves so slowly it might not get anywhere useful in a reasonable number of epochs. Loss barely moves.

Just right: loss falls smoothly and steadily. There's no formula for finding this — you experiment.

Typical values: 0.001 for Adam, 0.01 for plain SGD. These are starting points, not rules.

---

## Optimizer

The algorithm that uses the gradient to actually update the weights. The gradient says "go this direction" — the optimizer decides exactly how to take that step.

**SGD (Stochastic Gradient Descent)** — the simplest possible optimizer. Multiply gradient by learning rate, subtract from weight. Done.

**Adam** — the most commonly used optimizer. Keeps a running average of recent gradients and their sizes, which lets it adapt the effective learning rate for each weight individually. Weights that have been getting consistent gradients get bigger steps. Weights with noisy gradients get smaller steps. Almost always converges faster than plain SGD.

**AdamW** — Adam with weight decay added in. Weight decay is a small penalty for having large weights, which prevents the model from over-specializing and helps it generalize better.

---

## Overfitting

The network memorizes the training data instead of learning the general pattern.

Imagine studying for an exam by memorizing every practice question and answer word-for-word, rather than understanding the underlying concepts. You'd ace any question you'd seen before and fail anything new.

A network that's overfit gets near-perfect accuracy on training data but poor accuracy on data it hasn't seen. Signs: training accuracy keeps climbing but test accuracy plateaus or falls. Cause: too many epochs, too little data, too large a network for the amount of data.

---

## Underfitting

The opposite — the network hasn't learned enough. It performs poorly even on the training data. Usually caused by too few epochs, too small a network, or too high a learning rate preventing convergence.

---

## Parameters

All the weights and biases in the network. The network's total "knowledge storage." When people say a model has "7 billion parameters" they mean 7 billion individual numbers that were learned during training.

More parameters = more capacity to learn complex patterns. But also more data needed, more compute needed, and more risk of overfitting on small datasets.

---

## Hyperparameters

The settings you choose that aren't learned by the network — they're fixed before training starts. Learning rate, batch size, number of layers, number of neurons per layer, number of epochs.

Picking good hyperparameters is mostly done by experimentation. There's no formula. The skill of "hyperparameter tuning" is essentially: try a thing, look at the loss curve, reason about what it means, adjust.

---

## Embedding

A way to turn a discrete symbol (a word, a DNA base, an amino acid) into a vector of numbers that the network can work with.

You can't feed the letter "A" directly into a matrix multiplication — you need numbers. An embedding table is just a lookup table: for each possible symbol, there's a learned row of numbers. Similar symbols end up with similar vectors after training.

In DNA models, A, T, G, and C each get their own embedding vector. The network learns that A and T are "complementary" and that ATG (the start codon) forms a meaningful unit — not because you told it, but because those patterns are in the training data.

---

## Token

The basic unit of input to a language model. Could be a character, a word, a piece of a word, or (for DNA) a single base or k-mer.

Tokenization is the step of converting raw text or sequence into a list of tokens before feeding it to the model. The vocabulary is the set of all possible tokens — 4 for single-base DNA, ~50,000 for typical English word-piece models.

---

## Transformer

The architecture used by essentially every modern language model. The key innovation is the **attention mechanism** — instead of processing one token at a time in sequence, every token can directly look at every other token and decide how relevant it is.

Before transformers, models had to compress everything seen so far into a fixed-size "memory" vector. Long-range context got lost. With attention, a word at position 1 and a word at position 800 can directly influence each other, with no information bottleneck in between.

This is why transformers are so good at DNA and protein sequences — regulatory elements and structural signals that are far apart in sequence can still interact directly.

---

## Attention

The mechanism inside a transformer that lets each token decide which other tokens to pay attention to.

For every token, attention computes a weighted average of all other tokens' representations — but the weights are learned, so the model figures out on its own which tokens are relevant to which. A verb can learn to attend heavily to its subject. A codon can learn to attend to the start codon. A regulatory element can learn to attend to the gene it controls.

---

## Pre-training and fine-tuning

**Pre-training**: training a large model on a massive general dataset without any specific task in mind. A DNA model pre-trained on billions of base pairs learns the general statistical structure of DNA — codon usage, conserved motifs, sequence context.

**Fine-tuning**: taking a pre-trained model and training it further on a small, specific dataset for a specific task. "Does this promoter sequence drive high or low expression?" You don't have to learn DNA from scratch — the pre-trained model already understands DNA. You just teach it the specific task.

This is why pre-trained models like DNABERT-2 and ESM-2 are so useful. Someone else spent enormous compute pre-training them. You can fine-tune on your specific biological question with a relatively small labeled dataset.

---

## Inference

Using a trained network to make a prediction on new data. Not learning — just running the forward pass with fixed weights.

In this project: you draw a digit, it runs through the network, you get a prediction. The weights don't change. That's inference.

---

## Cross-entropy (for LLMs)

In a language model, after every token the model predicts the probability of every possible next token. Cross-entropy loss measures how much probability the model gave to the token that actually came next.

If the model said "90% chance the next word is 'mat'" and the next word was "mat" — low loss. If it said "1% chance" and the word was "mat" — high loss.

Every token in the training data is its own prediction with its own loss. A language model trained on a trillion tokens is doing this calculation a trillion times, averaging it all, and slowly adjusting a billion+ weights to make the average loss go down.

When loss goes down in an LLM, it means the model is getting better at predicting what comes next in real text. For English, that means learning grammar and facts. For DNA, it means learning the statistical patterns of biological sequences.

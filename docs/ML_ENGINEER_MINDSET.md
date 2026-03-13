# The Machine Learning Mindset: A Programmer's Guide

This guide breaks down the core concepts of Machine Learning (ML) using technical language familiar to TypeScript developers, focusing on the "Why" and "How" of the code in `train.py`.

## 1. Differentiable Programming: The Core "Mindset"

The biggest shift from standard programming to ML is the concept of **Differentiable Programming**.

In standard programming, functions are black boxes of logic. In ML, we ensure our functions are **mathematically differentiable**. This means for any function $f(x)$, we can calculate exactly how a small change in $x$ affects the output.

### Why JAX?
JAX is a "Transformational" library. Its most powerful tool is `jax.grad()`.
- **In TypeScript terms**: Imagine if you could pass any complex function into a `getGradient()` helper, and it returned a *new* function that calculates the exact slope (derivative) of that function at any point.
- **The Loop**: We use these gradients to perform **Gradient Descent**. If the slope is positive, we decrease the input; if negative, we increase it. This "nudges" the model towards the correct answer.

## 2. The Python Code (Explained for TS Devs)

Here is a breakdown of the critical sections in `train.py`.

### Model Forward Pass
This is the "Inference" logic. It translates a raw input vector into a prediction.

```python
# 'params' is a list of [Matrix, Vector] pairs representing each layer.
def predict(params, image):
    activations = image
    
    # Iterate through all layers except the last one
    for w, b in params[:-1]:
        # matrix multiplication (jnp.dot) is like a massive nested loop
        # mapping every input neuron to every output neuron.
        outputs = jnp.dot(w, activations) + b
        
        # ReLU: jnp.maximum(0, x). Standard non-linear activation.
        activations = relu(outputs)
    
    # Final Layer (Output)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    
    # logsumexp: A numerically stable way to compute probabilities.
    return logits - jax.nn.logsumexp(logits)
```

### The Training Step
This is where the "Learning" happens.

```python
@jit # Compiles this function to optimized machine code (XLA)
def update(params, opt_state, x, y, rng):
    # 1. Calculate the 'gradient' of our loss function.
    # grad(loss) returns a function that tells us how to change 'params'
    # to reduce the error.
    grads = grad(loss)(params, x, y)
    
    # 2. Use the Optimizer (AdamW) to calculate the specific 'nudges'.
    # AdamW handles things like 'momentum' (speeding up in consistent directions).
    updates, opt_state = optimizer.update(grads, opt_state, params)
    
    # 3. Apply the nudges to create NEW parameters.
    new_params = optax.apply_updates(params, updates)
    
    return new_params, opt_state
```

## 3. Math Functions Involved

| Function | What it does (Programmer's View) | Why we use it |
| :--- | :--- | :--- |
| **Dot Product** (`jnp.dot`) | A weighted sum of all inputs. | It's the "link" between layers. Every output neuron "looks" at every input neuron. |
| **ReLU** | `Math.max(0, x)` | Without this, a 100-layer network is mathematically identical to a 1-layer network. It allows the model to learn complex patterns. |
| **Softmax** | Normalizes a vector so all numbers sum to 1.0. | Converts raw scores (logits) into "confidence" percentages (probabilities). |
| **Cross-Entropy** | Calculates the "distance" between two probability distributions. | It's our "Error Metric." It gives a high penalty for being "confidently wrong." |

## 4. Why this isn't "Magic"

As an ML engineer, your mindset shouldn't be "The computer is thinking." Instead, think of it as **"High-Dimensional Optimization."**

You are defining a complex landscape (the Loss Function) and trying to find the lowest point in that landscape (the minimum error). JAX provides the "GPS" (Gradients) and the "Vehicle" (JIT/XLA) to get there efficiently.

### Mindset Checklist for Devs:
1. **Data is Code**: If the model is failing on '7', your "bug" isn't a syntax error; it's likely that your training data doesn't have enough variety in how '7' is drawn.
2. **Parameters are State**: 1.3 million variables are the "state" of your app. Training is just a state-machine that updates them based on feedback.
3. **Purity Matters**: JAX requires "pure" functions (no side effects). This is why we pass around `PRNGKey` (rng) explicitly for randomness.

# JAX vs PyTorch vs TensorFlow

A plain-English comparison. All three do the same fundamental thing — run matrix math on GPUs and compute gradients automatically — but they have very different personalities.

---

## The short version

**PyTorch** — what almost everyone uses today. Feels like normal Python. Best ecosystem. Where all the research is published.

**TensorFlow** — Google's older framework. Was dominant until ~2019. Still strong in production and mobile. Has the only serious JavaScript version.

**JAX** — Google's newer, lower-level tool. Not really a deep learning framework so much as a math transformation library. Powerful but you have to build more yourself. Used in serious research.

---

## PyTorch

Made by Facebook (Meta). Took over as the dominant framework around 2019–2020 and hasn't looked back.

**The personality:** It feels like regular Python. You write a loop, it runs line by line, you can print things and use a debugger exactly like any other Python program. This was a big deal because the alternative at the time (TensorFlow 1) was not like that at all.

**The mental model:** You define a model as a class, you write a training loop as a function, you call `.backward()` to compute gradients, you call `.step()` to update weights. The flow is explicit and readable.

```python
# This is what PyTorch training looks like
output = model(input)         # forward pass
loss = criterion(output, target)
loss.backward()               # backprop
optimizer.step()              # update weights
optimizer.zero_grad()         # clear gradients for next step
```

**Why most people use it:**
- Every research paper comes with PyTorch code
- The largest ecosystem of pre-built models (Hugging Face is essentially built on PyTorch)
- Easy to debug — when something breaks, the error message points to the actual line
- Feels natural to anyone who already knows Python

**The downsides:**
- Performance can lag JAX on some hardware configurations
- The flexibility that makes it easy to debug can make it harder to optimize for production

**Who uses it:** basically all of academic ML research, most startups, most open-source models including almost every LLM, DNA model, and protein model you'll encounter.

---

## TensorFlow

Made by Google. Was the dominant framework from roughly 2015 to 2019.

**The original personality (v1):** You first *defined* a computation graph — described all the math you wanted to do — and then separately *ran* it. The graph was like a blueprint; running it was like building the house. This was fast and good for deployment but deeply confusing for beginners and very hard to debug.

```python
# TensorFlow v1 — you had to think about it completely differently
x = tf.placeholder(tf.float32)      # not a value, a "slot"
y = tf.matmul(x, weights) + bias    # not computed yet, just described
with tf.Session() as sess:
    result = sess.run(y, {x: data}) # NOW it actually runs
```

If you've ever heard people complain that TensorFlow was confusing, it was this.

**After 2019 (v2):** TensorFlow added eager execution — it now runs line by line just like PyTorch. The core experience became much more similar. Keras (a friendlier high-level API) was folded in as the default interface.

**Why it still matters:**
- **TensorFlow.js** — the only major ML framework with first-class JavaScript support. You can run models in the browser or Node.js without Python. If you're building ML into a web app, this is the realistic path.
- **TFLite** — for running models on mobile phones and embedded devices. Strong production tooling.
- **Enterprise adoption** — large companies that built TF pipelines years ago haven't switched.

**The downsides:**
- Lost most of the research community to PyTorch
- Ecosystem is smaller for cutting-edge models
- API has layers of historical confusion (Keras on top of TF2 on top of TF1 concepts)

**Who uses it:** production ML systems at large companies, mobile/edge deployment, web ML (TensorFlow.js), and people who got into ML before 2019.

---

## JAX

Also made by Google, but different in nature from TensorFlow. JAX isn't really a deep learning framework — it's a library for transforming mathematical functions. You use it to build deep learning frameworks.

**The personality:** Functional and mathematical. JAX doesn't give you layers, optimizers, or training loops. It gives you three core operations that transform functions:

- `jit()` — compiles a Python function to fast machine code via XLA
- `grad()` — takes a function and returns a new function that computes its gradient
- `vmap()` — takes a function that works on one example and automatically vectorizes it to work on a batch

```python
# JAX thinking: you write math, JAX transforms it
def loss(params, x, y):
    prediction = predict(params, x)
    return -jnp.mean(prediction * y)

# grad() returns a new function that computes the gradient of loss
gradient_fn = grad(loss)
grads = gradient_fn(params, x, y)  # now you have gradients
```

The training loop in `train.py` in this project is JAX. Notice how you write the math yourself and JAX handles the gradient computation.

**Why it's interesting:**
- The cleanest expression of the underlying math — you write exactly what you mean
- Very fast, especially on TPUs (Google's custom AI chips)
- Functional style avoids a class of bugs from mutable state
- Popular at Google DeepMind and in theoretical ML research

**Why it's harder:**
- You write much more yourself — no built-in layers, no optimizer, no training loop
- Debugging is harder: `jit()` compiles your code, which means Python debuggers and print statements don't work normally inside compiled functions
- The "no mutable state" rule is strict and unintuitive at first
- You need a library on top (Flax, Haiku, Optax) to do anything resembling a normal deep learning workflow

**Who uses it:** Google DeepMind, academic researchers who want fine-grained control, situations where XLA performance matters a lot.

---

## Side by side

| | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| Made by | Meta | Google | Google |
| Dominant since | ~2019 | 2015–2019 | Still niche |
| Feels like | Normal Python | Improved over time | Math notation |
| Ease of learning | Easiest | Medium | Hardest |
| Debugging | Easy | Medium | Hard |
| Ecosystem / pre-built models | Best | Good | Minimal |
| Research papers use | Almost always | Rarely now | Sometimes |
| JavaScript version | No | Yes (TF.js) | No |
| Mobile/edge | PyTorch Mobile | TFLite (better) | No |
| Built-in layers/optimizers | Yes | Yes (Keras) | No — use Flax/Optax |
| Speed | Good | Good | Best (on TPUs) |

---

## For your specific situation

**For learning ML concepts:** The framework barely matters. The concepts — loss, backprop, optimizer, layers — are identical across all three. The browser training in this project is pure TypeScript with no framework at all, which is actually the best way to understand what's happening.

**For DNA and protein models:** PyTorch. Every major model (DNABERT-2, ESM-2, HyenaDNA, EVO) is in PyTorch. Hugging Face runs on PyTorch. When you start fine-tuning pre-trained biological models, you'll be in PyTorch whether you want to be or not.

**For JavaScript / browser ML:** TensorFlow.js. It's the only real option for running or training models in JavaScript. Transformers.js (from Hugging Face) wraps many PyTorch models into a JS-friendly format and is increasingly useful.

**For this project specifically:** JAX was chosen because it's expressive for showing the math clearly and produces very fast training via XLA. But for anything you build beyond this learning project, you'll probably use PyTorch.

---

## The honest take

PyTorch won. Not because it's necessarily best at everything, but because the network effects of "everyone uses it so all the models are in it so everyone uses it" are very strong. The research community converged on it around 2019 and the lead has only grown since.

TensorFlow.js is the notable exception — if you want to do ML in JavaScript, it's the path, and it's genuinely good.

JAX is for people who want to write custom research code, work closely with the math, or run on Google's TPU hardware. It's not where you start.

If you're going to learn one Python ML framework, learn PyTorch. Every tutorial, every model, every paper implementation you encounter will be in PyTorch.

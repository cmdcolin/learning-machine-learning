# The Beginner's Guide to this MNIST Demo

If you know how to write a `for` loop in TypeScript but have no idea what "Machine Learning" or "JAX" is, this guide is for you.

## 1. What is Machine Learning (ML)?

In traditional programming, you write the rules:
```typescript
if (pixel[10][10] > 0.5 && pixel[11][11] > 0.5) {
  return "It might be a 7";
}
```
In **Machine Learning**, you don't write the rules. Instead, you create a "Model" (a giant set of variables) and show it 60,000 images of digits. The computer "learns" by adjusting its variables until it consistently guesses the right answer.

## 2. The "Model" (A Programmer's View)

Think of a Neural Network as a **very long function** that takes an array of 784 numbers (the 28x28 image pixels) and returns an array of 10 numbers (the "probability" for each digit 0-9).

Inside that function, the math is surprisingly simple. It's mostly:
`output = (input * weight) + bias`

- **Weights**: These are numbers that decide how "important" a pixel is.
- **Biases**: These are "offset" numbers to fine-tune the result.

Our model has about **1.3 million** of these numbers. "Training" is the process of finding the *perfect* values for those 1.3 million numbers.

## 3. What is Python & JAX?

- **Python**: In the ML world, Python is the "glue" language. It's not necessarily fast, but it has the best libraries for math.
- **JAX**: This is a library created by Google. Think of it as **"NumPy on Steroids."** It takes the math we wrote in Python and compiles it into highly optimized machine code (using a technology called XLA) so it can train the model in seconds instead of hours.

## 4. How does "Training" work?

Training is a big `for` loop (called **Epochs**):
1. **Guess**: The model looks at a batch of images and guesses what they are.
2. **Score**: We calculate the **Loss** (how wrong the guess was).
3. **Correct**: We use calculus (automatically handled by JAX) to figure out how to slightly nudge those 1.3 million numbers to make the error smaller next time.
4. **Repeat**: We do this thousands of times.

## 5. What is WebGPU?

Normally, when you run code in TypeScript, the **CPU** executes your instructions one by one.
- **CPU**: Like a very fast mathematician who can only do one sum at a time.
- **GPU**: Like 5,000 average mathematicians who can all work at the same time.

Since our model needs to do over a million multiplications for *every single drawing*, the CPU can get a bit slow. **WebGPU** allows our TypeScript code to send those 1.3 million numbers to the Graphics Card and say: *"Do all these multiplications at once and give me the answer."*

## 6. How the Monorepo fits together

1. **`train.py` (The Teacher)**: Runs once. It uses JAX to find the 1.3 million perfect numbers and saves them into a `.json` file.
2. **`packages/core` (The Brain)**: Contains the TypeScript code to load that JSON and run the math on your CPU.
3. **`packages/webgpu` (The Turbo)**: Contains specialized code (called "Shaders") to run that same math on your GPU.
4. **`packages/web` (The Face)**: The React app where you draw and see the result.

---

### External Resources for Coders
- [Machine Learning for Absolute Beginners](https://www.youtube.com/watch?v=KNAWp2S3w94)
- [How to explain a Neural Network to a 5-year-old](https://www.youtube.com/watch?v=bfmFfD2RIcg)
- [JAX for the Uninitiated](https://colinraffel.com/blog/you-don-t-know-jax.html)

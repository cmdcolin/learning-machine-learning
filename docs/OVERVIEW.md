# MNIST JAX — Project Overview

Train a digit classifier in Python, run inference in the browser.

## Start here

**[GUIDE.md](./GUIDE.md)** — plain-English explanation of everything: what a neural network is, how training works step by step, and specific things to change to learn more.

## Setup

```bash
make setup  # install Python + Node dependencies
make train  # train the JAX model and export weights
make web    # start the web app (http://localhost:5173)
```

## Project structure

```
train.py                  ← Python/JAX training (784→1024→512→10, 60K images)
packages/
  core/src/index.ts       ← CPU inference (plain TypeScript for-loops)
  webgpu/src/index.ts     ← GPU inference (WebGPU compute shaders)
  web/src/
    train.ts              ← Browser training algorithm (backprop in TypeScript)
    TrainView.tsx         ← "Train in Browser" UI
    App.tsx               ← Inference UI + tab navigation
data/                     ← MNIST binary files (downloaded by make setup)
```

## The two ways to experience this

**Inference tab** — draw a digit, the pre-trained JAX model predicts it. Toggle CPU vs WebGPU to see both inference paths.

**Train in Browser tab** — watch a smaller network (784→128→10) learn from scratch on 500 images. Every step of the algorithm runs in TypeScript with no dependencies.

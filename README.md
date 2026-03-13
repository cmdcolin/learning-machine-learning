# MNIST JAX Demo

**This project was generated with AI assistance (Claude). I did not write the code myself and do not take credit for it. I'm using it as a learning tool to understand how neural networks work.**

Live demo: https://cmdcolin.github.io/learning-machine-learning/

## What it does

Trains and runs an MNIST digit classifier. JAX is used for training in Python; inference runs in the browser in pure TypeScript (with optional WebGPU acceleration).

## Setup

```bash
make setup   # install deps, download data
make train   # train with JAX, export weights
make web     # run the web app locally
```

## Docs

- [Beginner's Guide](./docs/BEGINNERS_GUIDE.md)
- [Guide](./docs/GUIDE.md)
- [Overview](./docs/OVERVIEW.md)
- [Training with JAX](./docs/TRAINING.md)
- [Training Deeper](./docs/TRAINING_DEEPER.md)
- [Inference Architecture](./docs/INFERENCE.md)
- [ML Engineer Mindset](./docs/ML_ENGINEER_MINDSET.md)
- [Mindset](./docs/MINDSET.md)
- [Why Python](./docs/WHY_PYTHON.md)
- [Python Future](./docs/PYTHON_FUTURE.md)
- [Frameworks](./docs/FRAMEWORKS.md)
- [Notebooks](./docs/NOTEBOOKS.md)
- [Pitfalls](./docs/PITFALLS.md)
- [Roadmap](./docs/ROADMAP.md)
- [History](./docs/HISTORY.md)
- [Glossary](./docs/GLOSSARY.md)

## Structure

- `train.py` — JAX training script, exports weights to JSON
- `packages/web` — Vite + React web app (inference + in-browser training)
- `packages/cli` — Node.js CLI inference demo

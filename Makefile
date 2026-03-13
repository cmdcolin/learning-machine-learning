.PHONY: setup train web cli help

help:
	@echo "MNIST JAX Demo Commands:"
	@echo "  make setup   - Download data and install all dependencies (Python & Node)"
	@echo "  make train   - Train the model using JAX and export weights"
	@echo "  make web     - Start the Vite + React web application"
	@echo "  make cli     - Run the Node.js CLI inference demo"

setup:
	@echo "Setting up environment..."
	./download_data.sh
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt
	pnpm install
	pnpm --filter @mnist-jax/core build

train:
	@echo "Training model..."
	./venv/bin/python3 train.py

web:
	@echo "Starting web app..."
	pnpm --filter web dev

cli:
	@echo "Running CLI demo..."
	pnpm --filter cli start

# Inference Architectures: CPU vs WebGPU

This project demonstrates two different ways to run the same neural network in the browser.

## 1. Native TypeScript (CPU)

Found in `packages/core/src/index.ts`.

- **Approach**: Traditional JavaScript arrays and nested for-loops.
- **Complexity**: $O(N \cdot M)$ for each layer (where $N$ and $M$ are neurons per layer).
- **Pros**: Zero dependencies, runs anywhere, very simple to understand.
- **Cons**: Slower for large networks, as it uses a single CPU thread and does not leverage SIMD hardware as effectively as a compiled engine.

### Math Logic
The CPU engine manually iterates through rows and columns of the weight matrix to perform matrix-vector multiplication ($Wx + b$).

## 2. WebGPU-Accelerated (GPU)

Found in `packages/webgpu/src/index.ts`.

- **Approach**: Compute Shaders written in **WGSL** (WebGPU Shading Language).
- **Parallelism**: Every neuron's output in a given layer is calculated **simultaneously** on a separate GPU thread (workgroup).
- **Pros**: Massive performance gains for wide layers (like our 1024-neuron layer).
- **Cons**: High initial complexity (requires shader compilation, buffer management, and memory synchronization).

### How it works:
1. **Buffers**: Parameters (weights and biases) are loaded onto the GPU memory into specialized **storage buffers**.
2. **Pipelines**: A "Compute Pipeline" is created using the WGSL shader.
3. **Dispatch**: The CPU tells the GPU: "Start 64 workgroups and run the shader."
4. **Mapping**: After the GPU is done, the final 10-neuron output is "mapped" back to CPU-accessible memory to read the result.

## 3. The Performance Tradeoff: Latency vs. Throughput

For a small network like this, the difference between CPU and GPU is often negligible or surprising:

- **The Overhead (The "Wait"):** To run the GPU, we must send the weights to the Graphics Card's memory. This "handshake" takes time.
- **Why use the CPU?** For simple inference (one image at a time), the CPU is already incredibly fast. By the time the GPU has finished "waking up" and receiving the data, the CPU might have already finished the task.
- **When is WebGPU better?** If our model had **millions of neurons** or we were processing **real-time video frames**, the GPU's massive parallelism would dominate the CPU.

---

### External Resources
- [WebGPU Official Specification](https://www.w3.org/TR/webgpu/)
- [WebGPU Fundamentals - Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [WGSL Documentation (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API/WGSL)

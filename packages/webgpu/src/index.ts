/// <reference types="@webgpu/types" />
import type { LayerWeights } from '@mnist-jax/core';

// --- WGSL Shaders ---

const MATMUL_SHADER = `
struct Params {
  input_dim: u32,
  output_dim: u32,
  apply_relu: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>; // [output_dim * input_dim]
@group(0) @binding(2) var<storage, read> biases: array<f32>;  // [output_dim]
@group(0) @binding(3) var<storage, read> input_vec: array<f32>; // [input_dim]
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>; // [output_dim]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  if (row >= params.output_dim) { return; }

  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.input_dim; i++) {
    sum += weights[row * params.input_dim + i] * input_vec[i];
  }
  
  var result = sum + biases[row];
  if (params.apply_relu == 1u) {
    result = max(0.0, result);
  }
  
  output_vec[row] = result;
}
`;

// --- WebGPU Engine ---

export class WebGPUInference {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter found.");
    this.device = await adapter.requestDevice();
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: MATMUL_SHADER }),
        entryPoint: 'main',
      },
    });
  }

  private createBuffer(data: ArrayBufferView, usage: GPUBufferUsageFlags) {
    if (!this.device) throw new Error("Device not initialized");
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true,
    });
    
    if (data instanceof Float32Array) {
      new Float32Array(buffer.getMappedRange()).set(data);
    } else if (data instanceof Uint32Array) {
      new Uint32Array(buffer.getMappedRange()).set(data);
    }
    
    buffer.unmap();
    return buffer;
  }

  async runInference(image: number[], layers: LayerWeights[]) {
    if (!this.device || !this.pipeline) throw new Error("Inference engine not initialized");

    const device = this.device;
    let currentInputBuffer = this.createBuffer(new Float32Array(image), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    for (let i = 0; i < layers.length; i++) {
      const { weights, biases } = layers[i];
      const flatWeights = new Float32Array(weights.flat());
      const flatBiases = new Float32Array(biases);
      const outDim = biases.length;
      const inDim = weights[0].length;
      const applyRelu = i < layers.length - 1 ? 1 : 0;

      const weightBuffer = this.createBuffer(flatWeights, GPUBufferUsage.STORAGE);
      const biasBuffer = this.createBuffer(flatBiases, GPUBufferUsage.STORAGE);
      const outputBuffer = device.createBuffer({
        size: outDim * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const paramsBuffer = this.createBuffer(new Uint32Array([inDim, outDim, applyRelu]), GPUBufferUsage.UNIFORM);

      const bindGroup = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: weightBuffer } },
          { binding: 2, resource: { buffer: biasBuffer } },
          { binding: 3, resource: { buffer: currentInputBuffer } },
          { binding: 4, resource: { buffer: outputBuffer } },
        ],
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(this.pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(outDim / 64));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);

      // The current output becomes the next input
      currentInputBuffer = outputBuffer;
    }

    // Final result (probabilities/logits)
    const readBuffer = device.createBuffer({
      size: currentInputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(currentInputBuffer, 0, readBuffer, 0, currentInputBuffer.size);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const finalArray = Array.from(result);
    readBuffer.unmap();

    // Compute Softmax and Argmax (on CPU for simplicity)
    const maxVal = Math.max(...finalArray);
    const exps = finalArray.map(val => Math.exp(val - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(val => val / sum);
    
    let maxIdx = -1;
    let maxProb = -1;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    return { prediction: maxIdx, probabilities: probs };
  }
}

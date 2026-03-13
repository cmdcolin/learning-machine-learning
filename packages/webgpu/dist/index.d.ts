import type { LayerWeights } from '@mnist-jax/core';
export declare class WebGPUInference {
    private device;
    private pipeline;
    init(): Promise<void>;
    private createBuffer;
    runInference(image: number[], layers: LayerWeights[]): Promise<{
        prediction: number;
        probabilities: number[];
    }>;
}

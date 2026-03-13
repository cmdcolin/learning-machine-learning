export type Matrix = number[][];
export type Vector = number[];
export interface LayerWeights {
    weights: Matrix;
    biases: Vector;
}
export interface TestExample {
    image: Vector;
    label: number;
}
export declare function runInference(image: Vector, layers: LayerWeights[]): {
    prediction: number;
    probabilities: number[];
};

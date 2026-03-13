// --- Linear Algebra Helpers ---
function dot(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}
function matMulVec(m, v) {
    const rows = m.length;
    const result = new Array(rows);
    for (let i = 0; i < rows; i++) {
        result[i] = dot(m[i], v);
    }
    return result;
}
function addVec(v1, v2) {
    const result = new Array(v1.length);
    for (let i = 0; i < v1.length; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}
function relu(v) {
    return v.map(val => Math.max(0, val));
}
function argmax(v) {
    let maxVal = -Infinity;
    let maxIdx = -1;
    for (let i = 0; i < v.length; i++) {
        if (v[i] > maxVal) {
            maxVal = v[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}
function softmax(v) {
    const maxVal = Math.max(...v);
    const exps = v.map(val => Math.exp(val - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(val => val / sum);
}
// --- Inference ---
export function runInference(image, layers) {
    let activations = image;
    for (let i = 0; i < layers.length; i++) {
        const { weights, biases } = layers[i];
        let output = matMulVec(weights, activations);
        output = addVec(output, biases);
        if (i < layers.length - 1) {
            activations = relu(output);
        }
        else {
            activations = output;
        }
    }
    return {
        prediction: argmax(activations),
        probabilities: softmax(activations)
    };
}

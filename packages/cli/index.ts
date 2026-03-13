import * as fs from 'fs';
import * as path from 'path';
import { runInference } from '@mnist-jax/core';
import type { LayerWeights, TestExample } from '@mnist-jax/core';

function main() {
    const weightsPath = path.join(__dirname, 'weights.json');
    const testDataPath = path.join(__dirname, 'test_images.json');

    if (!fs.existsSync(weightsPath) || !fs.existsSync(testDataPath)) {
        console.error("Error: weights.json or test_images.json not found. Run train.py first.");
        process.exit(1);
    }

    console.log("Loading weights and test data...");
    const layers: LayerWeights[] = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    const testExamples: TestExample[] = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));

    console.log(`Loaded ${layers.length} layers.`);
    console.log(`Running inference on ${testExamples.length} examples...\n`);

    let correct = 0;
    for (let i = 0; i < testExamples.length; i++) {
        const { image, label } = testExamples[i];
        const { prediction, probabilities } = runInference(image, layers);
        
        const isCorrect = prediction === label;
        if (isCorrect) correct++;

        console.log(`Example ${i + 1}:`);
        console.log(`  True Label: ${label}`);
        console.log(`  Prediction: ${prediction} (prob: ${probabilities[prediction].toFixed(4)})`);
        console.log(`  Result:     ${isCorrect ? "PASS" : "FAIL"}`);
        console.log("-------------------------");
    }

    console.log(`Accuracy on demo set: ${correct}/${testExamples.length} (${(correct / testExamples.length) * 100}%)`);
}

main();

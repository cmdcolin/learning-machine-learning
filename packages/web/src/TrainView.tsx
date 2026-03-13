/**
 * TrainView — trains an MNIST classifier directly in your browser.
 *
 * No Python, no server, no dependencies beyond what's already in this repo.
 * Everything runs in pure TypeScript so you can read exactly what each step does.
 *
 * Architecture: 784 → 128 (ReLU) → 10 (Softmax)
 * Optimizer:    Adam (lr = 0.001)
 * Data:         500 balanced training examples (50 per digit class)
 */

import { useState, useRef, useCallback } from 'react';
import LossChart from './LossChart';
import {
  initWeights,
  initAdamState,
  trainEpoch,
  inferBrowser,
  type MLPWeights,
  type AdamState,
  type TrainSample,
  type EpochResult,
} from './train';
import type { TestExample } from '@mnist-jax/core';

const TOTAL_EPOCHS = 20;
const BATCH_SIZE = 32;
const LEARNING_RATE = 0.001;

// Plain-English explanations of what's happening at each stage
const PHASE_EXPLANATIONS: Record<string, { title: string; detail: string }> = {
  idle: {
    title: 'Ready to train',
    detail:
      'Click "Start Training" to initialize 102,410 random weights and begin gradient descent on 500 handwritten digit images.',
  },
  init: {
    title: 'Initializing weights',
    detail:
      'Creating weight matrices W1 [128×784] and W2 [10×128] with Xavier initialization. Weights start small and random — near zero keeps early activations in a useful range.',
  },
  forward: {
    title: 'Forward pass',
    detail:
      'Running each image through the network: x → W1·x+b1 → ReLU → W2·a1+b2 → Softmax. The final 10 numbers are probabilities for each digit.',
  },
  loss: {
    title: 'Computing loss',
    detail:
      'Loss = −log(probability of correct class). If the network assigns 1% chance to the right digit, loss ≈ 4.6. Perfect confidence → loss ≈ 0. We average over a batch of 32.',
  },
  backward: {
    title: 'Backpropagation',
    detail:
      'Working backward through the chain rule to find ∂Loss/∂W for every weight. The gradient tells each weight: "move this much in this direction to reduce the loss."',
  },
  adam: {
    title: 'Adam optimizer update',
    detail:
      'Adam keeps a smoothed average of gradients (momentum) and their squares (adaptive rate). This lets it move confidently in consistent directions and slow down when gradients oscillate.',
  },
  done: {
    title: 'Training complete!',
    detail:
      'All epochs finished. The network has updated its weights ~3,120 times (500 samples ÷ 32 batch × 20 epochs × Adam steps). Switch to the Inference tab to test with the pre-trained big model, or draw digits above to test this one.',
  },
};

export default function TrainView({ testImages }: { testImages: TestExample[] }) {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentBatch, setCurrentBatch] = useState<{ idx: number; total: number } | null>(null);
  const [epochResults, setEpochResults] = useState<EpochResult[]>([]);
  const [currentBatchLoss, setCurrentBatchLoss] = useState<number | null>(null);
  const [phase, setPhase] = useState<string>('idle');
  const [testPredictions, setTestPredictions] = useState<{ label: number; pred: number }[]>([]);

  const cancelRef = useRef({ cancelled: false });
  const weightsRef = useRef<MLPWeights | null>(null);
  const adamRef = useRef<AdamState | null>(null);
  const trainDataRef = useRef<TrainSample[] | null>(null);

  const runTestPredictions = useCallback((W: MLPWeights) => {
    const results = testImages.map((ex) => ({
      label: ex.label,
      pred: inferBrowser(ex.image.map((v) => Math.round(v * 255)), W).prediction,
    }));
    setTestPredictions(results);
  }, []);

  const startTraining = useCallback(async () => {
    cancelRef.current = { cancelled: false };
    setIsTraining(true);
    setEpochResults([]);
    setCurrentBatchLoss(null);
    setCurrentEpoch(0);
    setCurrentBatch(null);
    setTestPredictions([]);

    setPhase('init');
    await new Promise<void>((r) => setTimeout(r, 80)); // let UI render "Initializing" state

    if (!trainDataRef.current) {
      const base = import.meta.env.BASE_URL;
      const res = await fetch(`${base}train_data.json`);
      trainDataRef.current = await res.json() as TrainSample[];
    }
    const trainData = trainDataRef.current;

    const W = initWeights();
    const adamState = initAdamState();
    weightsRef.current = W;
    adamRef.current = adamState;

    for (let epoch = 1; epoch <= TOTAL_EPOCHS; epoch++) {
      if (cancelRef.current.cancelled) break;
      setCurrentEpoch(epoch);

      let batchPhaseToggle = 0;
      const phases = ['forward', 'loss', 'backward', 'adam'];

      const { avgLoss, accuracy } = await trainEpoch(
        trainData,
        W,
        adamState,
        LEARNING_RATE,
        BATCH_SIZE,
        (batchLoss, batchIdx, totalBatches) => {
          setCurrentBatchLoss(batchLoss);
          setCurrentBatch({ idx: batchIdx, total: totalBatches });
          // Rotate through phase labels so the user sees each conceptual step
          setPhase(phases[batchPhaseToggle % phases.length]);
          batchPhaseToggle++;
        },
        cancelRef.current,
      );

      if (cancelRef.current.cancelled) break;

      const result: EpochResult = { epoch, loss: avgLoss, accuracy };
      setEpochResults((prev) => [...prev, result]);
      setCurrentBatchLoss(null);
      setCurrentBatch(null);
    }

    if (!cancelRef.current.cancelled) {
      setPhase('done');
      runTestPredictions(W);
    } else {
      setPhase('idle');
    }
    setIsTraining(false);
  }, [runTestPredictions]);

  const stopTraining = useCallback(() => {
    cancelRef.current.cancelled = true;
  }, []);

  const resetTraining = useCallback(() => {
    cancelRef.current.cancelled = true;
    setIsTraining(false);
    setCurrentEpoch(0);
    setCurrentBatch(null);
    setEpochResults([]);
    setCurrentBatchLoss(null);
    setPhase('idle');
    setTestPredictions([]);
    weightsRef.current = null;
    adamRef.current = null;
  }, []);

  const latestResult = epochResults[epochResults.length - 1];
  const phaseInfo = PHASE_EXPLANATIONS[phase] ?? PHASE_EXPLANATIONS.idle;

  const progressPct =
    currentEpoch > 0
      ? ((currentEpoch - 1 + (currentBatch ? currentBatch.idx / currentBatch.total : 0)) /
          TOTAL_EPOCHS) *
        100
      : 0;

  return (
    <div className="train-view">
      {/* Header + controls */}
      <div className="train-header">
        <div>
          <h2 style={{ margin: 0 }}>Train in Browser</h2>
          <p className="train-subtitle">
            Pure TypeScript · 500 samples · 784→128→10 MLP · Adam optimizer
          </p>
        </div>
        <div className="train-controls">
          {!isTraining ? (
            <button className="btn-primary" onClick={startTraining}>
              {epochResults.length > 0 ? 'Train Again' : 'Start Training'}
            </button>
          ) : (
            <button onClick={stopTraining}>Stop</button>
          )}
          {(epochResults.length > 0 || isTraining) && (
            <button onClick={resetTraining}>Reset</button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
      </div>

      <div className="train-body">
        {/* Left column: live metrics */}
        <div className="train-metrics">
          <div className="metric-grid">
            <div className="metric-card">
              <div className="metric-label">Epoch</div>
              <div className="metric-value">
                {currentEpoch > 0 ? `${currentEpoch} / ${TOTAL_EPOCHS}` : '—'}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Loss</div>
              <div className="metric-value">
                {currentBatchLoss !== null
                  ? currentBatchLoss.toFixed(4)
                  : latestResult
                    ? latestResult.loss.toFixed(4)
                    : '—'}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">
                {latestResult ? `${(latestResult.accuracy * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Batch</div>
              <div className="metric-value">
                {currentBatch ? `${currentBatch.idx}/${currentBatch.total}` : '—'}
              </div>
            </div>
          </div>

          {/* Phase explanation */}
          <div className="phase-card">
            <div className="phase-title">
              {isTraining && <span className="pulse-dot" />}
              {phaseInfo.title}
            </div>
            <div className="phase-detail">{phaseInfo.detail}</div>
          </div>

          {/* Epoch history table */}
          {epochResults.length > 0 && (
            <div className="epoch-table-wrap">
              <table className="epoch-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Loss</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {epochResults.map((r) => (
                    <tr key={r.epoch}>
                      <td>{r.epoch}</td>
                      <td>{r.loss.toFixed(4)}</td>
                      <td>{(r.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right column: chart + test results */}
        <div className="train-chart-col">
          <LossChart
            data={epochResults}
            currentBatchLoss={currentBatchLoss}
            totalEpochs={TOTAL_EPOCHS}
          />

          {/* Show test predictions once training finishes */}
          {testPredictions.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: '0.9rem', color: '#aaa', marginBottom: 8 }}>
                Test examples with trained weights
              </h3>
              <div className="test-grid">
                {testPredictions.map((r, i) => (
                  <div key={i} className="test-item">
                    <div className="test-label">True: {r.label}</div>
                    <div className={`test-pred ${r.label === r.pred ? 'correct' : 'wrong'}`}>
                      Pred: {r.pred}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Architecture explainer */}
          <div className="arch-box">
            <div className="arch-title">Network architecture</div>
            <div className="arch-layers">
              <div className="arch-layer">
                <div className="arch-layer-name">Input</div>
                <div className="arch-layer-size">784</div>
                <div className="arch-layer-desc">28×28 pixels</div>
              </div>
              <div className="arch-arrow">→</div>
              <div className="arch-layer">
                <div className="arch-layer-name">Hidden</div>
                <div className="arch-layer-size">128</div>
                <div className="arch-layer-desc">ReLU</div>
              </div>
              <div className="arch-arrow">→</div>
              <div className="arch-layer">
                <div className="arch-layer-name">Output</div>
                <div className="arch-layer-size">10</div>
                <div className="arch-layer-desc">Softmax</div>
              </div>
            </div>
            <div className="arch-params">
              102,410 total parameters (W1: 100,352 · b1: 128 · W2: 1,280 · b2: 10)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect, useRef } from 'react';
import './App.css';
import { runInference as runCpuInference } from '@mnist-jax/core';
import { WebGPUInference } from '@mnist-jax/webgpu';
import type { LayerWeights, TestExample } from '@mnist-jax/core';
import MNISTCanvas from './MNISTCanvas';
import TrainView from './TrainView';

type Tab = 'inference' | 'train' | 'about';

function fetchWithProgress(url: string, onProgress: (pct: number) => void) {
  return new Promise<unknown>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url);
    xhr.responseType = 'json';
    xhr.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(e.loaded / e.total);
      }
    };
    xhr.onload = () => resolve(xhr.response);
    xhr.onerror = reject;
    xhr.send();
  });
}

function LoadingScreen({ progress }: { progress: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: 16 }}>
      <div style={{ fontSize: '1rem', color: '#aaa' }}>Loading model weights…</div>
      <div style={{ width: 280, height: 8, background: '#333', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${progress * 100}%`, background: '#7c6af7', borderRadius: 4, transition: 'width 0.1s' }} />
      </div>
      <div style={{ fontSize: '0.8rem', color: '#666' }}>{Math.round(progress * 100)}%</div>
    </div>
  );
}

function InferenceView({ weights, testImages }: { weights: LayerWeights[]; testImages: TestExample[] }) {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probs, setProbs] = useState<number[]>([]);
  const [testResults, setTestResults] = useState<{label: number, pred: number}[]>([]);
  const [debugData, setDebugData] = useState<number[]>([]);
  const [useGpu, setUseGpu] = useState(false);
  const [gpuSupported, setGpuSupported] = useState(false);
  const gpuEngine = useRef<WebGPUInference | null>(null);
  const debugCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const initGpu = async () => {
      try {
        const engine = new WebGPUInference();
        await engine.init();
        gpuEngine.current = engine;
        setGpuSupported(true);
        setUseGpu(true);
      } catch (e) {
        console.warn("WebGPU not available:", e);
        setGpuSupported(false);
      }
    };
    initGpu();

    const results = testImages.map(ex => ({
      label: ex.label,
      pred: runCpuInference(ex.image, weights).prediction
    }));
    setTestResults(results);
  }, [weights, testImages]);

  useEffect(() => {
    if (debugCanvasRef.current && debugData.length === 784) {
      const ctx = debugCanvasRef.current.getContext('2d');
      if (ctx) {
        const imageData = ctx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
          const val = debugData[i] * 255;
          imageData.data[i * 4] = val;
          imageData.data[i * 4 + 1] = val;
          imageData.data[i * 4 + 2] = val;
          imageData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      }
    }
  }, [debugData]);

  const handleDraw = async (data: number[]) => {
    setDebugData(data);

    if (useGpu && gpuEngine.current) {
      const { prediction: pred, probabilities } = await gpuEngine.current.runInference(data, weights);
      setPrediction(pred);
      setProbs(probabilities);
    } else {
      const { prediction: pred, probabilities } = runCpuInference(data, weights);
      setPrediction(pred);
      setProbs(probabilities);
    }
  };

  const handleClear = () => {
    setPrediction(null);
    setProbs([]);
    setDebugData([]);
  };

  return (
    <>
      {/* CPU/GPU toggle lives here, only relevant for inference */}
      <div className="mode-selector" style={{ alignSelf: 'flex-end', marginBottom: 16 }}>
        <span className={!useGpu ? 'active' : ''}>CPU</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={useGpu}
            disabled={!gpuSupported}
            onChange={(e) => setUseGpu(e.target.checked)}
          />
          <span className="slider round"></span>
        </label>
        <span className={useGpu ? 'active' : ''}>
          WebGPU {!gpuSupported && <span className="not-supported">(Not Supported)</span>}
        </span>
      </div>

      <div className="container">
        <section className="demo-section">
          <h2>Draw a Digit</h2>
          <MNISTCanvas onDraw={handleDraw} onClear={handleClear} />

          <div className="debug-container">
            <div>
              <p>Normalized (28x28)</p>
              <canvas ref={debugCanvasRef} width={28} height={28} className="debug-canvas" />
            </div>
            {prediction !== null && (
              <div className="prediction-display">
                Prediction: <span className="winner">{prediction}</span>
              </div>
            )}
          </div>

          {probs.length > 0 && (
            <div className="prob-chart">
              {probs.map((p, i) => (
                <div key={i} className="prob-row">
                  <span className="digit-label">{i}</span>
                  <div className="prob-bar-container">
                    <div
                      className={`prob-bar ${i === prediction ? 'winner-bar' : ''}`}
                      style={{ width: `${p * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="test-section">
          <h2>Test Examples</h2>
          <p style={{ fontSize: '0.8rem', color: '#888', marginTop: 4 }}>
            Using the pre-trained JAX model (784→1024→512→10, 15 epochs on 60K images)
          </p>
          <div className="test-grid">
            {testResults.map((res, i) => (
              <div key={i} className="test-item">
                <div className="test-label">True: {res.label}</div>
                <div className={`test-pred ${res.label === res.pred ? 'correct' : 'wrong'}`}>
                  Pred: {res.pred}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}

function AboutView() {
  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: '0 24px', lineHeight: 1.7, color: '#ccc' }}>
      <h2>About this project</h2>
      <p>
        This project was generated with AI assistance (Claude). I did not write the code myself
        and do not take credit for it.
      </p>
      <p>
        I'm using it as a learning tool to understand how neural networks and machine learning work —
        the JAX training pipeline, browser-based inference, WebGPU compute shaders, and how
        MNIST classifiers are built end-to-end.
      </p>
      <p style={{ color: '#666', fontSize: '0.85rem' }}>
        Source: <a href="https://github.com/cmdcolin/learning-machine-learning" style={{ color: '#7c6af7' }}>github.com/cmdcolin/learning-machine-learning</a>
      </p>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState<Tab>('inference');
  const [weights, setWeights] = useState<LayerWeights[] | null>(null);
  const [testImages, setTestImages] = useState<TestExample[] | null>(null);
  const [loadProgress, setLoadProgress] = useState(0);

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    Promise.all([
      fetchWithProgress(`${base}weights.json`, setLoadProgress),
      fetch(`${base}test_images.json`).then(r => r.json()),
    ]).then(([w, t]) => {
      setWeights(w as LayerWeights[]);
      setTestImages(t as TestExample[]);
    });
  }, []);

  if (!weights || !testImages) {
    return (
      <div className="App">
        <header className="app-header">
          <h1>MNIST JAX Demo</h1>
        </header>
        <LoadingScreen progress={loadProgress} />
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>MNIST JAX Demo</h1>
        <nav className="tab-nav">
          <button
            className={`tab-btn ${tab === 'inference' ? 'tab-active' : ''}`}
            onClick={() => setTab('inference')}
          >
            Inference
          </button>
          <button
            className={`tab-btn ${tab === 'train' ? 'tab-active' : ''}`}
            onClick={() => setTab('train')}
          >
            Train in Browser
          </button>
          <button
            className={`tab-btn ${tab === 'about' ? 'tab-active' : ''}`}
            onClick={() => setTab('about')}
          >
            About
          </button>
        </nav>
      </header>

      {tab === 'inference' && <InferenceView weights={weights} testImages={testImages} />}
      {tab === 'train' && <TrainView testImages={testImages} />}
      {tab === 'about' && <AboutView />}
    </div>
  );
}

export default App;

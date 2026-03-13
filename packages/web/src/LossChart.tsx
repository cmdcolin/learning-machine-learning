import { useRef, useEffect } from 'react';

interface DataPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface LossChartProps {
  data: DataPoint[];
  currentBatchLoss: number | null; // live loss within current epoch
  totalEpochs: number;
}

const W = 480;
const H = 200;
const PAD = { top: 16, right: 16, bottom: 32, left: 48 };

export default function LossChart({ data, currentBatchLoss, totalEpochs }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    // Gather all loss values to determine Y range
    const allLosses = data.map((d) => d.loss);
    if (currentBatchLoss !== null) allLosses.push(currentBatchLoss);
    const maxLoss = allLosses.length ? Math.max(...allLosses, 0.1) : 2.5;
    const minLoss = 0;

    // Map epoch/loss to canvas coordinates
    const xOf = (epoch: number) =>
      PAD.left + ((epoch - 1) / Math.max(totalEpochs - 1, 1)) * plotW;
    const yOf = (loss: number) =>
      PAD.top + (1 - (loss - minLoss) / (maxLoss - minLoss)) * plotH;

    // Gridlines
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
    }

    // Y axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const loss = maxLoss - (maxLoss / 4) * i;
      const y = PAD.top + (plotH / 4) * i;
      ctx.fillText(loss.toFixed(2), PAD.left - 6, y + 4);
    }

    // X axis labels
    ctx.textAlign = 'center';
    const epochStep = Math.max(1, Math.floor(totalEpochs / 5));
    for (let e = 1; e <= totalEpochs; e += epochStep) {
      const x = xOf(e);
      ctx.fillText(String(e), x, H - PAD.bottom + 16);
    }

    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', PAD.left + plotW / 2, H - 2);

    // Accuracy line (green, dashed)
    if (data.length > 1) {
      ctx.strokeStyle = '#4CAF50';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      data.forEach((d, i) => {
        // Scale accuracy to the same y-axis range for comparison
        const scaledAcc = d.accuracy * maxLoss;
        if (i === 0) ctx.moveTo(xOf(d.epoch), yOf(scaledAcc));
        else ctx.lineTo(xOf(d.epoch), yOf(scaledAcc));
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Loss line (purple, solid)
    if (data.length > 1) {
      ctx.strokeStyle = '#646cff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((d, i) => {
        if (i === 0) ctx.moveTo(xOf(d.epoch), yOf(d.loss));
        else ctx.lineTo(xOf(d.epoch), yOf(d.loss));
      });
      ctx.stroke();
    }

    // Dots at each completed epoch
    for (const d of data) {
      ctx.fillStyle = '#646cff';
      ctx.beginPath();
      ctx.arc(xOf(d.epoch), yOf(d.loss), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Live batch loss indicator (if mid-epoch)
    if (currentBatchLoss !== null) {
      const currentEpoch = data.length + 1;
      const x = xOf(currentEpoch);
      const y = yOf(currentBatchLoss);
      ctx.strokeStyle = '#ff9800';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      if (data.length > 0) {
        const prev = data[data.length - 1];
        ctx.beginPath();
        ctx.moveTo(xOf(prev.epoch), yOf(prev.loss));
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = '#ff9800';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Legend
    ctx.font = '10px monospace';
    ctx.fillStyle = '#646cff';
    ctx.fillRect(PAD.left + 4, PAD.top + 4, 14, 2);
    ctx.fillStyle = '#aaa';
    ctx.textAlign = 'left';
    ctx.fillText('loss', PAD.left + 22, PAD.top + 9);

    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(PAD.left + 60, PAD.top + 5);
    ctx.lineTo(PAD.left + 74, PAD.top + 5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#aaa';
    ctx.fillText('acc (scaled)', PAD.left + 78, PAD.top + 9);
  }, [data, currentBatchLoss, totalEpochs]);

  return (
    <canvas
      ref={canvasRef}
      style={{ borderRadius: 8, border: '1px solid #2a2a4a', display: 'block' }}
    />
  );
}

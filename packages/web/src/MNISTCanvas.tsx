import React, { useRef, useEffect, useState } from 'react';

interface MNISTCanvasProps {
  onDraw: (data: number[]) => void;
  onClear: () => void;
}

const MNISTCanvas: React.FC<MNISTCanvasProps> = ({ onDraw, onClear }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 24; // Thicker stroke for better digit visibility
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    if ('touches' in e) {
        return {
            x: (e as React.TouchEvent).touches[0].clientX - rect.left,
            y: (e as React.TouchEvent).touches[0].clientY - rect.top
        };
    }
    return {
      x: (e as React.MouseEvent).clientX - rect.left,
      y: (e as React.MouseEvent).clientY - rect.top
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    const { x, y } = getPos(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const { x, y } = getPos(e);
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.lineTo(x, y);
    ctx.stroke();
    extractData();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clear = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    onClear();
  };

  const extractData = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // 1. Find bounding box of the drawing to center it
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const i = (y * canvas.width + x) * 4;
        if (data[i] > 20) { // If pixel is "bright" enough
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (!found) {
      onDraw(new Array(784).fill(0));
      return;
    }

    // 2. Create a temporary canvas to center and scale the digit
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, 28, 28);

    const contentW = maxX - minX;
    const contentH = maxY - minY;
    const size = Math.max(contentW, contentH);
    
    // Scale to fit in a 20x20 area (standard MNIST practice)
    const scale = 20 / size;
    const offsetX = (28 - contentW * scale) / 2;
    const offsetY = (28 - contentH * scale) / 2;

    tempCtx.drawImage(
      canvas,
      minX, minY, contentW, contentH,
      offsetX, offsetY, contentW * scale, contentH * scale
    );

    const finalImageData = tempCtx.getImageData(0, 0, 28, 28).data;
    const grayscale = [];
    for (let i = 0; i < finalImageData.length; i += 4) {
      grayscale.push(finalImageData[i] / 255.0);
    }
    onDraw(grayscale);
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        style={{ border: '2px solid #555', cursor: 'crosshair', touchAction: 'none' }}
      />
      <br />
      <button onClick={clear} style={{ marginTop: '10px' }}>Clear</button>
    </div>
  );
};

export default MNISTCanvas;

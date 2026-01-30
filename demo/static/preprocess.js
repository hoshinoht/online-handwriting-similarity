/**
 * Runtime Preprocessing for Handwriting Model
 * Implements: Simplification -> Resampling -> Normalization -> Feature Extraction
 */

// 1. Douglas-Peucker Simplification
function douglasPeucker(points, epsilon) {
  if (points.length < 3) return points;

  let dmax = 0;
  let index = 0;
  const end = points.length - 1;
  const p1 = points[0];
  const p2 = points[end];

  for (let i = 1; i < end; i++) {
    let d = 0;
    const p = points[i];

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;

    if (dx === 0 && dy === 0) {
      d = Math.hypot(p.x - p1.x, p.y - p1.y);
    } else {
      const num = Math.abs(dy * p.x - dx * p.y + p2.x * p1.y - p2.y * p1.x);
      const den = Math.hypot(dy, dx);
      d = num / den;
    }

    if (d > dmax) {
      index = i;
      dmax = d;
    }
  }

  if (dmax > epsilon) {
    const rec1 = douglasPeucker(points.slice(0, index + 1), epsilon);
    const rec2 = douglasPeucker(points.slice(index), epsilon);
    return [...rec1.slice(0, -1), ...rec2];
  } else {
    return [p1, p2];
  }
}

// 2. Resampling (Linear Interpolation)
function resample(points, targetLength = 128) {
  if (points.length <= 1) {
    return Array(targetLength).fill(points[0]);
  }

  let totalDist = 0;
  const cumDists = [0];
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    totalDist += d;
    cumDists.push(totalDist);
  }

  if (totalDist === 0) return Array(targetLength).fill(points[0]);

  const resampled = [];
  const step = totalDist / (targetLength - 1);

  for (let i = 0; i < targetLength; i++) {
    const targetDist = i * step;

    let idx = 0;
    while (idx < cumDists.length - 1 && cumDists[idx + 1] < targetDist) {
      idx++;
    }

    if (idx >= cumDists.length - 1) {
      resampled.push(points[points.length - 1]);
      continue;
    }

    const t = (targetDist - cumDists[idx]) / (cumDists[idx + 1] - cumDists[idx]);
    const pStart = points[idx];
    const pEnd = points[idx + 1];

    resampled.push({
      x: pStart.x + (pEnd.x - pStart.x) * t,
      y: pStart.y + (pEnd.y - pStart.y) * t
    });
  }
  return resampled;
}

// 3. Normalization
function normalize(points) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

  points.forEach(p => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });

  const width = maxX - minX;
  const height = maxY - minY;

  // Scale by the largest dimension to preserve aspect ratio
  const scale = Math.max(width, height) || 1;

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  return points.map(p => ({
    x: (p.x - centerX) / scale + 0.5,
    y: (p.y - centerY) / scale + 0.5
  }));
}

// 4. Feature Extraction
function extractFeatures(points) {
  const L = points.length;
  const features = [];

  const deltas = [];
  const thetas = [];

  // First pass: deltas and angles
  for (let i = 0; i < L; i++) {
    let dx, dy;
    if (i < L - 1) {
      dx = points[i + 1].x - points[i].x;
      dy = points[i + 1].y - points[i].y;
    } else {
      dx = (i > 0) ? deltas[i - 1].dx : 0;
      dy = (i > 0) ? deltas[i - 1].dy : 0;
    }
    deltas.push({ dx, dy });

    const norm = Math.hypot(dx, dy) + 1e-8;
    const sin = dy / norm;
    const cos = dx / norm;
    const theta = Math.atan2(dy, dx);

    thetas.push(theta);

    features[i] = { x: points[i].x, y: points[i].y, dx, dy, sin, cos };
  }

  // Second pass: curvature (kappa)
  const kappas = new Array(L).fill(0);
  for (let i = 1; i < L; i++) {
    let dTheta = thetas[i] - thetas[i - 1];
    while (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    while (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    kappas[i] = dTheta;
  }

  // Flatten to array of arrays
  return features.map((f, i) => [
    f.x, f.y, f.dx, f.dy, f.sin, f.cos, kappas[i]
  ]);
}

// Main Pipeline
function processUserStroke(rawPoints) {
  if (rawPoints.length < 2) return null;

  // Convert to {x,y} if needed
  const points = rawPoints.map(p => ({ x: p.x, y: p.y }));

  const simplified = douglasPeucker(points, 2.0);
  const resampled = resample(simplified, 128);
  const normalized = normalize(resampled);
  const featureVector = extractFeatures(normalized);

  return featureVector;
}

import type { UnfoldManifest } from "./io";

export interface UnfoldContext {
  manifest: UnfoldManifest;
  baseXYZ: Float32Array;
  segLen: Float32Array;
  theta: Float32Array;
  ap: Float32Array;
}

export interface ApplyParams {
  progress: number;
  phase1Frac: number;
  flipXZ: boolean;
  outXYZ: Float32Array;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function wrapAngle(theta: number): number {
  const twoPi = 2.0 * Math.PI;
  const shifted = theta + Math.PI;
  return ((shifted % twoPi) + twoPi) % twoPi - Math.PI;
}

function mapProgress(progress: number, phase1Frac: number): { alphaML: number; betaAP: number } {
  const p = clamp(progress, 0, 1);
  const phase = clamp(phase1Frac, 0.05, 0.95);
  if (p < phase) {
    return { alphaML: p / phase, betaAP: 0 };
  }
  return { alphaML: 1, betaAP: (p - phase) / (1 - phase) };
}

export function createUnfoldContext(
  manifest: UnfoldManifest,
  baseXYZ: Float32Array,
  segLen: Float32Array,
  theta: Float32Array,
  ap: Float32Array
): UnfoldContext {
  const nRows = manifest.n_rows;
  const nCols = manifest.n_cols;
  const nVerts = nRows * nCols;
  if (baseXYZ.length !== nVerts * 3) {
    throw new Error(`positions length mismatch: ${baseXYZ.length} vs ${nVerts * 3}`);
  }
  if (segLen.length !== nRows * (nCols - 1)) {
    throw new Error(`segLen length mismatch: ${segLen.length} vs ${nRows * (nCols - 1)}`);
  }
  if (theta.length !== nRows * (nCols - 1)) {
    throw new Error(`theta length mismatch: ${theta.length} vs ${nRows * (nCols - 1)}`);
  }
  if (ap.length !== nRows) {
    throw new Error(`ap length mismatch: ${ap.length} vs ${nRows}`);
  }
  return { manifest, baseXYZ, segLen, theta, ap };
}

export function applyUnfold(context: UnfoldContext, params: ApplyParams): void {
  const { manifest, baseXYZ, segLen, theta, ap } = context;
  const { outXYZ } = params;
  if (outXYZ.length !== baseXYZ.length) {
    throw new Error(`outXYZ length mismatch: ${outXYZ.length} vs ${baseXYZ.length}`);
  }

  const { alphaML, betaAP } = mapProgress(params.progress, params.phase1Frac);
  const nRows = manifest.n_rows;
  const nCols = manifest.n_cols;
  const anchorCol = manifest.anchor_col;
  const jPlane = manifest.j_plane;

  for (let row = 0; row < nRows; row += 1) {
    const rowBase = row * nCols * 3;
    const segBase = row * (nCols - 1);

    const anchorIdx = rowBase + anchorCol * 3;
    const anchorK = baseXYZ[anchorIdx + 0];
    const anchorJ = baseXYZ[anchorIdx + 1];

    // Start with source row.
    for (let col = 0; col < nCols; col += 1) {
      const idx = rowBase + col * 3;
      outXYZ[idx + 0] = baseXYZ[idx + 0];
      outXYZ[idx + 1] = baseXYZ[idx + 1];
      outXYZ[idx + 2] = baseXYZ[idx + 2];
    }

    outXYZ[anchorIdx + 0] = anchorK;
    outXYZ[anchorIdx + 1] = anchorJ;

    // Forward from anchor -> right.
    if (anchorCol < nCols - 1) {
      let prevK = anchorK;
      let prevJ = anchorJ;
      let prevTheta = 0;
      let prevOrig = 0;
      for (let seg = anchorCol; seg < nCols - 1; seg += 1) {
        const orig = theta[segBase + seg];
        let th: number;
        if (seg === anchorCol) {
          th = (1 - alphaML) * wrapAngle(orig);
        } else {
          const dth = wrapAngle(orig - prevOrig);
          th = prevTheta + (1 - alphaML) * dth;
        }
        const len = segLen[segBase + seg];
        prevK += len * Math.cos(th);
        prevJ += len * Math.sin(th);
        const toCol = seg + 1;
        const idx = rowBase + toCol * 3;
        outXYZ[idx + 0] = prevK;
        outXYZ[idx + 1] = prevJ;
        prevTheta = th;
        prevOrig = orig;
      }
    }

    // Backward from anchor -> left.
    if (anchorCol > 0) {
      let prevK = anchorK;
      let prevJ = anchorJ;
      let prevTheta = 0;
      let prevOrig = 0;
      for (let rev = 0; rev < anchorCol; rev += 1) {
        const seg = anchorCol - 1 - rev;
        const orig = wrapAngle(theta[segBase + seg] + Math.PI);
        let th: number;
        if (rev === 0) {
          th = (1 - alphaML) * wrapAngle(orig - Math.PI) + Math.PI;
        } else {
          const dth = wrapAngle(orig - prevOrig);
          th = prevTheta + (1 - alphaML) * dth;
        }
        const len = segLen[segBase + seg];
        prevK += len * Math.cos(th);
        prevJ += len * Math.sin(th);
        const toCol = seg;
        const idx = rowBase + toCol * 3;
        outXYZ[idx + 0] = prevK;
        outXYZ[idx + 1] = prevJ;
        prevTheta = th;
        prevOrig = orig;
      }
    }

    if (betaAP > 0) {
      const srcI = baseXYZ[rowBase + 2];
      const targetI = manifest.src_anchor_i + (ap[row] - manifest.ap_anchor);
      const iVal = (1 - betaAP) * srcI + betaAP * targetI;
      for (let col = 0; col < nCols; col += 1) {
        const idx = rowBase + col * 3;
        outXYZ[idx + 2] = iVal;
        outXYZ[idx + 1] = (1 - betaAP) * outXYZ[idx + 1] + betaAP * jPlane;
      }
    }
  }

  if (params.flipXZ) {
    for (let idx = 1; idx < outXYZ.length; idx += 3) {
      outXYZ[idx] = 2 * manifest.flip_y0 - outXYZ[idx];
    }
  }
}

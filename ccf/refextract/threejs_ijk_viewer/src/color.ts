export type ColorMode = "r_um" | "t_lookup" | "axis" | "leiden";

function clamp01(x: number): number {
  if (x < 0) {
    return 0;
  }
  if (x > 1) {
    return 1;
  }
  return x;
}

function colorRamp01(tIn: number): [number, number, number] {
  const t = clamp01(tIn);
  if (t < 0.33) {
    const u = t / 0.33;
    return [
      Math.round(33 + u * (80 - 33)),
      Math.round(68 + u * (181 - 68)),
      Math.round(186 + u * (137 - 186)),
    ];
  }
  if (t < 0.66) {
    const u = (t - 0.33) / 0.33;
    return [
      Math.round(80 + u * (240 - 80)),
      Math.round(181 + u * (211 - 181)),
      Math.round(137 + u * (90 - 137)),
    ];
  }
  const u = (t - 0.66) / 0.34;
  return [
    Math.round(240 + u * (217 - 240)),
    Math.round(211 + u * (67 - 211)),
    Math.round(90 + u * (54 - 90)),
  ];
}

function fillContinuous(params: {
  values: Float32Array;
  minVal: number;
  maxVal: number;
  outRgb: Uint8Array;
}): void {
  const { values, minVal, maxVal, outRgb } = params;
  const den = Math.max(maxVal - minVal, 1e-12);
  for (let i = 0; i < values.length; i += 1) {
    const t = (values[i] - minVal) / den;
    const [r, g, b] = colorRamp01(t);
    const j = 3 * i;
    outRgb[j + 0] = r;
    outRgb[j + 1] = g;
    outRgb[j + 2] = b;
  }
}

function fillAxis(params: { axis: Uint8Array; outRgb: Uint8Array }): void {
  const { axis, outRgb } = params;
  for (let i = 0; i < axis.length; i += 1) {
    const j = 3 * i;
    if (axis[i] === 0) {
      outRgb[j + 0] = 220;
      outRgb[j + 1] = 116;
      outRgb[j + 2] = 44;
    } else {
      outRgb[j + 0] = 42;
      outRgb[j + 1] = 128;
      outRgb[j + 2] = 194;
    }
  }
}

function hslToRgb01(h: number, s: number, l: number): [number, number, number] {
  const hue = ((h % 1) + 1) % 1;
  const sat = Math.max(0, Math.min(1, s));
  const lum = Math.max(0, Math.min(1, l));
  if (sat <= 1e-12) {
    return [lum, lum, lum];
  }
  const q = lum < 0.5 ? lum * (1 + sat) : lum + sat - lum * sat;
  const p = 2 * lum - q;
  const t = [hue + 1 / 3, hue, hue - 1 / 3];
  const rgb = t.map((tt) => {
    let x = tt;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  }) as [number, number, number];
  return rgb;
}

function fillLeiden(params: {
  leiden: Uint16Array;
  nCategories: number;
  outRgb: Uint8Array;
}): void {
  const { leiden, nCategories, outRgb } = params;
  const n = leiden.length;
  if (nCategories <= 0) {
    throw new Error(`Invalid nCategories=${nCategories} for leiden coloring.`);
  }
  const golden = 0.618033988749895;
  for (let i = 0; i < n; i += 1) {
    const code = leiden[i];
    const hue = ((Number(code) + 1) * golden) % 1.0;
    const [r01, g01, b01] = hslToRgb01(hue, 0.62, 0.52);
    const j = 3 * i;
    outRgb[j + 0] = Math.round(255 * r01);
    outRgb[j + 1] = Math.round(255 * g01);
    outRgb[j + 2] = Math.round(255 * b01);
  }
}

export function buildColors(params: {
  mode: ColorMode;
  rUm: Float32Array;
  tLookup: Float32Array;
  axis: Uint8Array;
  leiden?: Uint16Array;
  leidenNCategories?: number;
  stats: {
    r_um_min: number;
    r_um_max: number;
    t_lookup_min: number;
    t_lookup_max: number;
  };
  outRgb: Uint8Array;
}): void {
  const { mode, rUm, tLookup, axis, stats, outRgb, leiden, leidenNCategories } = params;
  if (outRgb.length !== rUm.length * 3) {
    throw new Error(`outRgb length mismatch: ${outRgb.length} vs ${rUm.length * 3}`);
  }
  if (tLookup.length !== rUm.length || axis.length !== rUm.length) {
    throw new Error(
      `attribute length mismatch: r_um=${rUm.length}, t_lookup=${tLookup.length}, axis=${axis.length}`,
    );
  }

  if (mode === "axis") {
    fillAxis({ axis, outRgb });
    return;
  }
  if (mode === "leiden") {
    if (leiden === undefined) {
      throw new Error("No leiden codes loaded for color mode 'leiden'.");
    }
    if (leiden.length !== rUm.length) {
      throw new Error(`leiden length mismatch: ${leiden.length} vs ${rUm.length}`);
    }
    fillLeiden({
      leiden,
      nCategories: leidenNCategories ?? 1,
      outRgb,
    });
    return;
  }
  if (mode === "r_um") {
    fillContinuous({
      values: rUm,
      minVal: stats.r_um_min,
      maxVal: stats.r_um_max,
      outRgb,
    });
    return;
  }
  fillContinuous({
    values: tLookup,
    minVal: stats.t_lookup_min,
    maxVal: stats.t_lookup_max,
    outRgb,
  });
}

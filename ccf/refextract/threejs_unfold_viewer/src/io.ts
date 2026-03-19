export interface UnfoldManifest {
  version: number;
  source: string;
  n_rows: number;
  n_cols: number;
  n_vertices: number;
  n_faces: number;
  anchor_col: number;
  anchor_row: number;
  phase1_frac_default: number;
  ap_anchor: number;
  src_anchor_i: number;
  j_plane: number;
  flip_y0: number;
  reference_lines: {
    n_points: number;
    n_ranges: number;
    kinds: {
      coronal: number;
      sagittal: number;
    };
    defaults: {
      ap_hline_step_um: number;
      sagittal_k_step: number;
      sagittal_n_sample: number;
    };
  };
  files: {
    positions_f32: string;
    faces_u32: string;
    colors_u8: string;
    seglen_f32: string;
    theta_f32: string;
    ap_um_f32: string;
    neo_t_support_u8?: string;
    line_points_f32: string;
    line_ranges_u32: string;
  };
}

export interface LoadedAssets {
  manifest: UnfoldManifest;
  positions: Float32Array;
  faces: Uint32Array;
  colors: Uint8Array;
  segLen: Float32Array;
  theta: Float32Array;
  ap: Float32Array;
  neoTSupport: Uint8Array | null;
  linePoints: Float32Array;
  lineRanges: Uint32Array;
}

function joinUrl(base: string, file: string): string {
  const b = base.endsWith("/") ? base.slice(0, -1) : base;
  const f = file.startsWith("/") ? file.slice(1) : file;
  return `${b}/${f}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const raw = await res.text();
  try {
    return JSON.parse(raw) as T;
  } catch (err) {
    const preview = raw.slice(0, 160).replace(/\s+/g, " ");
    throw new Error(
      `Failed to parse JSON from ${url}. Response begins with: '${preview}'. ` +
        `This usually means the assets path is wrong and HTML was returned instead.`,
    );
  }
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return await res.arrayBuffer();
}

export async function loadAssets(baseUrl: string): Promise<LoadedAssets> {
  const manifestUrl = joinUrl(baseUrl, "manifest.json");
  const manifest = await fetchJson<UnfoldManifest>(manifestUrl);
  if (manifest.version !== 2) {
    throw new Error(`Unsupported manifest version ${manifest.version}`);
  }
  const neoMaskUrl = manifest.files.neo_t_support_u8
    ? joinUrl(baseUrl, manifest.files.neo_t_support_u8)
    : null;
  const [
    positionsBuf,
    facesBuf,
    colorsBuf,
    segLenBuf,
    thetaBuf,
    apBuf,
    neoMaskBuf,
    linePointsBuf,
    lineRangesBuf,
  ] = await Promise.all([
    fetchBytes(joinUrl(baseUrl, manifest.files.positions_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.faces_u32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.colors_u8)),
    fetchBytes(joinUrl(baseUrl, manifest.files.seglen_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.theta_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.ap_um_f32)),
    neoMaskUrl ? fetchBytes(neoMaskUrl) : Promise.resolve(null),
    fetchBytes(joinUrl(baseUrl, manifest.files.line_points_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.line_ranges_u32)),
  ]);
  return {
    manifest,
    positions: new Float32Array(positionsBuf),
    faces: new Uint32Array(facesBuf),
    colors: new Uint8Array(colorsBuf),
    segLen: new Float32Array(segLenBuf),
    theta: new Float32Array(thetaBuf),
    ap: new Float32Array(apBuf),
    neoTSupport: neoMaskBuf ? new Uint8Array(neoMaskBuf) : null,
    linePoints: new Float32Array(linePointsBuf),
    lineRanges: new Uint32Array(lineRangesBuf),
  };
}

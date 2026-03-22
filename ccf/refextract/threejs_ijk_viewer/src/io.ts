import type { IjkManifest, LoadedIjkAssets } from "./types";

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
  } catch {
    const preview = raw.slice(0, 160).replace(/\s+/g, " ");
    throw new Error(
      `Failed to parse JSON from ${url}. Response starts with: '${preview}'. Check the assets path.`,
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

export async function loadIjkAssets(baseUrl: string): Promise<LoadedIjkAssets> {
  const manifestUrl = joinUrl(baseUrl, "manifest.json");
  const manifest = await fetchJson<IjkManifest>(manifestUrl);
  if (manifest.version !== 1) {
    throw new Error(`Unsupported manifest version ${manifest.version}; expected 1.`);
  }
  if (manifest.coord_space !== "ijk") {
    throw new Error(`Unsupported coord_space=${manifest.coord_space}; expected 'ijk'.`);
  }

  const wantLeiden =
    typeof manifest.files.leiden_u16 === "string" &&
    manifest.files.leiden_u16.trim() !== "" &&
    typeof manifest.files.leiden_categories_json === "string" &&
    manifest.files.leiden_categories_json.trim() !== "";

  if (
    (typeof manifest.files.leiden_u16 === "string") !==
    (typeof manifest.files.leiden_categories_json === "string")
  ) {
    throw new Error(
      "Manifest has inconsistent leiden fields: require both files.leiden_u16 and files.leiden_categories_json.",
    );
  }

  const [positionsBuf, rUmBuf, tLookupBuf, axisBuf, leidenBuf, leidenCats] =
    await Promise.all([
    fetchBytes(joinUrl(baseUrl, manifest.files.positions_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.r_um_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.t_lookup_f32)),
    fetchBytes(joinUrl(baseUrl, manifest.files.axis_u8)),
    wantLeiden ? fetchBytes(joinUrl(baseUrl, manifest.files.leiden_u16!)) : Promise.resolve(null),
    wantLeiden
      ? fetchJson<string[]>(joinUrl(baseUrl, manifest.files.leiden_categories_json!))
      : Promise.resolve(null),
  ]);

  const positions = new Float32Array(positionsBuf);
  const rUm = new Float32Array(rUmBuf);
  const tLookup = new Float32Array(tLookupBuf);
  const axis = new Uint8Array(axisBuf);
  const leiden = leidenBuf === null ? undefined : new Uint16Array(leidenBuf);
  const leidenCategories = leidenCats === null ? undefined : leidenCats;

  const n = manifest.n_points;
  if (positions.length !== n * 3) {
    throw new Error(`positions length mismatch: ${positions.length} vs ${n * 3}`);
  }
  if (rUm.length !== n || tLookup.length !== n || axis.length !== n) {
    throw new Error(
      `attribute length mismatch: r_um=${rUm.length}, t_lookup=${tLookup.length}, axis=${axis.length}, expected ${n}`,
    );
  }
  if (wantLeiden) {
    if (leiden === undefined || leidenCategories === undefined) {
      throw new Error("Internal error: wantLeiden but leiden buffers were not loaded.");
    }
    if (leiden.length !== n) {
      throw new Error(`leiden length mismatch: ${leiden.length} vs ${n}`);
    }
    if (!Array.isArray(leidenCategories) || leidenCategories.length <= 0) {
      throw new Error("Invalid leiden categories JSON (expected non-empty string array).");
    }
  }

  return { manifest, positions, rUm, tLookup, axis, leiden, leidenCategories };
}

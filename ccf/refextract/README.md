# `ccf/refextract`: cortical “mid-surface” extraction (DevCCF E15.5)

This folder contains small, script-like pipelines for extracting a 2D cortical scaffold from a 3D atlas mask and visualizing it.

## High-level goal: map sample sections to reference sections

The point of this module is to support a **one-to-one correspondence** between (1) a user-provided 2D *sample* section and
(2) a chosen 2D *reference atlas* section by working in a shared cortical chart.

At a high level:

- On the **reference atlas**, we extract a mid-surface and define a per-slice chart with a depth coordinate (`r01`) and an
  along-midline coordinate (`t_all ∈ [0,1]`) on the representative `u=0.5` curve for that slice. By design, `t_all` spans the
  *full* reference path (reference is typically longer than any sample’s coverage).
- For each **sample section**, the user selects which reference slice it maps to, creates a mask, and draws a principal curve
  in the `(t, r01)`-like chart. The curve parameter stored in the sample is `t_local` (slice-local, normalized so the anchor
  endpoints are `0` and `1`; depending on the fitting/extrapolation, `t_local` can extend beyond `[0,1]`).
  The sample's chosen reference axis/slice is typically stored alongside the `.h5ad` (e.g. `p1_landmarks.json` with
  `atlas_plane` and `atlas_slice_idx`).
- To compare sample and reference, we compute an **affine map** from sample `t_local` into reference `t_all` using that
  sample’s `t_endpoints` (loaded in `fishtools/ccf/cli_filter_h5ad_ccf.py`). Extrapolation is allowed (see `scripts/princurve/find_princurve.py`),
  so mapped `t_all` may be `<0` or `>1` outside the anchored interval. By definition, the mapping uses the reversal convention:
  `t_all = A0_all + (1 - t_local) * (A1_all - A0_all)`.

The current recommended pipeline is **EDT halfway mid-surface**:
`midsurface_neocortex_mesocortex_allocortex_ccf_3d.py` (+ viewer `*_plot.py`).

## EDT halfway mid-surface (fold-preserving)

### Inputs

- Atlas: `kim_dev_mouse_e15-5_lsfm_20um` (BrainGlobe / DevCCF E15.5).
- Mask: boolean volume `M ⊂ ℤ³` built from atlas `annotation` by taking `TERMS` and all descendants.
- Two boundary subsets on the ribbon (computed automatically):
  - `B_pial` (called `b0` in code): boundary voxels on the “outside brain” side
  - `B_inner` (called `b1` in code): boundary voxels on the “inside brain, non-cortex” side

### Field definitions

We compute Euclidean distances (in **microns**) using `scipy.ndimage.distance_transform_edt(..., sampling=resolution_um)`:

- `d_pial(x) = dist(x, B_pial)`
- `d_inner(x) = dist(x, B_inner)`

From these we derive two scalar fields:

1) **Halfway coordinate (unitless)**:

`u(x) = d_pial(x) / (d_pial(x) + d_inner(x))`  for `x ∈ M`

So `u≈0` near pial, `u≈1` near inner, and `u=0.5` is the equal-distance mid-surface.

2) **Depth coordinate in [0,1] (unitless)**:

`r01(x) = d_inner(x) / (d_pial(x) + d_inner(x)) = 1 - u(x)`

So `r01=0` near inner/ventricular, `r01=1` near pial, and `r01=0.5` on the mid-surface.

### Coordinate convention for queries/transforms

For `(axis, slice, t, r01)` coordinates used by `midsurface_coords.py`:

- `axis`: `coronal` or `sagittal`.
- `slice`: the **reference atlas** slice index on that axis (voxel coordinates).
- Slice ranges used in this repo (inclusive; DS=1): coronal `slice_i` 174-298, sagittal `slice_k` 161-235.
- `t` (aka `t_all`): per-slice **full-path** coordinate on the representative `u=0.5` midline curve (`[0,1]`, normalized
  arc-length along that slice’s curve).
- `r01`: per-point depth coordinate (`0` inner/ventricular, `1` pial).

Coronal and sagittal charts share the same underlying 3D mid-surface embedding; `--transform-to` uses cached LUTs and
reports residuals from discrete sampling.

Atlas voxel indexing convention (`ijk`):

- All atlas volumes are numpy arrays indexed as `vol[i, j, k]` (axis0, axis1, axis2). This repo uses `i/j/k` to mean those
  three voxel indices.
- For DevCCF E15.5 `kim_dev_mouse_e15-5_lsfm_20um`, BrainGlobe metadata specifies `orientation="asr"` (Anterior, Superior,
  Right). In practical anatomical terms:
  - `i` increases posterior → anterior (AP axis).
  - `j` increases ventral/inferior → dorsal/superior (DV axis).
  - `k` increases left → right (ML axis).
- Slice planes used throughout `ccf/refextract`:
  - **Coronal**: `slice_i` means fixed `i` (axis0). In-plane coordinates are `(j, k)` and when written/plotted as
    image-like coordinates we use `(y=j, x=k)`.
  - **Sagittal**: `slice_k` means fixed `k` (axis2). In-plane coordinates are `(i, j)` and when written/plotted as
    image-like coordinates we use `(y=i, x=j)`.

3) **Signed radial coordinate (microns)**:

`r_um(x) = d_inner(x) - d_pial(x)`

So `r_um=0` on the mid-surface, and **`r_um > 0` points toward pia** (by convention in this repo).

We also save thickness-like magnitude:

`thickness_um(x) = d_pial(x) + d_inner(x)`

### Mid-surface extraction

We extract the mid-surface as:

- A thin **voxel band** around `u=0.5`.
- A marching-cubes **mesh** at `u=0.5` (optionally clipped by `MIDLINE_TERMS`).

### Cropping & erosion

- Computation runs on a cropped bounding box around the fit mask (speed/memory).
- `ERODE_RADIUS_VOX_BEFORE_EDT` optionally erodes the fit mask before computing `u`.

### Outputs

All artifacts are written under:

- `ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/`

Key artifacts (non-exhaustive):

- Scalar fields: `halfway_u_3d_ds.npy`, `halfway_r01_crop.npy`, `halfway_thickness_um_crop.npy`, `halfway_r_um_crop.npy`.
- Mid-surface geometry/QC: `midsurface_halfway_midband_3d_ds.npy`, `midsurface_halfway_u0p5.ply`, `mpl3d_midsurface_halfway_edt.png`.
- Per-slice midline tables: `coronal_midline_columns.csv`, `sagittal_midline_columns.csv` (drive `(slice,t,r01) -> ijk`).
- Cached LUTs: `chart_map_*_t2d.npz`, `chart_map_*_ijk_from_tr.npz` (for fast axis transforms and coordinate lookup).

### `t` naming (reference vs sample)

There is **no global cross-slice `t`** in this workflow. All `t` coordinates are **slice-local**. A numeric `t` value from one
reference slice should not be compared to a numeric `t` from a different slice as if it were the same position.

- `t_all`: the **reference** per-slice full-path coordinate on the representative midline (`[0,1]`), defined as normalized
  arc-length along that slice's `u=0.5` curve.
- `t_local`: the **sample** per-slice principal-curve coordinate. This is the coordinate users interact with when
  drawing/curating the curve; `t_local=0` and `t_local=1` correspond to the chosen anchor endpoints on that slice.
  Values between anchors are typically in `[0,1]`, but `t_local` can be `<0` or `>1` when extrapolated.

## 2D coordinates for statistical modeling (AP/ML)

Flattening a curved 2D manifold into a 2D chart without distortion is not possible in general. The approach here is to define
a **stable, physically meaningful 2D coordinate system** on the reference midsurface that is good enough for modeling:

- **AP axis (`ap_um`)**: a 1D coordinate in microns across coronal slices, optimized so that one AP step approximates the
  typical 3D displacement of corresponding midline points between adjacent slices ("strip linking").
- **ML axis (`ml_um`)**: a 1D coordinate in microns along each slice's midline, defined as arclength along the `t_all` curve,
  but **anchored** so `ml_um=0` corresponds to the same anatomical location across slices.

Key design decisions:

- Keep `t_all` unchanged (slice-local, normalized arclength). All existing LUTs and transforms keep using `t_all`.
- Derive `ap_um` by matching adjacent coronal slice midlines (DTW on centered in-plane coordinates), then setting
  `Δap_um(slice_i→slice_{i+1}) := median_t ||P_i(t) - P_{i+1}(t')||` in 3D um, and finally `ap_um := cumsum(Δap_um)`.
  This makes AP spacing reflect how far the midsurface actually moves in 3D, rather than raw voxel index.
- Derive `ml_um` as signed arclength along the midline on each slice: `ml_um(t) = s_um(t) - s0_um(slice_i)`.
  The offset `s0_um(slice_i)` is computed by aligning each slice midline to a **fixed reference slice** (constant AP)
  and choosing the point that corresponds to a fixed `ref_t` on that reference slice.
- Map sagittal points into this chart via the existing cross-axis LUT: sag `(slice_k, t_s, r01)` → cor `(slice_i, t_c, r01)`
  using `midsurface_coords.py` LUTs, then compute `ap_um(slice_i)` and `ml_um(slice_i, t_c)`.

Scripts / artifacts:

- Build optimized AP axis (coronal): `ccf/refextract/optimize_ap_axis_from_strips.py`
  - Writes: `ap_axis_um_from_strips.npz`, `ap_axis_um_from_strips_qc.csv`, and optional QC plots under the outdir.
- Plot AP/ML heatmaps (coronal): `ccf/refextract/plot_ap_ml_heatmap.py`
  - Writes anchored heatmaps like `ap_ml_ml_um_anchored_thickness_heatmap.png`.

Notes:

- These coordinates are intended for modeling and visualization, not for exact metric computations on the surface. When you
  need accurate distances/areas, use 3D/mesh-space measurements and treat `(ap_um, ml_um)` as a chart.
- `resolution_ds_ijk_um.npy` is the preferred source of voxel size (um/voxel) for `DS`. If it is missing, pass
  `--res-ijk-um RI RJ RK` to the scripts (for this atlas with `DS=1`, it is typically `20 20 20`).

Mapping `t_local -> t_all`:

- The mapping is an **affine transform** defined by `t_endpoints` (per sample, per slice). In the current pipeline these
  endpoints live in the JSON `similarity_plus_syn_qc_zoom_masked_with_user_mask_t_axis_endpoints.json`, and are loaded into
  `.h5ad` metadata (e.g. via `fishtools/ccf/cli_filter_h5ad_ccf.py` as `uns['t_all_mapping']`).
- `t_endpoints` is **per mask/region label**, not one global span for the whole slice: it provides a `t_all` span
  `[A0_all, A1_all]` (stored as `begin`/`end` in that JSON) for each mask name, and each point inherits its
  `[A0_all, A1_all]` from its assigned mask label (e.g. `obs['ccf_adjusted']`).
  Then `t_local` is mapped into that span with **extrapolation allowed**, so mapped `t_all` can be `<0` or `>1` outside the
  anchored interval.
- If `obs['ccf_adjusted']` (or the endpoints JSON) is missing/empty, `t_all` cannot be derived and will remain `NaN`
  (which means LUT-based `(slice,t,r01) -> ijk` lookup cannot run).
- In `scripts/princurve/find_princurve.py`, the mapping is applied as:
  `t_all = A0_all + (1 - t_local) * (A1_all - A0_all)`
  where the `(1 - t_local)` accounts for the principal-curve orientation convention in that script.

### How to run

Generate:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_neocortex_mesocortex_allocortex_ccf_3d.py
```

Notes:

- Each reference slice uses a single representative midline curve derived from the `u=0.5` contour, and defines `t_all` as
  normalized arc-length along that curve.
- `midsurface_coords.py` may apply optional cross-slice smoothing when building the midline columns/LUTs; see that script for
  the specific knobs.

View overlays (coronal + sagittal sliders):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_neocortex_mesocortex_allocortex_ccf_3d_plot.py
```

The viewer prefers paths from `coronal_midline_columns.csv` / `sagittal_midline_columns.csv` (including smoothed paths) when available,
then falls back to manual override paths, then to raw per-slice `u=0.5` contours.

Query `(slice_i|slice_k, t, r01) → CCF ijk`:

Note: `midsurface_coords.py` derives per-slice midline columns directly from `halfway_u_3d_ds.npy` (+ masks), and writes
`coronal_midline_columns.csv` / `sagittal_midline_columns.csv` in `outdir`.

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --axis coronal --slice 200 --t 0.25 --r01 0.8
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --axis sagittal --slice 180 --t 0.25 --r01 0.8
```

Build LUT only (optional, default `--lut-nt=1024`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --outdir ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d --build-lut-only
```

Build IJK LUT only (optional):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --outdir ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d --build-ijk-lut-only
```

Fast transform between coronal and sagittal coordinate systems (cached 2D LUT):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --axis coronal --slice 200 --t 0.25 --r01 0.8 --transform-to sagittal
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_coords.py --axis sagittal --slice 180 --t 0.25 --r01 0.8 --transform-to coronal
```

### Three.js unfolding viewer (export + build)

Export unfolding assets (writes `manifest.json` and binary arrays under `threejs_unfold/`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/export_unfold_threejs_assets.py \
  --outdir ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d \
  --output-dir ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/threejs_unfold
```

Build and run the viewer:

```bash
cd ccf/refextract/threejs_unfold_viewer
npm install
npm run build
npm run dev
```

Open the viewer with the exported assets path:

```text
http://localhost:5173/?assets=../../out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/threejs_unfold
```

If you get a JSON parse error that starts with `<!doctype html>`, the assets URL is wrong and Vite returned `index.html`
instead of `manifest.json`. Use the URL above, or an absolute `/@fs/.../threejs_unfold` path.

Notes:

- In `*midline_columns.csv`, **`t` is `t_all`**: normalized along-midline arc-length on the full representative path (`[0,1]`; not the volumetric depth field `halfway_u_3d_ds.npy`).
- `*midline_columns.csv` `t` is slice-local; do not compare `t` numerically across different slices as if it were global.
- Query-to-`ijk` conversion uses cached per-axis segment LUTs on `(slice,t)` and linear interpolation by normalized local depth `r01`.
- LUT lookup is required for query-time conversion (no automatic exact-EDT fallback).
- `--transform-to` uses cached LUTs on `(slice,t)` and carries `r01` through as the shared depth coordinate.

### Testing

Atlas-backed conversion tests live in:

- `test/test_refextract_midsurface_coords_atlas.py`

Run the test:

```bash
CONDA_NO_PLUGINS=true conda run -n seq pytest -q test/test_refextract_midsurface_coords_atlas.py
```

The test expects the midsurface artifacts to already exist in:

- `ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/`

If missing, generate them first:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/midsurface_neocortex_mesocortex_allocortex_ccf_3d.py
```

## Notes / limitations

- This uses **Euclidean** distances (EDT). It preserves folds better than a harmonic/Laplace field, but in very tight folds Euclidean distances can still “shortcut” across nearby walls; the next step would be a **geodesic-in-mask** distance.
- Boundary seeding (`B_pial`, `B_inner`) is heuristic but deterministic; if it fails, the usual fix is adjusting erosion/cleanup so the boundary is well-defined.

### Troubleshooting: `u=0.5` self-intersections

If a slice’s extracted `u=0.5` contour self-intersects (often due to a too-thin/pinched cortex mask), the midline and LUTs can
become discontinuous. Preferred fix is upstream: strengthen mask conditioning and/or increase erosion before EDT in
`ccf/refextract/midsurface_neocortex_mesocortex_allocortex_ccf_3d.py`, then regenerate outputs.

Diagnostics:

- `CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/plot_halfway_u05_contours_ijk_3d.py --axis both --largest-only`
- `CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/plot_u05_self_intersections.py --axis both`

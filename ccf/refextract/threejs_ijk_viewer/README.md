# threejs_ijk_viewer

Vite + Three.js viewer for global CCF point clouds in `ijk` space.

## 1) Export browser assets

From repo root:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python ccf/refextract/export_ijk_threejs_assets.py \
  --summary-json results/refextract/princurve_h5ad_ijk_plot/phase1_summary.json \
  --output-dir results/refextract/princurve_h5ad_ijk_plot/threejs_ijk_assets \
  --max-points 1500000 \
  --seed 0
```

This writes `manifest.json` plus `*.bin` arrays in the output directory.

## 2) Run the viewer

```bash
cd ccf/refextract/threejs_ijk_viewer
npm install
npm run dev
```

Open:

```text
http://localhost:5173/?assets=/@fs/home/chaichontat/fishtools2/results/refextract/princurve_h5ad_ijk_plot/threejs_ijk_assets
```

If `?assets=...` is omitted, the app automatically falls back to:
`/@fs/<repo>/results/refextract/princurve_h5ad_ijk_plot/threejs_ijk_assets`.

If you see a JSON parse error that starts with HTML (`<!doctype html>`), your `assets` path is wrong.

## 3) Build check

```bash
npm run typecheck
npm run build
```

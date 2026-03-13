# CCF workflow (partial section → SyN → h5ad warp/filter)

This folder contains the “partial section to atlas” registration workflow used by the two installed CLIs:

- `ccf-warp-h5ad-spatial` (warps `AnnData.obsm["spatial"]` into atlas space using the SyN transforms)
- `ccf-filter-h5ad-ccf` (annotates an h5ad with a CCF/ImageJ ROI selection mask; can auto-annotate from atlas labels)

The workflow has three stages:

1) `ccf/register_partial_section.py` (interactive): pick rotation/flip, atlas slice, and paired landmarks; optionally pick a tissue threshold.
2) `ccf/ants_landmark_syn_init.py`: run ANTs “landmark linear init + SyN” and write a `similarity_plus_syn_summary.json`.
3) CLIs: warp `h5ad` coordinates into atlas space, then annotate by regions.

## 0) Prerequisites (what must exist in the workspace)

- A valid `Workspace` (directory with `analysis/`).
- A stitched mosaic for the ROI, used by the registration scripts:
  - `<workspace>/analysis/deconv/stitch--{roi}+{codebook}/fused.zarr`
- An exported segmentation h5ad to warp (for `ccf-warp-h5ad-spatial`):
  - `<workspace>/analysis/output/h5ads/{roi}.h5ad` (typically produced by `segment export ...`)

## 1) Phase 1: `register_partial_section.py` (interactive)

`ccf/register_partial_section.py` is a notebook-style script (pypercent / “Run cells”) that writes the phase-1 contract
under the ROI transform directory:

- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_landmarks.json`
- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_similarity.tfm`
- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_result.png`

It also has an optional “threshold” UI that writes:

- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_threshold.json`
- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_threshold_preview.png`

What you edit in the script:

- `WORKSPACE`, `ROI`, `STITCH_CODEBOOK`
- atlas parameters (`ATLAS_NAME`, `ATLAS_PLANE`)
- sample plane parameters (`SAMPLE_Z_IDX`, `SAMPLE_CHANNEL`, voxel sizes)

What you do interactively:

- pick a `prior_rotation_deg` and `prior_flip_x`
- pick an atlas slice index
- click paired landmarks (atlas point, then sample point)
- (optional but usually required) save `p1_threshold.json` for the mask threshold used downstream

## 2) Phase 2: `ants_landmark_syn_init.py` (landmark init + SyN refinement)

This step consumes the phase-1 outputs under:

- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_landmarks.json`
- `<workspace>/analysis/output/ccf-transforms/{roi}/p1_threshold.json`

…and writes a run directory under the same ROI:

- `<workspace>/analysis/output/ccf-transforms/{roi}/{run_dirname}/...`

The single most important output for downstream CLIs is:

- `<workspace>/analysis/output/ccf-transforms/{roi}/{run_dirname}/similarity_plus_syn_summary.json`

Run it directly (it is a Click app but not installed as a top-level script):

```bash
python ccf/ants_landmark_syn_init.py <workspace> <roi> --run-dirname landmark_syn_mi
```

Notes:

- The default `--run-dirname` is `landmark_syn_mi`.
- If `p1_threshold.json` is missing, this step fails fast and tells you to run the threshold cell in
  `register_partial_section.py`.

## 3) CLI: warp an h5ad into atlas space (`ccf-warp-h5ad-spatial`)

This CLI reads:

- the input h5ad from `<workspace>/analysis/output/h5ads/{roi}.h5ad` (override with `--h5ad-name`)
- the SyN summary from `<workspace>/analysis/output/ccf-transforms/{roi}/{run_dirname}/similarity_plus_syn_summary.json`

and writes (by default):

- `<workspace>/analysis/output/ccf-transforms/{roi}/{roi}.syn.h5ad`
- `<workspace>/analysis/output/ccf-transforms/{roi}/{roi}.syn.qc.png`
- `<workspace>/analysis/output/ccf-transforms/{roi}/{roi}.syn.metrics.json`

Example:

```bash
ccf-warp-h5ad-spatial <workspace> <roi> --run-dirname landmark_syn_mi
```

By default it warps `obsm["spatial"]` (XY pixel coordinates in “fused” space) into atlas crop pixel coordinates and
stores them in `obsm["spatial_ccf"]`. See `--in-key`, `--out-key`, `--input-space`, and `--output-space` for details.

## 4) CLI: annotate by CCF regions (`ccf-filter-h5ad-ccf`)

This CLI computes a region selection and writes term/atlas-derived labels into `adata.obs["ccf"]` (by default; empty
string if not selected). If you pass `--imagej-roi` (auto-detect) or `--imagej-roi-path`, the ROI name is written into
`adata.obs["ccf_adjusted"]` instead.
It does not drop observations.

Input expectations:

- `obsm["ccf"]` as a DataFrame with at least `id/acronym/name` columns, OR
- `obsm["spatial_ccf"]` (or `--coords-key`) plus access to the workspace so it can auto-annotate by sampling the atlas
  annotation slice specified by `p1_landmarks.json`.

Example (annotate cells in the Isocortex subtree for one ROI):

```bash
ccf-filter-h5ad-ccf <workspace> <roi> --term Isocortex
```

If `<roi>` is omitted, it runs for all discovered ROIs in the workspace.

Useful flags:

- multiple terms: repeat `--term ...`
- exact vs subtree match: `--match exact|subtree`
- require all terms: `--combine all`
- expand a region by distance: `--dilate-um 25`
- ImageJ ROI override: pass `--imagej-roi` to auto-detect a `RoiSet.zip`/`.roi` under
  `<workspace>/analysis/output/ccf-transforms/{roi}/{run_dirname}/mask_edit/`, or pass `--imagej-roi-path <path>` to
  select a specific file. ROI names are written to `adata.obs["ccf_adjusted"]` (by default).

When ImageJ ROI override is used and SyN outputs are present under `{run_dirname}`, the CLI also writes:

- `similarity_plus_syn_qc_zoom_masked_with_user_mask.png`
  - same fixed/warped masked zoom-style QC overlay with the user ROI mask shaded on top
  - includes the labeled t-axis overlay (major ticks at `0.1`, minor ticks at `0.05`)
- `similarity_plus_syn_qc_zoom_masked_with_user_mask_t_axis_endpoints.json`
  - per user-mask t-range on the same axis, with values normalized to `[0,1]`
  - format: `{"masks":[{"mask_index":...,"mask_name":...,"has_overlap_with_t_axis":...,"begin":...,"end":...}, ...]}`
  - `begin`/`end` are `null` when a mask does not intersect the t-axis

# `segment` CLI data contract

This document defines the data contracts for the segmentation CLI exposed by
`fishtools/segment/__init__.py` and its submodules:

- `segment extract` / `segment extract-single`
- `segment train`
- `segment run`
- `segment postproc`
- `segment overlay spots`
- `segment overlay intensity`
- `segment export`
- (advanced) `segment trt-build`, `segment distill`

The focus is on **what each command expects on disk and what it writes back**
so that it can be wired correctly into the rest of the pipeline documented in:

- `docs/cli_deconv_data_contract.md`
- `docs/cli_register_data_contract.md`
- `docs/align_prod_data_contract.md`
- `docs/cli_stitch_data_contract.md`

---

## Position in the end‑to‑end pipeline

Segmentation sits **after stitching and spot decoding**:

1. Raw tiles are deconvolved by `preprocess deconvnew` into:

   ```text
   <workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif
   ```

   See `docs/cli_deconv_data_contract.md` for details, including the TIFF
   metadata requirements (`axes`, `key`, `prenormalized`).

2. `preprocess register` aligns rounds per ROI, writing registered stacks and
   shifts under:

   ```text
   <workspace>/analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
   ```

   See `docs/cli_register_data_contract.md`.

3. `preprocess spots` (via `align_prod` and `preprocess spots optimize/batch`
   followed by `spots stitch` and `spots threshold`) decodes spots and writes
   the final ROI‑level parquet files used downstream:

   ```text
   <workspace>/analysis/output/{roi}+{sanitize(codebook)}.parquet
   ```

   See `docs/align_prod_data_contract.md`.

4. `preprocess stitch` fuses registered tiles into mosaics and Zarr stores
   under:

   ```text
   <workspace>/analysis/deconv/stitch--{roi}+{codebook}/fused.zarr
   ```

   and creates segmentation Zarrs like:

   ```text
   <workspace>/analysis/deconv/stitch--{roi}+{seg_codebook}/output_segmentation-sam.zarr
   <workspace>/analysis/deconv/stitch--{roi}+{seg_codebook}/output_segmentation-sam_postproc.zarr
   ```

   See `docs/cli_stitch_data_contract.md`.

5. The `segment` CLI consumes:

   - **Registered or stitched volumes** for training and inference.
   - **Stitched segmentation Zarrs** and **ROI‑level spots parquet** for
     overlays.
   - **Intensity Zarrs** for per‑cell intensity extraction.

   and produces:

   - Training tiles for Cellpose models.
   - Segmentation masks and Zarr stores.
   - Spot/segmentation overlays and per‑cell intensities.
   - H5AD files ready for Scanpy/Baysor/workflows via `segment export`.

---

## Shared workspace conventions

Throughout this document, `Workspace` refers to `fishtools.io.workspace.Workspace`
and its helpers:

- `ws.deconved` → `<workspace>/analysis/deconv`
- `ws.stitch(roi, codebook)` →

  ```text
  <workspace>/analysis/deconv/stitch--{roi}+{codebook}
  ```

- `ws.registered(roi, codebook)` →

  ```text
  <workspace>/analysis/deconv/registered--{roi}+{codebook}
  ```

- `ws.spots_parquet(roi, codebook)` →

  ```text
  <workspace>/analysis/output/{roi}+{sanitize(codebook)}.parquet
  ```

- `ws.segment(roi, codebook)` →

  ```text
  <workspace>/analysis/deconv/segment--{roi}+{codebook}
  ```

Segmentation overlays and exports also expect the TIFF/Zarr metadata
requirements described in `docs/cli_deconv_data_contract.md` and
`docs/cli_register_data_contract.md`, especially:

- Registered/stitched volumes carry `metadata["axes"]` (typically `"ZCYX"`).
- Channel names/bit labels are in `metadata["key"]` or `attrs["key"]` for Zarr.

---

## 1. Training data extraction (`segment extract`, `segment extract-single`)

### `segment extract`

Command:

```bash
segment extract z|ortho <workspace> [ROI] \
  --codebook <label> \
  [--out <dir>] [--dz N] [--n N] [--anisotropy N] \
  [--channels ...] [--crop N] [--threads N] [--upscale F] \
  [--seed N] [--every N] [--max-from <codebook>] \
  [--zarr] [--masks PATH] [--enrich-boundaries PATH] [--no-enrich-boundaries] \
  [--roi-points roi_points.roi]
```

**Inputs**

- `path`:

  - Must be a valid workspace root; `segment extract` wraps `Workspace(path)`.

- Registered inputs (per ROI, discovered via `run_workspace_extract`):

  - Preferred:

    ```text
    <workspace>/analysis/deconv/registered--{roi}+{codebook}/reg-*.tif
    ```

    TIFF stacks with shape `(Z, Y, X, C)` or `(Z, C, Y, X)`, normalized to a
    `(Z, Y, X, C)` volume internally.

  - Fallback (if `--zarr` or no TIFFs found):

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{codebook}/fused_n4.zarr
    ```

    A fused 4D Zarr with shape `(Z, Y, X, C)`.

- Optional **max‑projection source** (`--max-from`):

  - A codebook label resolved to `ws.registered(roi, max_from)` or a Zarr
    directory to compute an additional max‑projection channel for each tile.

- Optional **masks**:

  - Default per ROI (if `--masks` is not provided and the file exists):

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{codebook}/output_segmentation-sam_postproc.zarr
    ```

  - Explicit mask path via `--masks` overrides the default.

  - Contract:

    - Shape `(Z, Y, X)` or `(Z, 1, Y, X)`; mask is squeezed to `(Z, Y, X)`.
    - Spatial dimensions (Z, Y, X) must match the registered or fused volume.

- Optional **enrichment mask** (`--enrich-boundaries`, `--no-enrich-boundaries`):

  - Default (when enrichment is enabled and no explicit path is given):

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{codebook}/output_segmentation-sam.zarr
    ```

    used to bias slice/orthogonal sampling toward high‑diversity regions.

  - Explicit path via `--enrich-boundaries` overrides the default.

  - Shape contract same as masks (Z,Y,X).

- Optional **ROI points** (`--roi-points`):

  - For Zarr inputs only, a `.roi`/`.zip` file with ImageJ point ROIs used to
    seed extraction at specific coordinates (see `load_roi_points`).

**Outputs**

- Default output directory when `--out` is omitted:

  ```text
  <workspace>/analysis/deconv/segment/{roi}/
  ```

  via `_default_output_dir(ws, roi)`.

- File naming:

  - All extracted tiles and slices are prefixed with the ROI:

    ```text
    {roi}--<base>_z{zz}.tif
    {roi}--<base>_ortho{...}.tif
    ```

    where `<base>` is derived from the registered file stem or Zarr name.

- Image tiles:

  - Mode `"z"`:

    - Tiles are cropped Z‑slices written as `uint16` with axes `"CYX"` and
      metadata:

      ```python
      metadata = {
          "axes": "CYX",
          "channel_names": [...],
          "channels_arg": <original --channels string or None>,
          "upscale": <float>,
      }
      ```

  - Mode `"ortho"`:

    - Orthogonal slabs (XY/YZ/ZX) sampled with anisotropy and optional
      diversity enrichment; also written as `uint16` CYX TIFFs with the same
      metadata contract as above.

- Mask tiles (optional):

  - When masks are available, masks are written alongside each tile with
    filename suffix `_masks.tif` and metadata:

    ```python
    metadata = {"axes": "YX"}
    ```

  - Spatial alignment:

    - The mask slice is resampled with nearest‑neighbour to match the tile
      `upscale` factor, preserving label integrity.

**Integration**

- Primary use:

  - Provide training data for `segment train` by pointing `segment train` to
    the output directory of `segment extract` and defining model configs under
    `<out>/models/`.

- Upstream commands:

  - Require stitched/registered volumes from `preprocess stitch` and
    `preprocess register`.

- Downstream commands:

  - `segment train` consumes the tiles/masks.
  - `segment run` uses the trained models; its inputs are stitched/fused
    volumes, not the training tiles.

---

### `segment extract-single`

Command:

```bash
segment extract-single z|ortho REGISTERED \
  [--out <dir>] [--pattern "*.tif"] \
  [--dz N] [--n N] [--anisotropy N] \
  [--channels ...] [--crop N] [--threads N] \
  [--upscale F] [--seed N] \
  [--max-from PATH] [--label NAME] \
  [--masks PATH] [--enrich-boundaries PATH] [--overwrite] [--debug]
```

**Inputs**

- `REGISTERED`:

  - Either a single registered TIFF stack or a directory containing
    registered TIFF stacks (not Zarr).
  - Zarr inputs are explicitly rejected here; use `segment extract` for Zarr.
  - In directory mode:

    - All `*.tif` matching `--pattern` are processed, skipping existing
      outputs unless `--overwrite` is set.

- Optional `--max-from`:

  - A single stack or directory containing matching filenames for max‑projection.

- Optional masks / enrichment masks:

  - Same shape and alignment requirements as for `segment extract`.

**Outputs**

- File‑specific or directory‑level output:

  - If `REGISTERED` is a file:

    - Default out: `<input_parent>/segment_extract/`.
    - Label defaults to the input stem unless overridden by `--label`.

  - If `REGISTERED` is a directory:

    - Default out: `<REGISTERED>/segment_extract/`.
    - Each file uses its own stem as label unless `--label` is provided.

- Naming patterns:

  - For mode `"z"`:

    ```text
    {label}--{stem}_z*.tif         # image tiles
    {label}--{stem}_z*_masks.tif   # optional masks
    ```

  - For mode `"ortho"`:

    ```text
    {label}--{stem}_ortho*.tif
    {label}--{stem}_ortho*_masks.tif
    ```

- Metadata contracts identical to `segment extract` (CYX tiles with TIFF
  metadata and optional YX masks).

**Integration**

- Intended for ad‑hoc or debugging extraction outside of the full workspace
  layout (e.g. single stitched TIFF volumes).

---

## 2. Model training (`segment train`)

Command:

```bash
segment train <path> <name> \
  [--use-te/--no-use-te] \
  [--te-fp8/--no-te-fp8] \
  [--packed/--no-packed]
```

**Inputs**

- `path`:

  - A directory that typically contains:
    - Extracted tiles and masks (from `segment extract` or `extract-single`).
    - A `models/` subdirectory.

- Model configuration:

  - JSON config at:

    ```text
    <path>/models/<name>.json
    ```

  - The file may contain `//` line comments; they are stripped before parsing.

  - Config is validated against `TrainConfig` (see `fishtools.segment.train`).

**Outputs**

- Updated model configuration:

  ```text
  <path>/models/<name>.trained.json
  ```

  containing the final training hyperparameters, paths, and metadata.

### TrainConfig JSON format

The config file `<path>/models/<name>.json` is parsed into a `TrainConfig`
Pydantic model (`fishtools.segment.train.TrainConfig`). It must be valid JSON
(with optional `//` comment lines) and should define at least:

- `name` (string):
  - Human‑readable model name; typically matches `<name>` in the CLI.
- `base_model` (string or null):
  - Name or path of the initial Cellpose checkpoint (e.g. `"sam3d.0"`).
- `backend` (string, optional, default `"sam"`):
  - `"sam"` for transformer backend (Cellpose v4) or `"unet"` for legacy UNet.
- `channels` (array of two integers):
  - Channel indices (0‑based) exposed to Cellpose, usually matching the two
    channels used during training (e.g. `[1, 2]`).
- `training_paths` (array of strings):
  - List of relative paths (from `<path>`) to search for training data.
  - Each entry may point to:
    - A directory containing images/masks, or
    - A directory tree below which image/label pairs are discovered.
  - Images are discovered under these roots using `data_discovery._iter_image_dirs`,
    looking for files with extensions `[".tif", ".tiff", ".png", ".jpg", ".jpeg"]`
    and corresponding `_seg.npy` masks.

Common optional fields (all have defaults):

- `diameter` (number or null; default `60`):
  - Target rescale diameter (pixels) for SAM training; `null` lets the
    training loop infer per‑image diameters.
- `test_folder` (string or array of strings, or null):
  - Additional relative roots (same semantics as `training_paths`) used to
    build an explicit test set; when omitted, training proceeds without a
    dedicated test set.
- `include` / `exclude` (arrays of strings):
  - Lists of regex patterns applied to discovered image paths to keep or drop
    samples. Patterns are compiled with `re.compile` and matched against
    normalized paths; use with care to avoid filtering out all data.
- `n_epochs` (integer; default `200`):
  - Number of training epochs.
- Optimization hyperparameters:
  - `learning_rate` (float; default `0.008`)
  - `batch_size` (int; default `16`)
  - `bsize` (int; default `224`) – spatial batch size.
  - `weight_decay` (float; default `1e-5`)
  - `SGD` (bool; default `false`)
  - `optimizer` (string or null; currently `"adamw"` or `null`)
- Normalization and packing:
  - `normalization_percs` (array of two numbers; default `[1, 99.5]`)
  - `use_te` (bool; default `false`)
  - `te_fp8` (bool; default `false`)
  - `packed` (bool; default `false`)
  - `pack_k` (int; default `3`)
  - `pack_guard` (int; default `16`)
  - `pack_stripe_height` (int or null; default `68`)

Fields maintained by training (not usually set by hand):

- `train_losses` (array of numbers):
  - Updated by `segment train` to hold the per‑epoch training loss history.
- `model_md5` (string or null):
  - MD5 checksum of the trained model file; filled in by `segment train`.

Minimal example (`models/my_model.json`):

```jsonc
{
  "name": "my_model",
  "base_model": "sam3d.0",
  "backend": "sam",
  "channels": [1, 2],
  "training_paths": [
    "train"        // relative to <path>, e.g. <path>/train contains tiles + *_seg.npy
  ],
  "n_epochs": 150,
  "learning_rate": 0.008,
  "batch_size": 16,
  "include": [],
  "exclude": []
}
```

The corresponding CLI invocation would be:

```bash
segment train <path> my_model
```

which reads `<path>/models/my_model.json`, trains using the discovered
images/masks under `<path>/train`, and writes
`<path>/models/my_model.trained.json` with updated loss history and `model_md5`.

**Integration**

- Upstream:

  - Training data is usually produced by `segment extract` from stitched
    volumes under `<workspace>/analysis/deconv/stitch--{roi}+{codebook}`.

- Downstream:

  - The resulting trained checkpoint(s) referenced by
    `<name>.trained.json` are used as the `--model` input to `segment run`
    (directly or via exported Torch/TRT weights).

---

## 3. Segmentation inference (`segment run`, `segment postproc`)

### `segment run`

CLI wrapper defined in `fishtools/segment/__init__.py`:

```bash
segment run VOLUME \
  --model MODEL \
  [--channels "1,2"] [--anisotropy 4.0] \
  [--output-dir DIR] [--overwrite] \
  [--normalize-percentiles "1.0,99.0"] \
  [--save-flows] \
  [--ortho-model PATH] [--ortho-weights "wxy,wyz,wzx"] \
  [--backend sam|unet] \
  [--n N] [--seed N] [--pattern "*.tif"] \
  [--crop N] [--vanilla/--no-vanilla]
```

Internally this calls `fishtools.segment.run.run(config)` with a `RunConfig`
object.

**Inputs**

- `VOLUME`:

  - Single fused 4D TIFF: typically the fused stitched volume:

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{codebook}/fused.tif or fused_n4.tif
    ```

    with shape `(Z, C, Y, X)` or `(Z, Y, X, C)` and axes/keys consistent with
    the rest of the pipeline.

  - Or a directory containing such volumes (`--pattern` filters them).

- `MODEL`:

  - A u-Segment3D/Cellpose checkpoint compatible with either:
    - SAM backend (`--backend sam`), or
    - UNet backend (`--backend unet` with exactly two channels).

  - If a TensorRT plan exists (see `segment trt-build`), it will be used;
    otherwise Torch inference is used.

- Channels and metadata:

  - `--channels` specifies the channel indices to use; must be at least two
    and distinct.

  - `RunConfig` validates:
    - `channels` non‑negative and unique.
    - `normalize-percentiles` with `0 <= low < high <= 100`.

**Outputs**

For each processed volume:

- Output directory:

  ```text
  <volume_parent>/../segment2_5d/
  ```

  created next to the volume.

- Files:

  - Copy of the input volume (if not already there).

  - Segmentation masks:

    ```text
    segment2_5d/<volume_stem>_masks.tif
    ```

    - Integer labels, written as `uint32`.

  - Flow/metadata sidecar:

    ```text
    segment2_5d/<volume_stem>.pkl
    ```

    - Contains:
      - Raw flows and style embeddings from Cellpose.
      - Run configuration (`RunConfig`).
      - Other sidecar metadata.

  - Region properties:

    ```text
    segment2_5d/<volume_stem>_labels.parquet
    ```

    - A Polars/Parquet table with per‑label metrics (area, centroid, etc.).

**Integration**

- `segment run` is often used to generate initial masks that are then:

  - Converted into Zarr stores for stitching overlays, e.g.

    ```text
    output_segmentation-sam.zarr
    output_segmentation-sam_postproc.zarr
    ```

  - Fed into `segment postproc` for cleanup before overlays.

### `segment postproc`

Command:

```bash
segment postproc MASKS_PATH \
  [--output PATH] [--pattern "*_masks.tif"] \
  [--sigma "1.5,3.0,3.0"] \
  [--max-expansion N] [--erosion-fwhm-frac F] \
  [--v-min N] [--min-contact-fraction F] \
  [--backend cpu|cupy] \
  [--skip-smooth] [--skip-absorb] [--skip-donate]
```

**Inputs**

- `MASKS_PATH`:

  - Either a single masks TIFF file (e.g. produced by `segment run`), or a
    directory containing mask files (default pattern `*_masks.tif`).

  - Masks must be 2D or 3D (Z,Y,X) integer labels; non‑integer inputs are
    cast to integers.

**Outputs**

- If `MASKS_PATH` is a file:

  - Output mask:

    ```text
    <stem>_postproc.tif
    ```

    written alongside the input file or at `--output` if provided.

- If `MASKS_PATH` is a directory:

  - For each matching mask file:

    ```text
    <output_dir>/<stem>_postproc.tif
    ```

    where `output_dir` is `--output` if given, else the input directory.

**Integration**

- Post‑processed masks are the recommended inputs when converting to Zarr
  stores used by stitching and overlays, e.g.:

  - `output_segmentation-sam_postproc.zarr` used as:
    - Default `--masks` and enrichment mask for `segment extract`.
    - Segmentation reference for `segment overlay spots` and
      `segment overlay intensity`.

---

## 4. Spot overlays and per‑cell intensities

### `segment overlay spots`

Expose via the `segment overlay` subcommand:

```bash
segment overlay spots <workspace> [ROI] \
  --codebook <cb> \
  [--seg-codebook <seg_cb>] \
  [--spots PATH] \
  [--segmentation-name output_segmentation-sam.zarr] \
  [--overwrite/--no-overwrite] [--debug/--no-debug]
```

**Inputs**

- `path`:

  - Workspace root; wrapped into `Workspace(path)`.

- `ROI`:

  - Specific ROI or omitted for all ROIs (`*`).

- Segmentation store:

  - The stitched segmentation Zarr per ROI + segmentation codebook:

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{seg_cb}/{segmentation_name}
    ```

  - `seg_cb` defaults to the spots `--codebook` if `--seg-codebook` is not
    given.

- Tile configuration:

  - ROI‑level tile config at:

    ```text
    <workspace>/analysis/deconv/stitch--{roi}/TileConfiguration.registered.txt
    ```

  - Coordinate offsets are computed from this file and used to align stitched
    coordinates with ROI‑level coordinates.

- Spots parquet:

  - Default resolution via `Workspace.spots_parquet(roi, codebook)`:

    ```text
    <workspace>/analysis/output/{roi}+{sanitize(codebook)}.parquet
    ```

  - Or an explicit path/directory via `--spots`:
    - If directory: `<spots_dir>/{roi}+{sanitize(codebook)}.parquet`.
    - If file: reused for all ROIs in batch mode (with a warning).

**Outputs**

For each ROI:

- Inside the segmentation Zarr directory:

  ```text
  <workspace>/analysis/deconv/stitch--{roi}+{seg_cb}/{segmentation_name}/chunks+{cb_token}/
    ident-{z:02d}.parquet
    polygons-{z:02d}.parquet
  ```

  where:

  - `cb_token = Workspace.sanitize_codebook_name(codebook)`.
  - `ident-*.parquet` rows:
    - `spot_id`, `target`, `label` linking decoded spots to segmentation
      labels at each z‑slice.
  - `polygons-*.parquet` rows:
    - `polygon_id`, `label`, `area`, `centroid_y`, `centroid_x` describing
      segmented regions.

**Integration**

- Upstream:

  - Requires:
    - Final filtered spots parquet from `preprocess spots threshold`.
    - Stitched segmentation Zarrs (for the segmentation codebook).

- Downstream:

  - `segment export` aggregates these `ident` chunks along with intensity
    overlays into per‑cell counts and H5AD files.

---

### `segment overlay intensity`

This is exposed as a lazy subcommand under `segment overlay intensity`:

```bash
segment overlay intensity <workspace> [ROI] \
  --seg-codebook <seg_cb> \
  --intensity-codebook <cb> \
  [--segmentation-name output_segmentation-sam.zarr] \
  [--intensity-store fused.zarr] \
  [--channel NAME] [--threads N] [--overwrite]
```

**Inputs**

- Segmentation Zarr (same as `overlay spots`):

  ```text
  <workspace>/analysis/deconv/stitch--{roi}+{seg_cb}/{segmentation_name}
  ```

- Intensity Zarr store:

  - Resolved via `StitchPaths` and `resolve_intensity_store`:

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{intensity_cb}/{intensity_store}
    ```

  - Default `--intensity-store` is `fused.zarr`.

  - Contract:
    - Zarr array with `attrs["key"]` listing channel names.
    - Spatial axes match the segmentation Zarr (Z,Y,X).

- Channels:

  - If `--channel` is provided:
    - Only that channel is processed.

  - Otherwise:
    - Discover all channels from `attrs["key"]` on the intensity Zarr.

**Outputs**

- Per ROI, per channel, per slice:

  - Inside the segmentation Zarr directory:

    ```text
    <workspace>/analysis/deconv/stitch--{roi}+{seg_cb}/{segmentation_name}/
      intensity_{channel}/intensity-{z:02d}.parquet
    ```

  - Each Parquet file contains a regionprops‑like table for labels, including
    intensity metrics.

**Integration**

- Upstream:

  - Requires intensity Zarrs produced during stitching (e.g. fused volumes).

- Downstream:

  - `segment export` merges:
    - Polygon/ident data from `segment overlay spots`.
    - Intensity shards from `segment overlay intensity`.

---

## 5. Export to H5AD and per‑cell tables (`segment export`)

Command:

```bash
segment export <workspace> [ROI] \
  --seg-codebook <seg_cb> \
  --codebook <cb> [--codebook <cb2> ...] \
  [--segmentation-name output_segmentation-sam.zarr] \
  [--channels "auto"|"ch1,ch2,..."] \
  [--out-dir DIR] [--diag]
```

**Inputs**

- Workspace and ROIs:

  - `Workspace(path)` and either:
    - A specific `ROI`, or
    - All ROIs in the workspace when `ROI` is omitted.

- Segmentation and intensity artifacts:

  - Segmentation Zarr per ROI + segmentation codebook:

    ```text
    seg_zarr = <workspace>/analysis/deconv/stitch--{roi}+{seg_cb}/{segmentation_name}
    ```

  - Spot/segmentation overlays from `segment overlay spots`:

    ```text
    seg_zarr/chunks+{codebook}/ident-*.parquet
    seg_zarr/chunks+{codebook}/polygons-*.parquet
    ```

  - Intensity overlays from `segment overlay intensity`:

    ```text
    seg_zarr/intensity_{channel}/intensity-*.parquet
    ```

- Channels:

  - `--channels auto` (default):
    - Scans `intensity_*` directories under the segmentation Zarr across all
      ROIs to construct the set of channels.

  - Explicit `--channels "ch1,ch2,..."` overrides auto detection.

**Outputs**

Let `primary_cb` be the first `--codebook` provided and
`cb_token = Workspace.sanitize_codebook_name(primary_cb)`, and
`seg_stem = Path(segmentation_name).stem`.

- If a single ROI is selected:

  - In `ws.output`:

    ```text
    <workspace>/analysis/output/polygons+{roi}+{cb_token}+{seg_stem}.parquet
    <workspace>/analysis/output/h5ads/{roi}.h5ad
    ```

  - `polygons+{cb_token}.parquet`:
    - Consolidated per‑cell table containing centroids, areas, region metrics,
      and intensity summaries across channels.

  - `{cb_token}.h5ad`:
    - AnnData object with:
      - `X`: gene/transcript counts per cell (non‑Blank targets).
      - `obs`: per‑cell metadata (including spatial centroids).
      - `obsm["spatial"]`: XY coordinates (for spatial plotting).
      - QC metrics and filtered genes/cells.

- If multiple ROIs, the command writes one `.h5ad` (and one per-cell parquet) per ROI under the same directories above.

**Integration**

- This is the final bridge from segmentation and decoded spots into
  analysis‑ready data:

  - Takes:
    - Decoded/thresholded spots (via `preprocess spots threshold` →
      `segment overlay spots`).
    - Stitched intensity volumes (via `preprocess stitch` →
      `segment overlay intensity`).
    - Segmentation masks (via `segment run` + `segment postproc` and
      conversion to Zarr).

  - Produces:
    - Per‑cell gene expression matrices and QC metrics in H5AD.
    - Per‑cell polygons and intensity summaries in parquet.

---

## 6. Advanced helpers (`segment trt-build`, `segment distill`)

### `segment trt-build`

Command:

```bash
segment trt-build MODEL \
  [--batch-size N] [--backend sam|unet] [--opset 22]
```

**Inputs**

- `MODEL`:

  - Trained checkpoint path compatible with `TrainConfig`.

**Outputs**

- A TensorRT engine plan file written next to the model; path is resolved via
  `plan_path_for_device(MODEL, device_name)` inside `fishtools.segment.train`.

**Integration**

- When a TRT plan exists for a model (and optional ortho model), `segment run`
  will prefer it over Torch inference for the matching CUDA device.

### `segment distill`

Command:

```bash
segment distill <path> <outdir>
```

**Inputs**

- `path`:

  - Training directory, typically the same `path` passed to `segment train`.

- `outdir`:

  - Output directory for distilled models.

**Outputs**

- Distilled model artifacts in `outdir` plus a stream of warnings printed via
  `click.echo`.

**Integration**

- Advanced/experimental; core pipeline integration is through the trained
  checkpoints referenced in `TrainConfig` and passed to `segment run`.

---

## 7. End‑to‑end linkage summary

To summarize how the `segment` CLI fits into the full pipeline contracts:

1. **Deconvolution**:

   - `preprocess deconvnew` (see `docs/cli_deconv_data_contract.md`) writes:

     ```text
     analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif
     analysis/deconv_scaling/{round}.txt
     ```

2. **Registration**:

   - `preprocess register` (see `docs/cli_register_data_contract.md`) writes:

     ```text
     analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
     analysis/deconv/shifts--{roi}+{codebook}/...
     ```

3. **Spot decoding**:

   - `preprocess spots optimize` (mandatory stepwise optimization),
     `preprocess spots batch`, `preprocess spots stitch`, and
     `preprocess spots threshold` (see `docs/align_prod_data_contract.md`)
     write:

     ```text
     analysis/deconv/opt_{codebook}[+{roi}]/...
     analysis/deconv/registered--{roi}+{codebook}/decoded-*.pkl
     analysis/deconv/{roi}+{sanitize(codebook)}.parquet  # intermediate
     analysis/output/{roi}+{sanitize(codebook)}.parquet   # final spots parquet
     ```

4. **Stitching**:

   - `preprocess stitch register/fuse/combine` (see
     `docs/cli_stitch_data_contract.md`) write:

     ```text
     stitch--{roi}/TileConfiguration.registered.txt
     analysis/deconv/stitch--{roi}+{codebook}/fused.zarr
     analysis/deconv/stitch--{roi}+{codebook}/output_segmentation*.zarr
     ```

5. **Segmentation and overlays** (`segment` CLI, this document):

   - `segment extract` / `extract-single`: training tiles + masks from
     registered/stitched volumes.
   - `segment train` / `trt-build` / `distill`: training and export of
     segmentation models.
   - `segment run` / `postproc`: segmentation masks and post‑processed masks
     used to build segmentation Zarrs.
   - `segment overlay spots`: attaches decoded spots from
     `analysis/output/{roi}+{codebook}.parquet` to segmentation labels and
     writes `chunks+{codebook}/ident*.parquet` and polygon metadata into the
     segmentation Zarr.
   - `segment overlay intensity`: attaches stitched intensity Zarrs to
     segmentation masks and writes `intensity_{channel}/intensity-*.parquet`
     into the segmentation Zarr.
   - `segment export`: aggregates ident + intensity shards into
     per‑cell polygons parquet and H5AD matrices under either the segmentation
     Zarr (single ROI) or `analysis/output/` (multi‑ROI), ready for downstream
     analysis.

Together with the other CLI data‑contract docs, this document should give a
complete picture of how segmentation artifacts are wired into the rest of the
FISHtools pipeline.

# `preprocess stitch` (`cli_stitch`) data contract

This document defines the data contracts for the Click‑based stitching CLI in
`fishtools/preprocess/cli_stitch.py`.

Public commands:

- `preprocess stitch register-simple`
- `preprocess stitch register`
- `preprocess stitch fuse`
- `preprocess stitch combine`
- `preprocess stitch n4`
- `preprocess stitch run` (pipeline wrapper)

These commands sit **after** registration (`preprocess register`) and typically run on a workspace
where registered images already exist.

---

## Position in the end‑to‑end pipeline

Conceptually, `preprocess stitch` is the **image‑geometry** branch of the pipeline:

1. `preprocess register` aligns deconvolved tiles across rounds and writes per‑ROI, per‑codebook
   registered stacks:

   ```text
   analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
   ```

2. `preprocess stitch register` uses these registered stacks (and stage position CSVs) to generate
   `TileConfiguration.registered.txt` describing tile positions in a common mosaic frame.

3. `preprocess stitch fuse` and `preprocess stitch combine`:
   - Extract per‑channel tiles, apply optional illumination correction, downsample, and pass them
     to ImageJ Grid/Collection stitching.
   - Consolidate ImageJ outputs into a 4D Zarr mosaic:

     ```text
     analysis/deconv/stitch--{roi}+{codebook}/fused.zarr  # (Z, Y, X, C)
     ```

4. `preprocess stitch n4` optionally applies N4 bias‑field correction to the stitched mosaics, and
   writes corrected Zarr/TIFF volumes.

In parallel, `preprocess spots` (`align_prod.py`) operates directly on the same registered stacks
to decode spots per tile, then `preprocess spots stitch` projects those decoded spots into the same
global coordinate system defined by `TileConfiguration.registered.txt`. This is why the
contracts for registered TIFFs and tile configurations are shared between `cli_register`,
`cli_stitch`, and the spots pipeline. Downstream, the `segment` CLI consumes the stitched
volumes and tile configurations documented here when running segmentation, spot overlays, and
exports (see `docs/cli_segment_data_contract.md`).

---

## Workspace and directory layout

The `preprocess stitch` commands expect a fishtools workspace managed by `Workspace`, with at least:

```text
<workspace>/
  workspace.DONE or *.DONE
  analysis/
    deconv/
      {round}--{roi}/                  # deconvolved tiles (optional; used for coarse-shift mode)
      registered--{roi}+{codebook}/    # registered tiles from cli_register
      stitch--{roi}/                   # stitching dirs (no codebook)
      stitch--{roi}+{codebook}/        # stitching dirs (with codebook)
      fields+{codebook_slug}/          # TCYX field stores (illumination correction; optional)
```

The **registration working directory** for `stitch` is typically:

```text
<reg_root> = <workspace>/analysis/deconv
```

All CLI `path` arguments in this module either point to:

- the workspace root (`<workspace>`), or
- the registration working directory (`<reg_root>`), or
- a specific stitching subdirectory (`stitch--...`), depending on the command.

`Workspace(path)` is used throughout to normalize and resolve the root and derived paths:

- `ws.deconved = <workspace>/analysis/deconv`
- `ws.registered(roi, codebook) = ws.deconved / f"registered--{roi}+{codebook}"`
- `ws.stitch(roi, codebook=None|label) = ws.deconved / f"stitch--{roi}[+{codebook}]"`.

---

## Registered input TIFF contract (inputs from `preprocess register`)

Many commands operate on registered tiles produced by `preprocess register`:

- Path pattern:

  ```text
  <reg_root>/registered--{roi}+{codebook}/reg-{idx:04d}.tif
  ```

- Shape and dtype:

  ```text
  (Z, C, Y, X), dtype=uint16 or compatible
  ```

- Required TIFF metadata:

  - `axes` – typically `"ZCYX"` (required by some tools).
  - `key` – list of channel labels (bit names, wavelengths, etc.) used for:
    - channel selection, labeling, and downstream field correction.

If `key` is absent, commands fall back to synthetic labels like `channel_0`, `channel_1`, etc.

---

## Command: `stitch register-simple`

### CLI signature

```text
preprocess stitch register-simple PATH --tileconfig TILECONFIG [--fuse] [--downsample N] [--config JSON]
```

Arguments:

- `PATH`: directory where ImageJ will read/write its TileConfiguration and fused outputs.

Options:

- `--tileconfig TILECONFIG` (required)
  - Source tile configuration file, typically a `TileConfiguration.txt` or
    `TileConfiguration.registered.txt` produced by other tools.
- `--fuse`
  - If set, ImageJ is instructed to fuse tiles (not just compute overlap).
- `--downsample N` (default: `2`)
  - Integer factor by which to downsample coordinates in `TILECONFIG` before writing
    `PATH/TileConfiguration.txt`.
- `--config JSON`
  - Optional JSON config; only the `stitching` section (`StitchingConfig`) is used to fill
    ImageJ parameters (memory, threads, thresholds).

### Inputs

1. **Tile configuration file**

   - Path: `TILECONFIG`
   - Format: ImageJ Grid/Collection style:

     ```text
     dim=2
     0000.tif; ; (x0, y0)
     0001.tif; ; (x1, y1)
     ...
     ```

2. **ImageJ installation**

   - `run_imagej` expects `ImageJ-linux64` (Fiji) under `${HOME}/Fiji.app/`.

### Outputs

Under `PATH`:

- `TileConfiguration.txt`
  - Downsampled copy of the input configuration (positions divided by `downsample`).
- ImageJ artifacts:
  - `TileConfiguration.registered.txt` – registered TileConfiguration created by ImageJ.
  - `img_t1_z1_c1` – fused output (if `--fuse` is enabled). The CLI renames this file into
    `fused_*.tif` as appropriate in higher‑level commands; `register-simple` itself leaves it
    in ImageJ’s default location.

If ImageJ is not installed or `TileConfiguration.registered.txt` cannot be created, a
`FileNotFoundError` or `RuntimeError` is raised.

---

## Command: `stitch register`

### CLI signature

```text
preprocess stitch register PATH ROI [OPTIONS]
```

Arguments:

- `PATH`: workspace root (or a path resolvable to a workspace).
- `ROI`: ROI identifier, or `"*"` to process all ROIs (via `@batch_roi`).

Options (subset):

- `--codebook NAME`
  - Needed when multiple codebooks exist for the same ROI; identifies which
    `registered--{roi}+{codebook}` directory to use.
- `--position_file CSV`
  - Optional explicit positions CSV. If omitted, `Workspace.tile_positions_csv(roi)` is used.
- `--idx, -i`
  - Channel index to extract from registered TIFFs for feature‑based tile registration.
- `--fid`
  - Use fiducial images (`ws.fid(roi, idx)`) instead of registered CYX stacks when extracting.
- `--threshold FLOAT`
  - Regression threshold for ImageJ overlap computation (passed into the macro; optional).
- `--overwrite`
  - Rebuild TileConfiguration / rerun ImageJ even if outputs exist.
- `--max-proj`
  - Use maximum projection over Z/C when extracting alignment images.
- `--drop-disconnected/--keep-disconnected`
  - If enabled (default), drop tiles not connected to the main component in a 4‑neighborhood graph.
- `--config JSON`
  - Optional project config; only the `stitching` section is used.

### Inputs

Per ROI:

1. **Registered images**

   - Required directory:

     ```text
     ws.registered(roi, codebook) = analysis/deconv/registered--{roi}+{codebook}
     ```

   - Required files:

     ```text
     reg-*.tif     # from preprocess register
     ```

2. **Stage positions**

   - CSV path from either:
     - `--position_file`, or
     - `Workspace.tile_positions_csv(roi)` (typically `<workspace>/<roi>.csv`).

   - Format: no header, 2 columns `[y, x]` stage coordinates per tile in the same index order
     as the TIFF files.

3. **Optional fiducial images**

   - When `--fid` is used, per‑tile fiducial TIFFs:

     ```text
     ws.fid(roi, idx)  # usually analysis/deconv/fids--{roi}/fids-{idx:04d}.tif
     ```

### Outputs

Under `ws.stitch(roi)`:

1. **Per‑tile extracts for registration (temporary)**

   - Files:

     ```text
     stitch--{roi}/0000.tif
     stitch--{roi}/0001.tif
     ...
     ```

   - These are extracted from registered or fiducial stacks and written with metadata:

     ```python
     {
       "axes": "YX",
       "key": [channel_name],
       "processing": {
         "trim": trim,
         "downsample": downsample,
         "reduce_bit_depth": 0,
         "max_proj": bool(max_proj),
       },
     }
     ```

   - After registration completes, all digit‑named TIFFs in the stitch directory are removed.

2. **Tile configuration files**

   - `TileConfiguration.txt`
     - Created from the positions CSV filtered to the tiles that were extracted.
   - `TileConfiguration.registered.txt`
     - Written by ImageJ during the registration run.

3. **Stitch layout plot**

   - Saved under:

     ```text
     <workspace>/analysis/output/stitch_layout--{roi}.png
     ```

   - Visualizes tile positions and indices in micrometer coordinates.

### Error handling

`stitch register` fails when:

- No registered directory exists for the ROI/codebook.
- Positions CSV is missing or does not match tile indices.
- TileConfiguration parsing fails (invalid format).
- ImageJ fails to write `TileConfiguration.registered.txt`.

Disconnected tiles (in 4‑neighborhood space) are logged and optionally dropped.

---

## Command: `stitch fuse`

### CLI signature

```text
preprocess stitch fuse PATH ROI [OPTIONS]
```

Arguments:

- `PATH`: registration working directory (`<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` (via `@batch_roi("registered--*", include_codebook=True, split_codebook=True)`).

Options (subset):

- `--codebook NAME`
  - Required in the standard registered‑fusion mode.
- `--tile_config TILECONFIG`
  - Override TileConfiguration path; otherwise `ws.tileconfig(roi)` is used.
- `--split N` (default: `1`)
  - Number of tile groups (splits) per channel for ImageJ fusion. `final_stitch` recombines them.
- `--overwrite`
  - Remove existing fused outputs and re‑run.
- `--downsample, -d N` (default: `2`)
- `--subsample-z N` (default: `1`)
- `--is-2d`
  - Run in 2D mode; see `extract`.
- `--threads, -t` (default: `8`) – for extraction and ImageJ invocations.
- `--channels STR` (default: `"all"`)
  - Comma‑separated slice syntax or `"all"`; parsed into channel indices.
- `--max-proj`
  - Fuse maximum projections instead of full Z stacks.
- `--max-from NAME`
  - Use a maximum projection from another registered codebook as an additional channel.
- `--field-zarr PATH`
  - TCYX field Zarr store (see illumination section) for applying correction during extraction.
- `--json-config JSON`
  - Optional project config; only `.stitching` is used.
- `--coarse-shifts coarse_shifts.json`
  - Enable coarse‑shifted fusion directly on deconvolved images (see below).
- `--round-name ROUND`
  - Specific round to use within `coarse_shifts.json` when multiple rounds are present.

### Modes

1. **Standard registered fusion (typical)**

   - Requires `--codebook`.
   - Uses:

     ```text
     path_img = ws.registered(roi, codebook)
     stitch_dir = ws.stitch(roi, codebook)
     ```

2. **Coarse‑shifted fusion (on deconvolved tiles)**

   - Requires `--coarse-shifts coarse_shifts.json`.
   - Uses:

     ```text
     stitch_dir = ws.deconved / f"stitch--{roi}--shifted-{round_name}"
     path_img = ws.deconved / f"{round_name}--{roi}"
     ```

   - Coarse shifts JSON schema (produced by `preprocess register fix-shifts`):

     ```jsonc
     {
       "reference": "2_10_18",
       "use_fft": false,
       "tiles": {
         "0001": {
           "1_9_17": {"dx": 12.3, "dy": -4.5, "magnitude": 13.1, "residual": 0.12},
           "3_11_19": { "...": "..." }
         },
         "0002": { "...": "..." }
       }
     }
     ```

   - `fuse` selects a single `round_name` and builds a `shift_lookup` mapping tile index to `(dx, dy)`.
   - Deconvolved tiles are assumed to be shaped `[ZC]YX` and reshaped to `ZCYX` using:
     - `n_channels_reshape = len(round_name.split("_"))`
     - `n_fids_reshape = 2` (fiducials removed from the end).

### Inputs

Per ROI:

1. **Tile configuration**

   - For standard fusion:

     ```text
     ws.tileconfig(roi)  # typically stitch--{roi}/TileConfiguration.registered.txt
     ```

2. **Input TIFF tiles**

   - Standard fusion:

     ```text
     path_img = ws.registered(roi, codebook)
     path_img/*.tif
     ```

   - Coarse‑shifted fusion:

     ```text
     path_img = ws.deconved / f"{round_name}--{roi}"
     path_img/{round_name}-*.tif
     ```

3. **Optional field Zarr**

   - TCYX field store used to correct illumination during extraction.

### Outputs

Under `stitch_dir`:

1. **Per‑tile channel folders**

   - For each Z and channel, `extract` creates:

     ```text
     stitch--{roi}[+{codebook}]/ZZ/CC/{idx:04d}.tif
     ```

     where:

     - `ZZ` is two‑digit Z index.
     - `CC` is two‑digit channel index.

   - Metadata:

     ```python
     {
       "axes": "YX",
       "key": [channel_name],
       "processing": {...},
     }
     ```

2. **Intermediate fused TIFFs per channel and split**

   - After running ImageJ Grid/Collection stitching on each channel/split folder:

     ```text
     stitch--{roi}[+{codebook}]/CC/fused_{CC:02d}-{split}.tif
     ```

3. **Final per‑channel fused TIFF**

   - When `split > 1`, `final_stitch` consolidates `fused_*` images into:

     ```text
     stitch--{roi}[+{codebook}]/CC/fused.tif
     ```

     with metadata:

     ```python
     {
       "axes": "YX",
       "key": [channel_label],
       "processing": {"split": split_count},
     }
     ```

4. **Fused Zarr volume**

   - `combine` (see next section) turns the per‑channel files into a 4D Zarr array:

     ```text
     stitch--{roi}[+{codebook}]/fused.zarr
     ```

---

## Command: `stitch combine`

### CLI signature

```text
preprocess stitch combine PATH ROI --codebook CODEBOOK [--chunk-size N --overwrite]
```

Arguments:

- `PATH`: registration working directory (`<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` (via `@batch_roi("stitch--*", include_codebook=True, split_codebook=True)`).

Options:

- `--codebook CODEBOOK` (required)
- `--chunk-size N` (default: `2048`)
  - Used as Y/X chunk size for the Zarr array.
- `--overwrite`
  - Currently not used to gate Zarr creation (combine always writes a fresh array).

### Inputs

For each resolved ROI (`ws.resolve_rois`):

1. **Stitched folders**

   - Root:

     ```text
     stitched_dir = ws.stitch(roi, codebook)
     ```

   - Expected structure:

     ```text
     stitch--{roi}+{codebook}/
       ZZ/
         CC/
           fused_{CC:02d}-1.tif
     ```

   - `walk_fused(stitched_dir)` returns `dict[int, list[Path]]` mapping Z index to channel folder paths.

2. **Registered TIFFs for channel labels (optional)**

   - `combine` may inspect a representative registered TIFF in
     `ws.registered(roi, codebook)` to derive channel labels from metadata["key"].

### Outputs

Under `stitched_dir`:

1. **Zarr fused volume**

   - Path:

     ```text
     fused.zarr
     ```

   - Shape:

     ```text
     (Z, Y, X, C)
     ```

     where:

     - `Z` is `max(z_idx) + 1` from `folders_by_z`.
     - `C` is `max(channel_index) + 1` inferred from folder names.
     - `Y`, `X` are the spatial dimensions of the fused TIFFs.

   - Chunks:

     ```text
     (1, chunk_size, chunk_size, C)
     ```

   - Attributes:

     - `attrs["key"]` (optional):
       - When channel names are available or can be inferred, this is a list of channel
         labels (e.g. `["dapi", "bit1", "bit2"]`).
       - When names cannot be determined reliably, `key` may be omitted.

2. **Per‑Z thumbnails**

   - Thumbnails directory:

     ```text
     stitch--{roi}+{codebook}/thumbnails/
     ```

   - Files:

     ```text
     thumbnail_z{z_idx:02d}.png
     ```

   - Each thumbnail is a quicklook visualization of a given Z plane.

Older behavior that wrote `normalization.json` has been removed.

---

## Command: `stitch n4`

### CLI signature

```text
preprocess stitch n4 WORKSPACE ROI --codebook CODEBOOK --z-index Z [OPTIONS]
```

Arguments:

- `WORKSPACE`: workspace root (contains `analysis/`).
- `ROI`: ROI identifier or `"*"` (via `@batch_roi("stitch--*", include_codebook=True, split_codebook=True)`).

Options (subset):

- `--codebook CODEBOOK` (required)
- `--channels STR`
  - Comma‑separated channel names or indices to correct; default: all channels.
- `--shrink N` (default: `4`)
- `--spline-lowres-px FLOAT` (default: `128.0`)
- `--z-index Z` (required)
- `--threshold STR`
- `--field-output PATH`
- `--corrected-output PATH`
- `--apply/--field-only`
- `--overwrite`
- `--single-plane`
- `--debug`
- `--unsharp-mask/--no-unsharp-mask`

### Inputs

Within `run_cli_workflow` (from `fishtools.preprocess.n4`), the command expects:

- Fused Zarr or TIFF mosaics under:

  ```text
  ws.stitch(roi, codebook) = analysis/deconv/stitch--{roi}+{codebook}
  ```

  typically:

  - `fused.zarr`
  - or per‑channel `fused.tif`.

### Outputs

`n4` delegates to `run_cli_workflow` and reports the first `N4Result`:

- `field_path`: correction field TIFF (bias field).
- `corrected_path`: corrected imagery (Zarr or TIFF), if `--apply` is enabled.

These paths are printed to stdout; the actual locations are determined by the workflow in
`fishtools.preprocess.n4`.

---

## Command: `stitch run` (pipeline wrapper)

### CLI signature

```text
preprocess stitch run WORKSPACE ROI --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `WORKSPACE`: workspace root.
- `ROI`: ROI identifier or `"*"` (via `@batch_roi()`).

Options:

- `--codebook CODEBOOK` (required)
- `--tile_config TILECONFIG`
- `--overwrite`
- `--downsample, -d N` (default: `2`)
- `--subsample-z N` (default: `1`)
- `--threads, -t N` (default: `8`)
- `--channels STR` (default: `"-3,-2,-1"`)

### Behavior

For each ROI:

1. `stitch register` with:

   ```text
   fid=True, position_file=None, overwrite as given
   ```

2. `stitch fuse` with:

   ```text
   split=True, downsample, threads, channels, subsample_z
   ```

3. `stitch combine` to write `fused.zarr` and thumbnails.

### Inputs and outputs

`stitch run` is essentially a composition of the contracts described above:

- Inputs:
  - Registered TIFFs from `preprocess register`.
  - Fiducial stacks (for registration).
  - Position CSVs.
- Outputs:
  - TileConfiguration files and layout plots.
  - Stitched per‑channel TIFFs.
  - Consolidated Zarr mosaics in `stitch--{roi}+{codebook}/fused.zarr`.

---

## Summary

`cli_stitch` turns registered tiles into stitched mosaics and Zarr volumes:

- It consumes:
  - Registered TIFFs (`registered--{roi}+{codebook}/reg-*.tif`) with `ZCYX` layout and `key`
    metadata.
  - Position CSV files to build TileConfiguration for ImageJ.
  - Optional TCYX field Zarr stores for illumination correction.
  - Optional coarse shift JSONs from `preprocess register fix-shifts`.
  - Optional project `Config` JSON for `StitchingConfig`.
- It produces:
  - TileConfiguration files and layout plots.
  - Per‑tile per‑channel TIFFs in `stitch--{roi}[+{codebook}]/ZZ/CC/`.
  - Fused per‑channel TIFFs and a consolidated `fused.zarr` per ROI+codebook.
  - Optional N4 bias fields and corrected mosaics.

These contracts are intended to remain stable across internal refactors of
`fishtools/preprocess/cli_stitch.py`.

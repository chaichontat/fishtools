# `align_prod` (`preprocess spots`) data contract

This document defines the data contracts for the `align_prod.py` pipeline, which performs
spot calling and channel optimization on **registered** images produced by
`preprocess register` (see `docs/cli_register_data_contract.md`).

All commands in this file live under the Click group:

- `preprocess spots ...`

Key public commands implemented here:

- `spots run` – per‑tile spot calling / channel optimization
- `spots batch` – batch execution of `run` over many tiles
- `spots optimize` – mandatory stepwise optimization wrapper for production
- `spots step_optimize` – low‑level optimization over subsets of tiles
- `spots combine` – aggregate deviations into global scaling files
- `spots find-threshold` – derive per‑round magnitude thresholds
- `spots plotall` – generate per‑gene plots from ROI‑level parquet

The `spots` group also attaches other subcommands from separate modules:

- `spots optimize` (from `align_batchoptimize.py`)
- `spots stitch` (from `stitch_spot_prod.py`)
- `spots threshold` (from `cli_spotlook.py`)

Those have their own contracts; this document focuses on what is implemented directly in
`align_prod.py` and its interaction with `preprocess register`.

---

## Workspace and registration inputs

### Registration working directory

Unless otherwise noted, the `path` argument for commands in this module is the **registration
working directory**, typically:

```text
<workspace>/analysis/deconv
```

This directory must contain:

- `registered--{roi}+{codebook}/reg-{idx:04d}.tif` – registered stacks from `preprocess register`.
- `opt_{codebook}[+{roi}]/` – optimization artifacts (created by `spots combine` and
  `spots find-threshold`).
- Optional: `fields+{codebook}/field--{roi}+{codebook}.zarr` – illumination field stores
  created by `preprocess correct-illum export-field`.

`Workspace(path)` is used in several commands to resolve the workspace root; if `path` is
`analysis/deconv`, this root is discovered automatically using the sentinel `*.DONE` file.

### Registered TIFF stacks (inputs from `preprocess register`)

All core commands consume registered stacks written by `cli_register`:

- Path pattern:

  ```text
  <regdir> = <registration_path>/registered--{roi}+{codebook_stem}
  <regdir>/reg-{idx:04d}.tif
  ```

- Shape and dtype:

  ```text
  (Z, C, Y, X), dtype=uint16
  ```

  - `Z`: number of slices after any Z‑collapse applied in registration.
  - `C`: number of bits in the codebook that are present for this ROI.
  - `Y`, `X`: cropped/downsampled spatial dimensions.

- Required TIFF metadata (produced by `cli_register`):

  ```python
  {
      "key": list_of_bit_names,  # e.g. ["1", "2", "3", ...]
      "axes": "ZCYX",
      # additional fields like 'shifts' and 'config' may be present
  }
  ```

  `align_prod` relies on:

  - `metadata["key"]` – used to build a `bit_mapping: dict[str, int]` from bit name to channel index.
  - The `ZCYX` layout – assumed when slicing and feeding into `Starfish` as an `ImageStack`.

If `metadata["key"]` is missing or inconsistent with the codebook, `spots run` will fail.

---

## Codebook JSON data contract (for `align_prod`)

`align_prod` uses the same codebook JSON as `preprocess register`, with some additional
assumptions needed for optimization and blank handling.

### Path and label

- CLI options:
  - `--codebook PATH` (for `run`, `batch`, `optimize`, `step_optimize`, `combine`, `find-threshold`)
  - `--codebook LABEL_OR_PATH` (for `plotall`)
- The **codebook label** is taken from `Path(codebook_path).stem` and used in:
  - Registered directory names: `registered--{roi}+{codebook_stem}`
  - Optimization directory names: `opt_{codebook_stem}[+{roi}]`
  - Decoded output directories: `decoded-{codebook_stem}`

### JSON structure and filtering

Expected JSON structure:

```jsonc
{
  "Gene1": [1, 3, 5],
  "Gene2": [2, 4, 6],
  "Blank1": [7, 8],
  "Blank2": [9, 10]
}
```

Contract enforced by `load_codebook`:

- Top‑level must be a JSON object of `{gene_name: [bit, bit, ...]}`.
- Bit identifiers must be integers (`> 0`).
- Before use:
  - Any genes listed in `exclude` (e.g. `"Malat1-201"`, `"tdTomato-Ai9-extra"`) are dropped.
  - In `simple` mode (`--simple`), only the **first bit** for each gene is kept.
  - Genes whose bits are not present in the registered TIFF metadata (`metadata["key"]`) are
    dropped.
  - If nothing remains after this filtering, a `ValueError` is raised.

Derived artifacts:

- `used_bits`: sorted list of all bit indices that appear in any retained gene.
- `names`: `numpy.ndarray[str]` of gene names in the order used in the codebook.
- `arr_zeroblank`: boolean array of shape `(n_genes, n_used_bits)`:
  - `True` where a gene uses a bit.
  - Zeroed out for genes whose names start with `"Blank"` (used for deviation masking).

The Starfish `Codebook` object created by `load_codebook` has:

- `n_round = 1`
- `n_channel = len(used_bits)`
- Bit patterns matching the filtered JSON.

---

## Illumination field correction (TCYX stores)

Several commands support `--field-correct`, which applies a global illumination field during
spot calling or threshold optimization.

### Field store layout

Field stores are Zarr arrays with `axes="TCYX"` produced by:

```text
preprocess correct-illum export-field <model.npz> \
    --what both --downsample 1 --output <workspace>/analysis/deconv/fields+{codebook}/field--{roi}+{codebook}.zarr
```

Contract for the field store:

- Path:

  ```text
  <workspace>/analysis/deconv/fields+{codebook_label}/field--{roi}+{codebook_label}.zarr
  ```

- Shape:

  ```text
  (T, C, Y_ds, X_ds)
  ```

  with `T = 2` (for `low` and `range` planes) and the same number of channels `C` as the registered
  stack (or compatible via channel remapping).

- Attributes:

  - `axes = "TCYX"` (required; anything else raises an error).
  - `t_labels = ["low", "range"]` (order matters; used to identify `T` indices).
  - `model_meta` (dict) containing:
    - `x0`, `y0` (float): tile origin in downsampled coordinates.
    - `downsample` (int): spatial downsample factor relative to registered tiles.
    - `channels` or `attrs["channel_names"]`: list of channel labels (strings) matching
      the registered TIFF `key` metadata when possible.

Field correction workflow in `make_fetcher`:

- Uses the registered tile origin from the tile configuration and any local XY offset from
  quadrant slicing to extract the relevant `(low, range)` planes via `_slice_field_ds_for_tile`.
- Aligns `low`/`range` channels with the selected image channels via `c_index_map`.
- Applies per‑pixel correction:

  ```text
  corrected = max(raw - low, 0) * range
  ```

  and normalizes to `[0, 1]` float32.

If no compatible field store is found, `--field-correct` fails with a `click.ClickException`.

---
## Relationship to registration, stitching, and segmentation

`align_prod.py` sits between registration and higher‑level stitching/analysis steps.

- **Upstream**:
  - Consumes registered stacks created by `preprocess register` under
    `analysis/deconv/registered--{roi}+{codebook}/reg-*.tif` (see
    `docs/cli_register_data_contract.md`).
  - Assumes the `key` metadata encodes the same bit identifiers that appear in the codebook JSON.
  - Optionally uses the same TCYX illumination fields as `preprocess stitch fuse` via the
    `--field-correct` flag.

- **Parallel image path (`preprocess stitch`)**:
  - While `align_prod` operates in the **channel/bit space** of registered tiles to decode spots,
    `preprocess stitch` operates on the same registered stacks to create fused mosaics in
    `analysis/deconv/stitch--{roi}+{codebook}/` (see `docs/cli_stitch_data_contract.md`).
  - Both use the shared `Workspace` layout and TileConfiguration metadata so that decoded spots
    and fused images live in a consistent coordinate system.

- **Downstream**:
  - Decoded per‑tile pickles (`decoded-{codebook}/reg-*.pkl`) are stitched into per‑ROI parquet
    files (via `preprocess spots stitch`), which `spots plotall` then visualizes.
  - The ROI‑level parquet (`{roi}+{codebook}.parquet`) and fused mosaics feed directly into the
    segmentation branch:
    - `segment overlay spots` and `segment overlay intensity` intersect decoded/thresholded spots
      and fused intensity Zarrs with segmentation masks in
      `analysis/deconv/stitch--{roi}+{seg_codebook}/output_segmentation*.zarr`.
    - `segment export` aggregates these overlays into per‑cell polygons and H5AD matrices.
  - See `docs/cli_segment_data_contract.md` for the detailed contracts governing segmentation,
    overlays, and export.

This shared contract ensures that once registration is complete, both intensity pipelines
(`preprocess stitch`), spot pipelines (`preprocess spots`), and segmentation pipelines
(`segment`) can safely compose over the same workspace without re‑interpreting bit/channel
semantics.

---

## Command: `spots run`

### CLI signature

```text
preprocess spots run PATH --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH` (required): path to a **single registered TIFF**:

  ```text
  PATH = <registration_path>/registered--{roi}+{codebook_stem}/reg-{idx:04d}.tif
  ```

Options (subset):

- `--global-scale PATH`
  - Required for production decoding (when `--highpass-only` and `--calc-deviations` are both
    false).
  - Path to `global_scale.txt` created by `spots combine`.
- `--round N`
  - Required when `--calc-deviations` is set (optimization mode).
- `--calc-deviations`
  - Enable channel optimization mode. Writes JSON deviation records and per‑tile pickles into
    `opt_{codebook}[+{roi}]`.
- `--highpass-only`
  - Produce and save highpass TIFFs for threshold selection instead of decoding spots.
- `--split N`
  - Quadrant index (`0`–`3`) for sub‑tile processing; affects slicing and output file naming.
- `--simple`
  - Use a simple decoding mode with a different Starfish pipeline.
- `--roi ROI`
  - Optional ROI tag; used only in optimization output paths (`opt_{codebook}+{roi}`).
- `--blank NAME`
  - Optional codebook/round label for blanks (used in commented‑out blank subtraction code).
- `--config JSON_PATH`
  - Optional project config. Must be parseable into `Config` via `load_config`; only the
    `.spot_decode` section is used here.
- `--field-correct/--no-field-correct`
  - Enable or disable illumination field correction.

### Inputs

1. **Registered TIFF stack**

   - As described in “Registered TIFF stacks” above.

2. **Global scale and minima** (production decoding only)

   - Required unless `--highpass-only` or `--calc-deviations` is set:

     ```text
     <opt_dir>/global_scale.txt
     <opt_dir>/global_min.txt
     ```

     where:

     ```text
     opt_dir = <registration_path>/opt_{codebook_stem}[+{roi}]
     ```

   - Written by `spots combine`.
   - `global_scale.txt`: 2D array (rounds × channels) of scaling factors.
   - `global_min.txt`: 1D array of per‑channel minima.

3. **Percentiles JSON** (production decoding only)

   - Required for magnitude‑based filtering:

     ```text
     <opt_dir>/percentiles.json
     ```

   - Written by `spots find-threshold`.

4. **Configuration (optional)**

   - JSON file compatible with `Config` from `fishtools.preprocess.config`.
   - Only `spot_decode` section is consumed here, providing overrides for:
     - `min_intensity`, `max_distance`, `min_area`, `max_area`, `sigma`, etc.

5. **Field store (optional, when `--field-correct`)**

   - TCYX Zarr store as described in the previous section.

### Outputs

Depending on flags, `spots run` produces different artifacts.

1. **Highpass images** (`--highpass-only`)

   - Path:

     ```text
     <regdir>/_highpassed/reg-{idx:04d}_{codebook_stem}.hp.tif
     ```

   - Shape and dtype:

     ```text
     (Z, C, Y_hp, X_hp), dtype=float32
     ```

     - Derived from the Starfish `ImageStack` after Gaussian high‑pass filtering and optional
       slicing/downsampling.
     - Metadata:

       ```python
       {"keys": img_keys}  # same bit labels as the input TIFF
       ```

   - Used by `spots find-threshold` to estimate per‑round magnitude percentiles.

2. **Optimization deviation and pickle files** (`--calc-deviations`)

   - JSON deviations:

     ```text
     <opt_dir>/{reg_stem}--{roi_token}.json
     ```

     where:

     - `opt_dir = <registration_path>/opt_{codebook_stem}[+{roi}]`
     - `reg_stem = Path(PATH).stem` (e.g. `reg-0001`)
     - `roi_token = PATH.parent.name.split("--")[1]` (e.g. `1_9_17--cortex`)

   - Schema (validated by `Deviations`):

     ```jsonc
     [
       {
         "initial_scale": [float, ...],
         "round_num": 0,
         "mins": [float, ...]
       },
       {
         "n": 12345,
         "deviation": [float, ...],
         "percent_blanks": 0.12,
         "round_num": 1
       },
       ...
     ]
     ```

     - Round `0` entry is written during the initial optimization pass.
     - Subsequent rounds append `Deviation` entries for the same JSON.

   - Per‑tile pickle:

     ```text
     <opt_dir>/{reg_stem}--{roi_token}_opt{round_num:02d}.pkl
     ```

     Contents (pickled Python tuple):

     ```python
     (
         decoded_spots: DecodedIntensityTable,
         morph: list[dict[str, Any]],  # region properties, may be empty
         meta: {
             "fishtools_commit": git_hash(),
             "config": SpotDecodeConfig.model_dump(),
         },
     )
     ```

3. **Decoded spot pickles** (production decoding)

   - Path:

     ```text
     <regdir>/decoded-{codebook_stem}/reg-{idx:04d}[-{split}].pkl
     ```

     where `split` is `0`–`3` when quadrant splitting is used.

   - Contents: same `(decoded_spots, morph, meta)` tuple as above.
   - These files are consumed by `stitch_spot_prod.stitch` to build ROI‑level parquet with
     global coordinates.

### Error conditions

`spots run` fails when:

- `--calc-deviations` is set but `--round` is missing.
- `--highpass-only` and `--calc-deviations` are both false and `--global-scale` is missing.
- The input TIFF is missing `metadata["key"]` or is incompatible with the codebook.
- The global scale/min files or percentiles JSON are missing (production mode).

---

## Command: `spots step_optimize`

### CLI signature

```text
preprocess spots step_optimize PATH ROI --round N --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH`: registration working directory (typically `<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` to sample across all ROIs.

Options:

- `--round N` (required)
- `--codebook CODEBOOK_PATH` (required)
- `--batch-size, -n` (default: `40`)
- `--threads, -t` (default: `8`)
- `--overwrite`
- `--split`
- `--blank NAME`
- `--field-correct/--no-field-correct`
- `--config JSON_CONFIG`

### Inputs

1. Registered TIFFs:

   - Sampled from:

     ```text
     PATH/registered--{roi}+{codebook_stem}/reg*.tif
     ```

   - Highpass variants (`*.hp.tif`) are excluded from sampling.

2. Percentiles JSON for later rounds:

   - For `round_num > 0`, requires:

     ```text
     PATH/opt_{codebook_stem}[+{roi}]/percentiles.json
     ```

### Outputs

`step_optimize` does not itself write new data; it:

- Copies the codebook into `PATH/codebooks/` for reproducibility.
- Optionally verifies that field stores exist when `--field-correct` is enabled.
- Dispatches a series of `spots run` invocations with:
  - `--calc-deviations`
  - `--global-scale PATH/opt_{codebook_stem}[+{roi}]/global_scale.txt`

The actual deviation JSON and pickles are written by `spots run` as described above.

---

## Command: `spots combine`

### CLI signature

```text
preprocess spots combine PATH ROI --codebook CODEBOOK [--batch-size N --round N]
```

Arguments:

- `PATH`: registration working directory (typically `<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` (used in optimization directory naming).

Options:

- `--codebook CODEBOOK_PATH` (required)
- `--batch-size, -n` (default: `50`)
- `--round N` (required)

### Inputs

1. Optimization deviation JSON files:

   - As written by `spots run --calc-deviations` via `step_optimize`:

     ```text
     PATH/opt_{codebook_stem}[+{roi}]/{reg_stem}--{roi_token}.json
     ```

2. For `round_num > 0`, existing global scale/min files:

   ```text
   PATH/opt_{codebook_stem}[+{roi}]/global_scale.txt
   PATH/opt_{codebook_stem}[+{roi}]/global_min.txt
   ```

### Outputs

Within:

```text
opt_dir = PATH/opt_{codebook_stem}[+{roi}]
```

- Round `0`:
  - `global_scale.txt` – single row vector of per‑bit scale factors.
  - `global_min.txt` – per‑bit minima (averaged across tiles).
- Round `>0`:
  - `global_scale.txt` – appended rows, where row `round_num` is derived from
    deviations at that round.
  - `mse.txt` – tab‑separated text with lines:

    ```text
    {round_num:02d}\t{cv:0.4f}
    ```

    where `cv` is the coefficient of variation of deviation across bits.

These files are later consumed by `spots find-threshold` and `spots run` in production mode.
In practice you should not call `step_optimize` and `combine` directly; use
`spots optimize`, which orchestrates these commands and `spots find-threshold`
round‑by‑round (see below).

---

## Command: `spots find-threshold`

### CLI signature

```text
preprocess spots find-threshold PATH ROI --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH`: registration working directory (typically `<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` (only one ROI is supported per invocation).

Options:

- `--codebook CODEBOOK_PATH` (required)
- `--overwrite`
- `--round N` (default: `0`)
- `--blank NAME`
- `--config JSON_CONFIG`
- `--field-correct/--no-field-correct`

### Inputs

1. Registered stacks:

   - Same as for `spots run`:

     ```text
     PATH/registered--{roi}+{codebook_stem}/reg*.tif
     ```

2. Optimization outputs:

   - Requires:

     ```text
     PATH/opt_{codebook_stem}[+{roi}]/global_scale.txt
     PATH/opt_{codebook_stem}[+{roi}]/global_min.txt
     ```

3. Optional config (`Config` JSON) to load `spot_decode.clip_percentile`.

### Outputs

1. **Highpass images** (if not already present):

   - Under:

     ```text
     PATH/registered--{roi}+{codebook_stem}/_highpassed/reg-{idx:04d}_{codebook_stem}.hp.tif
     ```

   - Produced via `spots run --highpass-only`.

2. **Percentiles JSON**:

   - Path:

     ```text
     PATH/opt_{codebook_stem}[+{roi}]/percentiles.json
     ```

   - Content:

     ```jsonc
     [
       {
         "registered--{roi}+{codebook_stem}-reg-0001_{codebook_stem}.hp.tif": 123.45,
         "registered--{roi}+{codebook_stem}-reg-0002_{codebook_stem}.hp.tif": 130.67,
         ...
       },
       ...
     ]
     ```

   - Each dict in the list corresponds to one round:
     - Round `0`: first element.
     - Round `n`: `(n)`‑th element.
   - Values are per‑image percentiles of the channel norm after scaling.
   - The mean of the current round’s values is used in `spots run` as the magnitude threshold
     for `Filter.ZeroByChannelMagnitude`.

In production workflows `spots find-threshold` is invoked automatically by
`spots optimize` and does not usually need to be called directly.

---

## Command: `spots optimize`

`spots optimize` is the **recommended and mandatory** entry point for stepwise
optimization. It combines `spots step_optimize`, `spots combine`, and
`spots find-threshold` into a single command, iterating over optimization
rounds until convergence (or the configured maximum number of rounds).

### CLI signature

```text
preprocess spots optimize PATH ROI --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH`: registration working directory (typically `<workspace>/analysis/deconv`).
- `ROI`: ROI identifier or `"*"` to optimize across all ROIs.

Options (subset):

- `--codebook CODEBOOK_PATH` (required)
- `--rounds N` (default: `10`)
  - Maximum number of optimization rounds to perform.
- `--threads N` (default: `10`)
- `--batch-size N` (default: `50`)
- `--blank NAME`
- `--threshold FLOAT` (default: `0.008`)
  - CV threshold for early stopping based on `mse.txt`.
- `--config JSON_CONFIG`
- `--field-correct/--no-field-correct`
  - Apply TCYX illumination fields for all subcommands.

### Inputs

- Registered TIFFs under `PATH/registered--{roi}+{codebook_stem}/reg*.tif`.
- Optional TCYX field stores under
  `PATH/fields+{codebook_stem}/field--{roi}+{codebook_stem}.zarr` when
  `--field-correct` is enabled.

### Outputs

Within:

```text
opt_dir = PATH/opt_{codebook_stem}[+{roi}]
```

`spots optimize` ensures, round‑by‑round, that:

- `global_scale.txt` and `global_min.txt` are created and extended as in
  `spots combine`.
- `mse.txt` is updated with per‑round coefficients of variation.
- `percentiles.json` is updated as in `spots find-threshold`.

Optimization rounds stop early when the CV from `mse.txt` drops below the
configured `--threshold`.

In production decoding, running `preprocess spots optimize` before
`preprocess spots batch` is mandatory so that `spots run` sees consistent
global scale/min/percentile files for all rounds.

---

## Command: `spots batch`

### CLI signature

```text
preprocess spots batch PATH ROI --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH`: registration working directory (typically `<workspace>/analysis/deconv`).
- `ROI`: specific ROI name or `"*"` (all ROIs).

Options (subset):

- `--codebook CODEBOOK_PATH` (required)
- `--threads, -t` (default: `13`)
- `--overwrite`
- `--simple`
- `--since DURATION`
  - Only process files modified in the last `DURATION` (e.g. `"30m"`, `"2h"`, `"1d"`).
- `--delete-corrupted`
  - Delete registered TIFFs that fail `Workspace.ensure_tiff_readable`.
- `--split`
  - Kept for backward compat; always treated as `True` (quadrant mode).
- `--local-opt`
  - Use a global scale file at `PATH/opt_{codebook_stem}/global_scale.txt` instead of
    per‑ROI `opt_{codebook_stem}+{roi}` (for local experimentation).
- `--blank NAME`
- `--config JSON_CONFIG`
- `--stagger`, `--stagger-jitter`
  - Control scheduling jitter for worker processes.
- `--field-correct/--no-field-correct`

### Inputs

1. Registered stacks:

   - Discovered via:

     ```python
     file_map, _ = Workspace(PATH).registered_file_map(codebook_stem, rois=roi_filter)
     ```

   - Must be in:

     ```text
     <workspace>/analysis/deconv/registered--{roi}+{codebook_stem}/reg-{idx:04d}.tif
     ```

2. Global scale files:

   - In standard pipeline:

     ```text
     PATH/opt_{codebook_stem}/global_scale.txt
     ```

   - Typically created by `spots combine` with `PATH = <workspace>/analysis/deconv`.

3. Optional TCYX field stores (when `--field-correct` is used).

### Outputs

`spots batch` orchestrates many `spots run` invocations and therefore produces:

- Highpass images (when `--highpass-only` is propagated, e.g. via `find-threshold`).
- Optimization JSON/pickles (when `--calc-deviations` is used in underlying commands).
- Decoded spot pickles in:

  ```text
  <registration_path>/registered--{roi}+{codebook_stem}/decoded-{codebook_stem}/reg-{idx:04d}-{split}.pkl
  ```

The command does not itself write new summary artifacts beyond logging.

---

## Command: `spots plotall`

### CLI signature

```text
preprocess spots plotall WORKSPACE [ROI] --codebook LABEL_OR_PATH [OPTIONS]
```

Arguments:

- `WORKSPACE`: path to the workspace root (directory containing `analysis/`).
- `ROI` (optional): specific ROI, `"*"`, or omitted (all).

Options (subset):

- `--codebook LABEL_OR_PATH` (required)
  - If a path is provided, its stem is used as the codebook label.
- `--threads, -t` (default: `8`)
- `--only-blank`
- `--dark`
- `--outdir`
- `--overwrite`
- `--max-per-plot`
- `--figure-size`
- `--cmap`

### Inputs

For each ROI, `plotall` looks for a ROI‑level spots parquet:

```text
<workspace>/analysis/output/{roi}+{codebook_label_sanitized}.parquet
<workspace>/analysis/output/{roi}+{codebook_label}.parquet
<workspace>/analysis/deconv/{roi}+{codebook_label_sanitized}.parquet
<workspace>/analysis/deconv/{roi}+{codebook_label}.parquet
```

where `codebook_label_sanitized = codebook_label.replace("-", "_").replace(" ", "_")`.

The parquet is expected to contain at least:

- `target` – gene/target name per spot.
- `x`, `y` – global mosaic coordinates.

These parquets are typically produced by `preprocess spots stitch` (`stitch_spot_prod.py`) after
decoding with `spots batch`.

### Outputs

Per‑ROI PNG figures saved under:

```text
OUTDIR = <workspace>/analysis/output/plots  # default
plotall--{roi}+{codebook_label_sanitized}[--blank][--dark].png
plotall--{roi}+{codebook_label_sanitized}[--blank][--dark].{part}.png  # when split
```

Each figure shows per‑gene density/hexbin plots for all or a subset of targets, depending on
`--only-blank` and `--max-per-plot`.

---

## Summary

`align_prod.py` forms the core of the production spot‑calling pipeline:

- **Inputs**:
  - Registered TIFFs (`registered--{roi}+{codebook}/reg-*.tif`) produced by `preprocess register`.
  - Codebook JSON files mapping genes to bit indices.
  - Optional TCYX illumination field stores and project configuration JSONs.
- **Intermediate outputs**:
  - Per‑tile decoded pickles (`decoded-{codebook}/reg-*.pkl`).
  - Highpass images (`_highpassed/*.hp.tif`).
  - Optimization artifacts (`opt_{codebook}[+{roi}]` containing deviations, global scale/min,
    and per‑round percentiles).
- **Downstream outputs**:
  - ROI‑level parquet (`{roi}+{codebook}.parquet`) and plots produced via `stitch` and
    `plotall`.

These contracts are designed to remain stable even as internal implementation details of
`align_prod.py` evolve.

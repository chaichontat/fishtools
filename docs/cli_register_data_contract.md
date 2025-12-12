# `preprocess register` (`cli_register`) data contract

This document defines the data contracts for the legacy Click‑based registration CLI implemented in
`fishtools/preprocess/cli_register.py`. It covers all three public commands:

- `preprocess register run`
- `preprocess register batch`
- `preprocess register fix-shifts`

The goal is to make all filesystem layouts, file formats, and metadata expectations explicit so that
other tools can safely interoperate with these CLIs.

---

## Workspace and path conventions

### Workspace root

All commands assume a valid fishtools workspace, as detected by `Workspace`:

- A workspace root contains at least one `*.DONE` sentinel file (e.g. `workspace.DONE`).
- Processed data lives under `<workspace>/analysis/deconv/`.

The `preprocess register` commands take different `path` arguments:

- `run` / `batch`: `path` must point inside the **deconvolved data tree**
  (typically `<workspace>/analysis/deconv`).
- `fix-shifts`: `path` must point to the **workspace root** (directories like `{round}--{roi}` live
  directly under this path).

`setup_cli_logging` resolves the true workspace root from any of these paths and writes logs under:

- `<workspace>/analysis/logs/<component>.log`

### Round / ROI layout

For registration, rounds and ROIs must satisfy:

- Deconvolved tiles for a round/ROI live at
  `<workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif`
- `{round}` is an arbitrary token (often `a_b_c`) without the forbidden prefixes below.
- `{roi}` is a free‑form identifier (no `/`) such as `cortex`, `roiA`, `4`, etc.
- Tile indices `idx` are 0‑based integers formatted with 4 digits (`0000`, `0001`, …).

The registration CLIs ignore any directory whose **round prefix** starts with one of:

```text
10x, registered, shifts, fids
```

and any prefixes configured in `Config.exclude`.

### Required deconvolution scaling (per round)

Unless input tiles are explicitly marked as prenormalized (see below), each imaging round must have
a global deconvolution scaling file:

- Path: `<workspace>/analysis/deconv_scaling/{round}.txt`
- Type: plain text, loadable via `np.loadtxt`, reshapeable to `(2, n_channels)`.

If `prenormalized` is **not** set in the TIFF metadata and this file is missing, registration fails
with a `ValueError`.

---

## Codebook JSON data contract

Both `run` and `batch` require a `--codebook` argument pointing to a JSON file. The CLI uses only
the **union of bit identifiers** across all entries.

### Path and label

- CLI argument: `--codebook PATH`
- The **codebook label** is taken from `Path(codebook).stem` and is used in output directory names:
  - `registered--{roi}+{codebook_stem}`
  - `shifts--{roi}+{codebook_stem}`

### JSON structure

Expected shape (keys and values are representative; only the union of bit identifiers is used):

```jsonc
{
  "gene_or_target_1": [1, 2, 3],
  "gene_or_target_2": [4, "5"]
}
```

Contract:

- Top‑level value must be a JSON object (`dict` in Python).
- Each value must be an **array** of bit identifiers; elements must be convertible to `str`.
- The **codebook bits** used by the CLI are:

  ```python
  codebook_bits = {str(bit) for bit in chain.from_iterable(cb.values())}
  ```

- For a given ROI, every bit in `codebook_bits` must correspond to at least one deconvolved
  directory whose name starts with that bit:

  ```text
  <bit1>_<bit2>_..._<bitN>--{roi}
  ```

If any bits in `codebook_bits` are missing for a ROI, `run`/`batch` aborts with:

```text
Missing codebook bits for ROI {roi}: {missing_bits} (available: {available_bits})
```

---

## Input TIFF contract (deconvolved tiles)

All registration logic ultimately consumes per‑tile TIFF stacks via `Image.from_file(...)`.

### Location and naming

For a given `{round}`, `{roi}`, and `idx`:

- Input path: `<workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif`

### Shape and dtype

- Expected shape (after loading with `tifffile.TiffFile(...).asarray()`):

  ```text
  (Z_total, 2048, 2048)
  ```

  where:

  - `Z_total > n_fids`
  - `Z_total - n_fids` is divisible by `n_channels`

- `n_fids` is the number of fiducial frames at the end of the stack:
  - In `run` / `batch`, `n_fids` is taken from `RegisterConfig.fiducial.n_fids`
    (default `2` in the CLI).
  - In `fix-shifts`, `n_fids` comes from the `--n-fids` option (default `2`).

- Internally, the no‑fiducial portion is reshaped to:

  ```python
  nofid = img[:-n_fids].reshape(-1, n_channels, 2048, 2048)  # (Z, C, Y, X)
  ```

### Required TIFF metadata

The following metadata is required per tile:

- `waveform`
  - Source: `tif.shaped_metadata[0]["waveform"]` (dict or JSON string), or
  - Fallback: TOML file next to the TIFF: `<round>.toml`.
  - Must contain entries for each channel in:

    ```python
    ["ilm405", "ilm488", "ilm560", "ilm650", "ilm750"]
    ```

  - For each such key used in registration, the CLI expects:
    - `waveform[key]["sequence"]`: list of integers; summed to determine whether the channel is active.
    - `waveform[key]["power"]`: numeric power used to derive per‑bit channel labels.

- Optional: `waveform["params"]["powers"]`
  - When present, this dictionary overrides the per‑channel powers calculated from `ilm*` entries.

- Optional: `prenormalized` (bool)
  - If `True`, deconvolution scaling is **not** applied and
    `analysis/deconv_scaling/{round}.txt` is not required.
  - If `False` or absent, deconvolution scaling **must** exist (see above).

Other metadata (e.g. `axes`) is not required for registration but may be present.

---

## Shared semantics: bits, channels, and shifts

### Bits and channel mapping

Per tile:

- The tile stem `{round}` before `--` encodes bit names separated by `_`:

  ```text
  {bit1}_{bit2}_..._{bitN}--{roi}
  ```

- The `waveform` powers determine which physical channels are present, and they must match the
  bit count:

  ```python
  bits = name.split("_")
  powers = {channel_wavelength: power_value, ...}  # derived from waveform
  assert len(powers) == len(bits)
  ```

- Channel mapping used in registration:

  ```python
  channels: dict[str, str]  # bit -> wavelength (e.g. "560", "650")
  ```

- Bits are split out into individual `(Z, Y, X)` volumes and re‑aligned.

### Fiducials and shifts

For each round participating in registration:

- Fiducial images are built from the last `n_fids` frames and optionally preprocessed:

  - `Image.fid_raw`: raw fiducial image (`float32`).
  - `Image.fid`: LoG‑filtered fiducial image used for spot‑based alignment.

- Shifts are computed via `align_fiducials`:

  - Output per round: `shift = np.array([dx, dy], dtype=float)` in **pixel units**.
  - Positive `dx`: shift to the right along X.
    Positive `dy`: shift down along Y.

- Internal storage and output:

  - Shifts are stored in memory as `[dx, dy]`.
  - When applying to images, `scipy.ndimage.shift` is called with `[dy, dx]`.

These shifts are exposed both as standalone JSON files (see below) and as TIFF metadata in the
registered stacks.

---
## Downstream consumers and pipeline integration

The outputs of `preprocess register` are designed to be consumed by several later stages in the
pipeline.

### Stitching (`preprocess stitch`)

- `preprocess stitch register` and `preprocess stitch fuse` read the registered stacks in:

  ```text
  <workspace>/analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
  ```

- They rely on:
  - The `ZCYX` layout and consistent spatial shape across all channels.
  - The `key` metadata (bit/channel labels) to propagate channel names into stitched mosaics and
    fused Zarr volumes.
- The generated per‑channel mosaics and `fused.zarr` live under:

  ```text
  <workspace>/analysis/deconv/stitch--{roi}+{codebook}/
  ```

  and are used for segmentation (`segment` CLI), overlays, and N4 bias‑field correction.

### Spot decoding (`preprocess spots`, `align_prod.py`)

- `preprocess spots run` (in `align_prod.py`) also reads the registered stacks from:

  ```text
  <workspace>/analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
  ```

- It uses:
  - `metadata["key"]` – to build a bit→channel mapping compatible with the codebook JSON.
  - The same bit indices that were used in registration, ensuring that codebook bits and
    registered channels stay aligned.
- Decoded spot pickles are written next to the registered TIFFs, and later stitched into
  ROI‑level parquet files for downstream analysis.

### Coarse shifts and illumination

- `preprocess register fix-shifts` writes `shifts--{roi}/coarse_shifts.json`, which can be used
  by `preprocess stitch fuse --coarse-shifts` to adjust tile positions before ImageJ fusion.
- Optional TCYX illumination fields produced by `preprocess correct-illum export-field` are
  aligned using the same tile origins and ROI names as the registered outputs, so the
  registration and stitching contracts remain consistent.

Together, these contracts ensure that round/ROI naming, bit/channel semantics, and file layouts
are stable across registration (`preprocess register`), stitching (`preprocess stitch`), and
spot decoding (`preprocess spots`).

---

## Command: `preprocess register run`

### CLI signature

```text
preprocess register run PATH IDX --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH` (required): directory under the workspace, typically `<workspace>/analysis/deconv`.
- `IDX` (required): tile index (integer; will be formatted as `{idx:04d}` internally).

Options:

- `--codebook PATH` (required)
  - Path to a codebook JSON file (see “Codebook JSON data contract”).
- `--roi ROI` (default: `"*"`)
  - When `"*"`, automatically discover all ROIs present under `PATH`.
  - Otherwise, process only the specified ROI.
- `--reference, -r ROUND` (default: `"4_12_20"`)
  - Round identifier used as the fiducial reference.
- `--threshold FLOAT` (default: `5.0`)
  - Fiducial spot detection threshold in sigma above median.
- `--fwhm FLOAT` (default: `4.0`)
  - Fiducial spot FWHM in pixels.
- `--overwrite`
  - If present, overwrite any existing `reg-{idx:04d}.tif` outputs.
  - If absent, skip indices for which the registered output already exists.
- `--no-priors`
  - Disable use of priors from existing `shifts--{roi}+{codebook}/shifts-*.json`.
- `--use-fft`
  - Use FFT phase correlation for alignment instead of spot‑based registration.
- `--use-itk`
  - Use SimpleITK translation registration.
- `--anchors PATH`
  - Path to an ImageJ `RoiSet.zip` with anchor points; enables anchor‑ROI‑based alignment.
- `--use-brightest N` (default: `20`)
  - If `>0`, use only the N brightest fiducial spots per image for alignment.
- `--allow-large-drifts` / `--ignore-large-shifts`
  - Allow shifts larger than the configured drift threshold instead of failing (the latter is an alias).
- `--debug`
  - Enable verbose logging and additional debug artifacts.

### Inputs

For each ROI selected:

- Deconvolved tiles:
  - Required: `<workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif` for rounds that
    share bits with either the reference or the codebook bits.
  - The union of all bit tokens across these rounds must cover every bit in `codebook_bits`.
- Deconvolution scaling:
  - Required per round when the tile metadata does **not** set `prenormalized=True`:
    `<workspace>/analysis/deconv_scaling/{round}.txt`.
- Optional BaSiC templates:
  - If present: `<workspace>/analysis/deconv/basic/{round_name}.pkl`; used to construct per‑bit
    BaSiC correction maps (not directly exposed in outputs).

### Outputs

For each ROI and tile index `idx`:

1. **Registered stack**

   - Path:

     ```text
     <workspace>/analysis/deconv/registered--{roi}+{codebook_stem}/reg-{idx:04d}.tif
     ```

   - Shape and dtype:

     ```text
     (Z, C, Y, X), dtype=uint16
     ```

     where:

     - `Z` is the number of slices after applying `registration.slices` (either 1 or the number of
       slice windows).
     - `C` is the number of bits in `codebook_bits` that are present for this ROI.
     - `Y`, `X` are the spatial dimensions after cropping and optional downsampling.

   - TIFF metadata:

     ```python
     metadata = {
         "key": list_of_bit_names,            # list[str], sorted by numeric then lexicographic order
         "axes": "ZCYX",
         "shifts": json.dumps(shifts_dict),   # see “Shifts JSON schema”
         "config": json.dumps(config_dict),   # Config.model_dump() for provenance
     }
     ```

     - `shifts_dict`: mapping from round name to per‑round shift metrics (see below).
     - `config_dict`: full `Config` object used for this run, serialized using `NumpyEncoder`.

2. **Per‑index shifts JSON**

   - Path:

     ```text
     <workspace>/analysis/deconv/shifts--{roi}+{codebook_stem}/shifts-{idx:04d}.json
     ```

   - Schema (logical; exact key set depends on participating rounds):

     ```jsonc
     {
       "round_name_1": {
         "shifts": [dx, dy],
         "residual": float,
         "corr": float,
         "iterations": int | null,
         "final_fwhm": float | null,
         "final_threshold": float | null,
         "n_spots": int | null,
         "mode": "spots" | "fft" | "itk" | null,
         "algorithm": string | null
       },
       "round_name_2": { "...": "..." },
       "reference_round": {
         "shifts": [0.0, 0.0],
         "residual": 0.0,
         "corr": 1.0,
         "iterations": 0,
         "final_fwhm": float | null,
         "final_threshold": float | null,
         "n_spots": int | null,
         "mode": "spots" | "fft" | "itk" | null,
         "algorithm": string | null
       }
     }
     ```

     - `dx`, `dy`: drift in pixels relative to the reference round.
     - `residual`: alignment residual metric.
     - `corr`: correlation coefficient vs the reference fiducial region.
     - `iterations`: number of iterations of the alignment algorithm executed for this round (0 for the reference); for spot-based alignment this is the drift refinement loop count, for ITK this is the optimizer iteration count, and for FFT it is 0.
     - `final_fwhm`: effective FWHM used for fiducial detection in the successful spot-based attempt, after any automatic adjustments (null for FFT/ITK).
     - `final_threshold`: effective detection threshold (σ above median) used in the successful spot-based attempt (null for FFT/ITK).
     - `n_spots`: number of fiducial spots used for alignment in the final spot-based attempt (0 or null for FFT/ITK).
     - `mode`: alignment backend used for this round: `"spots"` (fiducial-spot matcher), `"fft"` (phase cross-correlation), `"itk"` (SimpleITK translation), or `null` when not applicable (e.g. anchor-ROI).
     - `algorithm`: human-readable algorithm name when applicable. For ITK alignment this is `"OnePlusOneEvo"` (SimpleITK's OnePlusOneEvolutionary optimizer); for other modes this is typically `null`.

     For FFT‑ and ITK‑based alignment modes, and for anchor‑ROI–based alignment, the spot-related
     diagnostic fields may be `null` when no fiducial-spot refinement is performed.

3. **Fiducial stacks (always)**

   - Path:

     ```text
     <workspace>/analysis/deconv/registered--{roi}+{codebook_stem}/_fids/_fids-{idx:04d}.tif
     ```

   - Data:
     - Stack of fiducial images for all participating rounds, ordered by round name.
   - Metadata:

     ```python
     {
       "axes": "CYX",
       "key": ordered_round_names,  # list[str]
     }
     ```

4. **Debug fiducial artifacts (when `--debug` is set)**

   - Directory: `<workspace>/analysis/deconv/fids_debug/{roi}/`
   - Files per index:
     - Raw fiducials:

       ```text
       {roi}-{idx:04d}.tif   # axes: CYX, key: round names
       ```

     - Shifted fiducials:

       ```text
       {roi}-shifted-{idx:04d}.tif
       ```

     - RGB overlays (`.png`) comparing each round to the reference:

       ```text
       {roi}-{idx:04d}-{round}.png
       ```

### Error conditions

`run` fails (non‑zero exit) in at least these situations:

- No tiles found for the requested `idx` in any round/ROI.
- Required deconvolution scaling file missing for a non‑prenormalized round.
- Any codebook bit is not available in the ROI.
- Registered images for different bits end up with different shapes.

---

## Command: `preprocess register batch`

### CLI signature

```text
preprocess register batch PATH ROI --codebook CODEBOOK [OPTIONS]
```

Arguments:

- `PATH` (required): directory under the workspace, typically `<workspace>/analysis/deconv`.
- `ROI` (required): either a specific ROI or a wildcard:
  - `"*"` or `"all"`: process all ROIs discovered by `Workspace.rois`.
  - Any other string: process only that ROI.

Options:

- `--codebook PATH` (required)
  - Path to a codebook JSON file.
- `--ref ROUND` (optional)
  - Reference round identifier. If omitted, defaults are:
    - `"2_10_18"` if present in `Workspace.rounds`, otherwise
    - `"7_15_23"` if present, otherwise
    - error (`ValueError`).
- `--fwhm FLOAT` (default: `4`)
- `--threshold FLOAT` (default: `6`)
- `--threads INT` (default: `15`)
  - Maximum number of worker threads used for parallel tile registration.
- `--overwrite`
- `--debug`
- `--verify`
  - After registration, verify that all `reg-*.tif` outputs are readable and shape‑consistent.
  - Any failures trigger per‑index reruns with `--overwrite`, followed by re‑verification.
- `--use-fft`, `--use-itk`, `--use-brightest N`, `--allow-large-drifts`/`--ignore-large-shifts`
  - Forwarded to the underlying `run` invocation exactly as specified.

### Inputs

Shared with `run`, with the following batch‑specific expectations:

- Reference tiles must exist for each ROI:

  ```text
  PATH/{ref}--{roi}/{ref}-{idx:04d}.tif
  ```

- For each ROI, the set of tile indices to process is:

  ```python
  idxs = [
      int(name.stem.split("-")[1])
      for name in PATH.rglob(f"{ref}--{roi}/{ref}*.tif")
      if overwrite
      or not (PATH / f"registered--{roi}+{codebook_stem}/reg-{idx:04d}.tif").exists()
  ]
  ```

- If `idxs` is empty and `--verify` is not set, the ROI is skipped with a warning.

### Codebook handling in `batch`

Before any tile work, `batch` materializes the codebook into the workspace for reproducibility:

- Input: `--codebook PATH`
- Operation:

  ```python
  workspace = Workspace(PATH)  # resolved from PATH
  destination = workspace.deconved / "codebooks" / source.name
  shutil.copy2(source, destination)
  ```

- All child `run` invocations use the copied codebook path.

### Outputs

`batch` orchestrates calls to `preprocess register run` and therefore produces the same artifacts as
`run` for each ROI / tile index, plus:

1. **Copied codebook**

   - Path:

     ```text
     <workspace>/analysis/deconv/codebooks/{original_codebook_filename}
     ```

   - Contents: byte‑for‑byte copy of the original codebook.

2. **Optional verification and recovery**

   When `--verify` is set:

   - For each ROI, `batch` establishes a baseline registered shape from the first readable
     `reg-{idx:04d}.tif`.
   - All indices for the ROI are checked:
     - Missing files, read errors, or shape mismatches are recorded as failures.
   - Failed indices are re‑run via:

     ```text
        preprocess register run PATH IDX --codebook CODEBOOK --fwhm ... --threshold ... \
        --reference REF --roi ROI --overwrite [--allow-large-drifts|--ignore-large-shifts]
     ```

   - Each rerun is re‑verified; persistent mismatches are logged as errors.

### Error conditions

`batch` fails (non‑zero exit) in at least these situations:

- No reference tiles are found for a given `ref`/`roi` pairing.
- No suitable reference round exists when `--ref` is omitted.
- Underlying `run` invocations fail for reasons described in the `run` section.

---

## Command: `preprocess register fix-shifts`

### CLI signature

```text
preprocess register fix-shifts PATH --roi ROI --rounds "ROUND1,ROUND2,..." [OPTIONS]
```

Arguments:

- `PATH` (required): **workspace root** (directories like `{round}--{roi}` live directly here).

Options:

- `--roi, -o ROI` (required)
  - ROI to process (must match the `--roi` token in directory names).
- `--reference, -r ROUND` (default: `"2_10_18"`)
  - Reference round name.
- `--rounds STRING` (required)
  - Comma‑separated list of round names to measure against the reference
    (e.g. `"1_9_17,3_11_19"`).
- `--use-fft/--use-spots` (default: `--use-spots`)
  - Choose FFT‑based or spot‑based fiducial alignment.
- `--n-fids INT` (default: `2`)
- `--threshold FLOAT` (default: `5.0`)
- `--fwhm FLOAT` (default: `4.0`)
- `--prior "dx,dy"`
  - Optional coarse prior shift for a single target round, specified as `"dx,dy"`.
  - Example: `"150,-30"` for a +150 px, −30 px prior shift.
- `--debug`

### Inputs

For each ROI:

- Reference tiles:

  ```text
  PATH/{reference}--{roi}/{reference}-{idx:04d}.tif
  ```

- Target tiles:

  ```text
  PATH/{round}--{roi}/{round}-{idx:04d}.tif
  ```

  for each `round` in `--rounds`. Missing tiles for a round/idx pair are logged and skipped.

Tile TIFFs must satisfy the same input TIFF contract described earlier (`Image.from_file`).

### Outputs

All results are summarized in a single JSON file per ROI:

- Directory:

  ```text
  PATH/shifts--{roi}
  ```

- File:

  ```text
  PATH/shifts--{roi}/coarse_shifts.json
  ```

- Schema:

  ```jsonc
  {
    "reference": "2_10_18",
    "use_fft": false,
    "tiles": {
      "0001": {
        "1_9_17": {
          "dx": 12.3,
          "dy": -4.5,
          "magnitude": 13.1,
          "residual": 0.12
        },
        "3_11_19": { "...": "..." }
      },
      "0002": { "...": "..." }
    },
    "prior": {          // present only if --prior was provided
      "dx": 150.0,
      "dy": -30.0
    }
  }
  ```

Interpretation:

- `dx`, `dy`: total shifts in pixels from the target round to the reference (including any prior).
- `magnitude`: Euclidean norm `sqrt(dx^2 + dy^2)`.
- `residual`: alignment residual metric reported by the fiducial alignment.

This file is intended as an **offline diagnostic** and as a source of informed priors that can be
manually translated into `RegisterConfig.fiducial.priors` before running `preprocess register batch`.

### Error conditions

`fix-shifts` fails (non‑zero exit) when:

- The reference or any requested round directory does not exist for the specified ROI.
- The reference directory contains no tiles.

---

## Summary

At a high level:

- Inputs:
  - Deconvolved TIFF stacks in `<workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif`,
    with appropriate fiducial frames and waveform metadata.
  - Per‑round deconvolution scaling files in `analysis/deconv_scaling/` (unless prenormalized).
  - A codebook JSON whose union of bits matches the set of bit tokens present in the
    `{round}--{roi}` directories.
- Outputs:
  - Registered stacks in `registered--{roi}+{codebook}/reg-{idx:04d}.tif` with `ZCYX` axes and
    rich metadata.
  - Per‑index shifts JSON files under `shifts--{roi}+{codebook}/`.
  - Optional coarse shift summaries for manual inspection under `shifts--{roi}/coarse_shifts.json`.
  - Consolidated logs under `<workspace>/analysis/logs/`.

These contracts are intended to remain stable across internal refactors of
`fishtools/preprocess/cli_register.py`.

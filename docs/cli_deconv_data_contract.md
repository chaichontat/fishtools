# `preprocess deconvnew` (`cli_deconv`) data contract

This document defines the data contracts for the modern deconvolution CLI in
`fishtools/preprocess/cli_deconv.py`. It focuses on the `deconvnew` Click group and its
subcommands:

- `preprocess deconvnew precompute`
- `preprocess deconvnew quantize`
- `preprocess deconvnew run`
- `preprocess deconvnew batch`
- `preprocess deconvnew easy`

The legacy CLI (`cli_deconv_old.py`) is kept for backward compatibility and is not documented here.

---

## Position in the end‑to‑end pipeline

`deconvnew` is the **first major processing stage** after raw acquisition:

1. Raw tiles per round/ROI live under:

   ```text
   <workspace>/{round}--{roi}/{round}-{idx:04d}.tif
   ```

2. `preprocess deconvnew run` (or `batch`) performs 3D deconvolution:

   - Produces float32 staging outputs under:

     ```text
     <workspace>/analysis/deconv32/{round}--{roi}/{round}-{idx:04d}.tif
     ```

   - Produces (optionally) U16 deliverables under:

     ```text
     <workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif
     ```

3. `preprocess deconvnew precompute` and `quantize`:

   - Compute global scaling statistics and generate quantized U16 outputs used for delivery.

4. `preprocess register` consumes the **U16 deconvolved tiles** from `analysis/deconv` and writes:

   ```text
   analysis/deconv/registered--{roi}+{codebook}/reg-{idx:04d}.tif
   ```

5. `preprocess stitch` and `preprocess spots` then work on these registered stacks (see
   `docs/cli_register_data_contract.md`, `docs/cli_stitch_data_contract.md`,
   `docs/align_prod_data_contract.md`).

Global scaling files written by `deconvnew` (`analysis/deconv_scaling/{round}.txt`) are also
consumed by `preprocess register` to apply consistent intensity normalization across all rounds.

---

## Workspace and path conventions

### Raw data layout

Deconvolution operates on raw tiles organized as:

```text
<workspace>/
  {round}--{roi}/
    {round}-0000.tif
    {round}-0001.tif
    ...
  analysis/
    deconv/      # U16 deliverables (deconvolved tiles) + global scaling files
    deconv32/    # float32 staging / histograms
```

Naming conventions:

- `{round}`: arbitrary round token (often `1_9_17` etc.).
- `{roi}`: ROI identifier (e.g. `cortex`, `roiA`).
- `idx`: 0‑based integer, formatted as `{idx:04d}`.

### Deconvolved outputs

`deconvnew` writes deconvolution outputs in two parallel trees:

1. Float32 staging (`deconv32`):

   ```text
   <workspace>/analysis/deconv32/{round}--{roi}/{round}-{idx:04d}.tif
   ```

   - Shape: `(Z, C, Y, X)` or `(Z*C, Y, X)` depending on backend; the CLI ensures a consistent
     interpretation for histogram/quantization.
   - Dtype: `float32`.
   - Used for:
     - Global quantization (`precompute` / `quantize`).
     - Optional re‑processing or debugging.

2. U16 deliverables (`deconv`):

   ```text
   <workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif
   ```

   - Shape: `(Z, C, Y, X)` with fiducial planes appended at the end (see below).
   - Dtype: `uint16`.
   - Consumed by:
     - `preprocess register` as the source for registration and spot pipelines.

### Global scaling files

For each round, global scaling for deconvolution is stored as:

```text
<workspace>/analysis/deconv_scaling/{round}.txt
```

Contract:

- Plain text, loadable via `np.loadtxt`.
- Reshapeable to `(2, n_channels)`:
  - First row: per‑channel `m_glob` (offsets).
  - Second row: per‑channel `s_glob` (scale factors).

These files are consumed by:

- `deconvnew` backends when `mode=u16` (loading via `load_global_scaling`).
- `preprocess register` via `Workspace.deconved`/`Workspace.deconv_scaling(round)` when
  computing deconvolution rescaling unless input tiles are marked as `prenormalized`.

---

## Raw input TIFF contract

Deconvolution operates on raw TIFF stacks written by the microscope/acquisition stage.

### Location and naming

- Input path pattern for a given `{round}`, `{roi}`, and `idx`:

  ```text
  <workspace>/{round}--{roi}/{round}-{idx:04d}.tif
  ```

### Shape and fiducials

- Raw tiles are assumed to have shape:

  ```text
  (Z_total, Y, X) or (Z_total, C, Y, X)
  ```

  where:

  - `Z_total` includes both imaging planes and fiducial planes.
  - `n_fids` fiducial frames are appended at the end (configurable via `--n-fids`, default `2`).

- The CLI distinguishes imaging vs fiducial planes:

  - Imaging planes: first `Z_total - n_fids` planes.
  - Fiducial planes: last `n_fids` planes.

Depending on backend and configuration, deconvolution may operate on imaging planes only, while
fiducial planes are kept for downstream registration.

### Required metadata

Deconvolution uses metadata to infer:

- Waveform / power information for channel selection:

  - Raw tiles typically have a `waveform` JSON in TIFF metadata or adjacent files, which includes
    `params.step` and `params.powers`.
  - `cli_deconv` (and helpers) use this information to build per‑channel BaSiC paths like
    `basic/all-<wavelength>.pkl`.

- PSF step size:

  - If `waveform["params"]["step"]` is present, it is used to infer the PSF sampling step via
    `infer_psf_step`.
  - Otherwise, a default `step=6` is used (see `test_cli_deconv_multi.py`).

If metadata is malformed or missing critical fields, deconvolution may fall back to defaults or
raise errors in helper functions (e.g., BaSiC profile resolution).

---

## BaSiC profiles and illumination correction

`deconvnew` expects BaSiC illumination profiles under:

```text
<workspace>/basic/all-<channel>.pkl
```

Contract:

- `all-<wl>.pkl` (e.g. `all-405.pkl`, `all-560.pkl`) for each wavelength/channel.
- Each `.pkl` contains either:
  - A `DummyBasic`‑like object with `.darkfield` and `.flatfield`, or
  - A dict `{"basic": <BasicObject>}`.

Channel resolution:

- `cli_deconv` prefers **wavelength**‑based labels (`"405"`, `"560"`, etc.) from raw metadata to
  resolve BaSiC profiles, even when human‑readable names exist in `deconv32` metadata
  (see `test_cli_deconv_prefers_wavelengths_for_basic_lookup`).

If the required BaSiC profiles for a round/channel are missing, deconvolution will fail when the
backend attempts to set up illumination correction.

---

## Backends and output modes

The CLI supports three deconvolution output modes via `DeconvolutionOutputMode`:

- `"float32"` / `"f32"` → `DeconvolutionOutputMode.F32` (Float32HistBackend)
- `"u16"` → `DeconvolutionOutputMode.U16` (U16PrenormBackend)
- `"legacy"` / `"u16_tile"` → `DeconvolutionOutputMode.LEGACY` (LegacyPerTileU16Backend)

Backend selection controls which artifacts are produced:

- F32:
  - Writes float32 tiles under `analysis/deconv32`.
  - Produces per‑tile histograms for quantization.
- U16:
  - Writes U16 deliverables under `analysis/deconv`.
  - Uses global scaling from `analysis/deconv_scaling/{round}.txt`.
- Legacy:
  - Older per‑tile U16 backend; in current production pipelines it is used
    specifically for **bit rounds** (rounds whose tokens encode RNA/bit
    patterns) to preserve existing quantization semantics.

The flags `--skip-quantized` and `--skip-non-bit` in `run`/`batch` control whether U16 outputs are
written and which rounds are considered.

### Recommended usage: bit vs non‑bit rounds

In the standard production workflow:

- **Bit rounds** (RNA / barcode rounds whose names encode bit patterns, e.g.
  `1_9_17`, `2_10_18`):
  - Processed with the **legacy** backend:

    ```bash
    preprocess deconvnew run <workspace> 1_9_17 --mode=legacy
    ```

  - This uses `LegacyPerTileU16Backend` to produce per‑tile U16 outputs under
    `analysis/deconv/` with the same quantization behaviour as prior versions
    of the pipeline.

- **Non‑bit rounds** (e.g. protein, structural, or auxiliary rounds that do not
  participate directly in the codebook):
  - Processed with the **modern F32+quantization** path via the `easy` wrapper:

    ```bash
    preprocess deconvnew easy <workspace> <round_name>
    ```

  - `easy` ensures:
    - global scaling exists for the round (running `prepare` and `precompute`
      if needed), and
    - both U16 deliverables and backing float32 artifacts are generated using
      the newer backends.

This split allows bit rounds to retain legacy quantization while non‑bit rounds
benefit from the improved global histogram machinery.

---

## Command: `preprocess deconvnew precompute`

### CLI signature

```text
preprocess deconvnew precompute WORKSPACE ROUND_NAME [OPTIONS]
```

Arguments:

- `WORKSPACE`: path to the workspace root.
- `ROUND_NAME`: round token (e.g. `1_9_17`).

Options:

- `--bins INT` (default: `8192`)
- `--p-low FLOAT` (default: `0.001`)
- `--p-high FLOAT` (default: `0.99999`)
- `--gamma FLOAT` (default: `1.05`)
- `--i-max INT` (default: `65535`)

### Inputs

- Float32 staging tiles for the given round under:

  ```text
  <workspace>/analysis/deconv32/{round_name}--{roi}/{round_name}-{idx:04d}.tif
  ```

  produced by `deconvnew run` in `F32` mode (or `multi_run`/`batch` with `mode=float32`).

### Outputs

- Global scaling file:

  ```text
  <workspace>/analysis/deconv_scaling/{round_name}.txt
  ```

  containing two rows `[m_glob; s_glob]` for each channel, derived from aggregated histograms.

This file is later used by:

- `deconvnew quantize` to convert float32 tiles to U16 deliverables.
- `preprocess register` to rescale U16 tiles unless `prenormalized=True`.

---

## Command: `preprocess deconvnew quantize`

### CLI signature

```text
preprocess deconvnew quantize WORKSPACE ROUND_NAME [--roi ROI ...] [--n-fids N] [--overwrite/--no-overwrite]
```

Arguments:

- `WORKSPACE`: workspace root.
- `ROUND_NAME`: round token.

Options:

- `--roi ROI` (repeatable)
  - Restrict quantization to specific ROI names; default is all ROIs.
- `--n-fids N` (default: `2`)
  - Number of fiducial planes appended to each raw tile; used to preserve fiducial planes when
    generating U16 deliverables.
- `--overwrite/--no-overwrite` (default: `--no-overwrite`)

### Inputs

1. Float32 staging tiles:

   ```text
   <workspace>/analysis/deconv32/{round_name}--{roi}/{round_name}-{idx:04d}.tif
   ```

2. Global scaling file:

   ```text
   <workspace>/analysis/deconv_scaling/{round_name}.txt
   ```

### Outputs

1. U16 deliverables:

   ```text
   <workspace>/analysis/deconv/{round_name}--{roi}/{round_name}-{idx:04d}.tif
   ```

   - Shape: `(Z, C, Y, X)` including fiducial planes.
   - Dtype: `uint16` scaled using the global scaling parameters.

2. Cleanup:

   - After successful quantization, `quantize` deletes any `analysis/deconv32/{round_name}--{roi}` directories
     so only U16 deliverables remain.

These U16 tiles are the canonical inputs for `preprocess register`.

---

## Command: `preprocess deconvnew run`

### CLI signature

```text
preprocess deconvnew run PATH [ROUND_NAME] [OPTIONS]
```

Arguments:

- `PATH`: workspace root (or path resolvable to a workspace).
- `ROUND_NAME` (optional):
  - When provided, restricts processing to a single round.
  - When omitted or `"*"`, `run` discovers rounds via `Workspace.discover_rounds`.

Options (subset):

- `--roi ROI` (default: `"*"`)
- `--ref ROUND`
- `--limit INT`
- `--mode/--backend {u16,float32,legacy}` (default: `DeconvolutionConfig().output_mode`)
- `--histogram-bins INT` (default: `8192`)
- `--overwrite`
- `--delete-origin/--no-delete-origin` (default: `--delete-origin`)
- `--n-fids INT` (default: `2`)
- `--basic-name STR`
- `--debug`
- `--devices STR` (e.g. `"auto"`, `"0,1"`)
- `--stop-on-error/--continue-on-error` (default: stop)
- `--skip-quantized/--include-quantized` (default: `--skip-quantized=False`)
- `--skip-non-bit`

### Inputs

1. Raw tiles:

   ```text
   <workspace>/{round}--{roi}/{round}-{idx:04d}.tif
   ```

2. BaSiC profiles:

   ```text
   <workspace>/basic/all-<channel>.pkl
   ```

3. PSF reference:

   - Typically `fishtools/data/PSF GL.tif` (bundled with the package), discovered by
     `infer_psf_step`.

4. Optional global scaling:

   - When `mode=u16` and `skip-quantized=False`, `run` may load global scaling to write U16 tiles.
   - In the typical pipeline, U16 quantization is handled by `quantize`, so `run` is often used
     with `mode=float32` or `skip-quantized`.

### Outputs

Depending on `mode` and flags:

- F32 staging tiles in `analysis/deconv32/{round}--{roi}`.
- Optional U16 deliverables in `analysis/deconv/{round}--{roi}` (when `mode=u16` and not
  `skip-quantized`).

When `delete-origin=True`, `run` may call `safe_delete_origin_dirs` to remove raw input directories
once outputs are successfully created, preserving only deconvolved and registered‑ready tiles.

---

## Command: `preprocess deconvnew batch`

`batch` is a thin wrapper around `run` that forwards most options and allows for more
CLI‑friendly invocation. It shares the same input/output contract as `run`, but provides:

- Round discovery and iteration over multiple rounds.
- Simpler flag usage for common workflows.

Internally it calls:

```python
run.callback(path, round_name, ...)
```

---

## Command: `preprocess deconvnew easy`

### CLI signature

```text
preprocess deconvnew easy WORKSPACE [ROUND_NAME]
```

Behavior:

- For each selected round:
  1. Ensures `analysis/deconv_scaling/{round}.txt` exists by running:

     ```text
     preprocess deconvnew prepare WORKSPACE ROUND
     preprocess deconvnew precompute WORKSPACE ROUND
     ```

     (`prepare` is implemented in the deconvolution subpackage and handles histogram prep.)

  2. Launches `quantize` and `run --mode=u16` concurrently via `ThreadPoolExecutor`:

     ```text
     preprocess deconvnew quantize WORKSPACE ROUND
     preprocess deconvnew run --mode=u16 WORKSPACE ROUND
     ```

- This provides a single command to:
  - Precompute scaling.
  - Quantize float32 deconv tiles.
  - Produce U16 deliverables usable by `preprocess register`.

Inputs/outputs follow the contracts described in the `precompute`, `quantize`, and `run` sections.

---

## Summary

`cli_deconv` (`deconvnew`) is responsible for:

- Turning raw tiles `{round}--{roi}/{round}-{idx:04d}.tif` into:
  - Float32 deconv tiles under `analysis/deconv32`.
  - U16 deliverables under `analysis/deconv`.
  - Global scaling files under `analysis/deconv_scaling`.
- Producing artifacts that are **directly consumed** by:
  - `preprocess register` (U16 deconv tiles + global scaling).
  - `preprocess stitch` and `preprocess spots` via the registered stacks that follow.

These contracts ensure that geometry, bit/channel semantics, and illumination scaling are consistent
from deconvolution through registration and on into stitching and decoding.

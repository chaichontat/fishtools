# fishtools

[![Install and run](https://github.com/chaichontat/fishtools/actions/workflows/test.yml/badge.svg)](https://github.com/chaichontat/fishtools/actions/workflows/test.yml)

Tools for FISH analysis

## Installation

There are two environment files. One for

```sh
mamba env create -n fishtools -f environment_rsc_rapids_25.04.yml
mamba env update -n fishtools -f environment.yml
```

`mamba` is a drop-in replacement for `conda` that is faster and more reliable.
You can install `mamba` using `conda`:

```sh
conda install mamba -c conda-forge
```

## INM-Aware GAM (scripts/gam)

This repo includes a small R/mgcv workflow for INM-aware gene modeling in `scripts/gam/`:

1. Estimate an INM coupling curve `m(theta)` with a cyclic GAM: `x ~ s(theta, bs="cc")`.
2. Compute residual depth `r = x - m(theta)`.
3. Per gene fit a negative-binomial GAM with size-factor offset and smooth terms:
   - `offset(log(sf))`
   - `s(r)` (spatial / differentiation)
   - `s(theta, bs="cc")` (cell-cycle)
   - `ti(r, theta, bs=c("tp","cc"))` (gating interaction; used with marginals to avoid identifiability pathologies)
   - optional `batch` factor for pooled multi-dataset panels

Entry points and diagnostics are documented in `scripts/gam/README.md` (including `check_gam_diagnostics.R` which writes `m_hat(theta)`/`r` plots, `concurvity.tsv`, and `edf.tsv`).

## BrdU/EdU regression (scripts/brdu_regression)

### CNMF Usage_6 → BrdU+ retention (BrdU+EdU+ / BrdU+) model

This workflow fits a matched (stratum fixed-effect) grouped-binomial GLM on **BrdU+ cells only** to estimate:

- `f_hat = P(EdU+ | BrdU+)` as a function of **Usage_6** (binned into global quantiles)
- companions: `1/f_hat` and `T_S/Δt ≈ 1/(1−f_hat)` (standard convention)
- plus a marginal EdU labeling-index model `pE_hat = P(EdU+)`, used to derive canonical `T_C/Δt ≈ (T_S/Δt)/pE_hat`.

Run the model:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/usage6_brdue_fraction_model.py \
  --outdir scripts/_out/usage6_run \
  --include-leiden 7 8 10 \
  --pool-leiden \
  --write-by-unit
```

- Calling/threshold sensitivity: by default the script uses `adata.obs["brdu_pos"]` / `adata.obs["edu_pos"]`. To re-threshold from intensity columns, pass e.g. `--brdu-threshold 0.2 --edu-threshold 0.2` (defaults use `brdu_mean` / `edu_mean`).

- `--include-leiden`: which Leiden clusters to include from `~/nvme/all_progenitors.h5ad`
- `--pool-leiden`: pools the included Leidens into one curve per animal, while still stratifying by `dataset×theta_bin×leiden`
- `--write-by-unit`: writes `usage6_by_unit.csv` with unit = `dataset×roi×ccf_adjusted` (used for weighted error bars)

Outputs in `--outdir`:

- `usage6_by_animal.csv`: per-animal curves (used for gray per-animal lines)
- `usage6_by_unit.csv` (when `--write-by-unit`): per-unit predictions + weights for error bars
- `usage6_meta.csv`: simple across-animal summaries
- `call_audit_by_dataset.csv`: per-dataset BrdU/EdU prevalence in the analysis subset (useful for threshold drift diagnostics)
- `run.log`: progress + parameters

Plot results:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/plot_usage6_brdue_fraction_model.py \
  --indir scripts/_out/usage6_run
```

This writes:

- `usage6_f_hat.png`
- `usage6_inv_f_hat.png`
- `usage6_Ts_over_dt.png`
- `usage6_pE_hat.png` (if present in inputs)
- `usage6_Tc_over_dt.png` (if present in inputs)

If `usage6_by_unit.csv` exists, plots use **dataset×roi×ccf_adjusted-weighted** unit heterogeneity bands (weighted median with 10–90% bands; not confidence intervals), plus per-animal bands. For robustness, low-support unit×bin rows can be treated as missing via `--min-unit-trials` (default 20).

If you want hour/minute-scaled `T_S`/`T_C`, pass the pulse lag (e.g. 90 minutes):

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/plot_usage6_brdue_fraction_model.py \
  --indir scripts/_out/usage6_run \
  --delta-t-minutes 90
```

This also writes `usage6_Ts_minutes.png` and `usage6_Tc_minutes.png`.

## Mclust label smoothing

To regenerate the current smoothed label images for ROI 3 with clusters `4` and `6` merged, run:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/mclust_label_image_smooth.py \
  --in-h5ad /fast2/cs_outputs/all.stagate.h5ad \
  --labels-parquet /fast2/cs_outputs/all.stagate.mclust_sweep.parquet \
  --dataset 20251229_JaxA4_Sag5 \
  --roi 3 \
  --ccf-adjusted cortex \
  --outdir /tmp/mclust_label_demo_point_roi3_less_smooth \
  --label-key mclust_7 \
  --merge-labels 4,6 \
  --sigma 25.0 \
  --radius-multiplier 3.0 \
  --pixel-size 10.0 \
  --support-radius 3 \
  --raster-sigma 4.0 \
  --min-island-size 512 \
  --spline-smoothing 24.0
```

This overwrites the outputs in `/tmp/mclust_label_demo_point_roi3_less_smooth`.

Fit a hierarchical linear model across animals (unit = `dataset×roi×ccf_adjusted`):

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/hierarchical_usage6_mixedlm.py \
  --indir scripts/_out/usage6_run \
  --outcome tc_over_dt
```

For unit-level uncertainty, a block bootstrap is available via `--bootstrap-units-within-animal N`, and low-support unit×bin rows can be dropped via `--min-trials-b1` / `--min-trials-all`.

By default this treats animal as a random effect (MixedLM) and writes `hier_mixedlm_summary.txt` and `hier_mixedlm_fixed_effects.csv` into `--indir` (or `--outdir` if provided).

To treat animal as a fixed effect (recommended when you want explicit per-animal offsets), add `--animal-fixed`:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/hierarchical_usage6_mixedlm.py \
  --indir scripts/_out/usage6_run \
  --outcome tc_over_dt \
  --animal-fixed
```

This writes `hier_animal_fixed_summary.txt`, `hier_animal_fixed_coefs.csv`, and `hier_animal_fixed_pred_curve_by_animal.csv`.

## Compression

- `fishtools compress` is a command-line interface (CLI) tool that converts TIFF, JP2, and DAX image files to JPEG XL (JXL) files.
- `fishtools decompress` is a CLI tool that converts JXL files back to DAX files.

![Comparison of different compression quality](https://github.com/chaichontat/fishtools/assets/34997334/95230a08-4817-433d-a98d-67b5c442439d)

#### Usage

To use `fishtools`, simply run the `fishtools` command followed by the subcommand and path to the directory containing TIFF, JP2, or DAX files that you want to convert:

```sh
fishtools compress path/to/directory
```

By default, `fishtools compress` will convert all TIFF, JP2, and DAX files in the specified directory and its subdirectories. The converted JXL files will be saved in the same directory as the original files with the same name but with a `.jxl` extension.

You can also specify the quality level of the JXL files using the `--quality` or `-q` option. The quality level should be an integer between -inf and 100, where 100 is lossless. The default quality level is 99 (about 10x reduction in file size).

When the lossless option is selected, the output file is a `.tif` file with JPEG-XR encoding so that the file can be opened in ImageJ/BioFormats.

> BioFormats in ImageJ does not support JPEG-XL yet.
> It does support JPEG-XR which provides the same performance for lossless compression.
> JPEG-XR does not compress >8-channel images (compression scheme not in the specification).

If you want to delete the original files after conversion, you can use the `--delete` or `-d` option.

To use `fishtools decompress`, simply run the `fishtools decompress` command followed by the path to the JXL file or directory containing JXL files that you want to convert.
By default, `fishtools decompress` will convert all JXL files in the specified directory and its subdirectories.
The converted DAX files will be saved in the same directory as the original JXL files with the same name but with a `.dax` extension.

#### Examples

Convert all TIFF, JP2, and DAX files in a directory and its subdirectories to lossless TIFFs with JPEG-XR encoding:

```sh
fishtools compress path/to/directory
```

Convert all TIFF, JP2, and DAX files in a directory to JPEG-XL files and delete the original files:

```sh
fishtools compress path/to/directory --quality 99 --delete
```

Convert a single JXL file to DAX:

```sh
fishtools decompress path/to/file.jxl
```

Convert all JXL files in a directory to DAX:

```sh
fishtools decompress path/to/directory
```

## Probe ordering checklist

1. Verify simulation
2. BLAST some probes, make sure orientation is Plus/Minus.
3. Delete all old final files, both remote and local.
4. Run the script one last time.
5. Download said file and open to copy/paste into the Excel order sheet.
6. Save as a new file with today's date.
7. In the email, upload and redownload, verify that it's the same file.

## SLURM Dashboard

`slurmdash` launches a two‑pane terminal UI to monitor your SLURM jobs and tail their stdout/stderr.

```sh
slurmdash
```

Key bindings: `↑/↓` select job, `r` toggle running filter (turn off to load 7‑day history), `n` toggle other filter, `o/e/b` stdout/stderr/both, `PgUp/PgDn` scroll logs, `Enter` toggle follow, `q` quit.

## License

This project is licensed under the [MIT License](LICENSE).

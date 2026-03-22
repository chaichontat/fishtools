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
2. Record the coupling fit as a diagnostic (`inm_r2`); the per-gene model is fit on exported `r_um`/`AP_um`/`ML_um` covariates rather than `x - m(theta)`.
3. Per gene fit a negative-binomial GAM with size-factor offset and smooth terms:
   - `offset(log(sf))`
   - `s(r_um)` (radial / differentiation)
   - `s(theta, bs="cc")` (cell-cycle)
   - `s(AP_um, ML_um)` (planar spatial smooth)
   - `ti(AP_um, ML_um, r_um)` (planar-by-depth interaction)
   - `ti(r_um, theta)` (depth-by-cycle interaction; optional `--no-theta` removes theta terms)
   - pooled-panel random effects via `s(animal, bs="re")` and `s(animal:batch, bs="re")`
   - optional `brdu_pos + edu_pos + brdu_pos:edu_pos`
   - optional `Usage_*` covariates when exported in `cells.tsv`

The fitter currently expects `cells.tsv` to include a `batch` column for pooled analysis. Entry points and diagnostics are documented in `scripts/gam/README.md`; `fit_inm_panel.R` also writes per-gene `.rds` fits, optional diagnostics, and a BH-adjusted `*.qbh.tsv`.

### Usage pseudotime `bam` rerun

The current one-off Usage pseudotime analysis is driven by the Python orchestrator at `scripts/vz_usage_mgcv_pseudotime.py`. It reads `~/nvme/vz.h5ad`, fits one `mgcv::bam` negative-binomial model per gene through `scripts/gam/fit_pseudotime_panel.R`, and keeps all R/BLAS/OpenMP thread counts at `1` while parallelizing across genes in Python.

Current filter and pseudotime:

- include cells with `Usage_1 >= 0.2` or `Usage_7 >= 0.2`
- exclude cells with `Usage_3 > 0.2`, `Usage_4 > 0.2`, or `Usage_6 > 0.2`
- pseudotime is `Usage_7 / (Usage_1 + Usage_7)`

Run it with:

```sh
OMP_NUM_THREADS=1 \
OMP_THREAD_LIMIT=1 \
OMP_DYNAMIC=FALSE \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
MKL_DYNAMIC=FALSE \
BLIS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
CONDA_NO_PLUGINS=true \
MPLBACKEND=Agg \
conda run -n seq python -u scripts/vz_usage_mgcv_pseudotime.py
```

Current script settings:

- `raw_sct_corrected` layer
- `k=12`
- `30` Python workers
- one subprocess per gene
- `400` display bins for the fitted trajectories

The latest archived result set is here:

- `scripts/_out/vz_usage7_over_usage1plus7_k12_excl_u3_u4_u6_results`

The archived per-gene fit cache, including the per-gene `.rds` directories, is here:

- `scripts/_out/vz_usage7_over_usage1plus7_k12_excl_u3_u4_u6_fits`

## BrdU/EdU regression (scripts/brdu_regression)

### CNMF usage program → BrdU+ retention (BrdU+EdU+ / BrdU+) model

This workflow fits a matched (stratum fixed-effect) grouped-binomial GLM on **BrdU+ cells only** to estimate:

- `f_hat = P(EdU+ | BrdU+)` as a function of a selected CNMF usage program (binned into global quantiles; default `Usage_6`)
- companions: `1/f_hat` and `T_S/Δt ≈ 1/(1−f_hat)` (standard convention)
- plus a marginal EdU labeling-index model `pE_hat = P(EdU+)`, used to derive canonical `T_C/Δt ≈ (T_S/Δt)/pE_hat`.

Defaults:

- `--h5ad ~/nvme/all_progenitors.h5ad`
- `--tricycle-ref-csv neuroRef.csv`
- `--usage-parquet ~/nvme/cnmf_all_progenitors/usage_norm.k9.dt0.1.parquet`
- `--usage-col Usage_6`
- `--usage-bins 5` (quintiles)
- `--include-leiden 7 8 9 10`

Stage 1: fit the per-animal model:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/usage6_brdue_fraction_model.py \
  --outdir scripts/_out/usage6_run \
  --include-leiden 7 8 10 \
  --pool-leiden \
  --write-by-unit
```

- Calling/threshold sensitivity: by default the script uses `adata.obs["brdu_pos"]` / `adata.obs["edu_pos"]`. To re-threshold from intensity columns, pass e.g. `--brdu-threshold 0.2 --edu-threshold 0.2` (defaults use `brdu_mean` / `edu_mean`).
- `--usage-col`: selects which CNMF program to analyze. Output filenames use the matching stem, for example `Usage_7` writes `usage7_by_animal.csv`, `usage7_Ts_over_dt.png`, etc.
- `--usage-renorm-exclude-cols`: optional renormalization of the selected program by excluding other programs from the denominator, row-wise:
  `usage = usage_col / (1 - sum(excluded_cols))`
- `--include-leiden`: which Leiden clusters to include from the input H5AD
- `--obs-eq-filter`: optional exact-match `obs` filters applied before Leiden selection, for example `--obs-eq-filter manual_layer=1`
- `--exclude-animals`: optional animal IDs to remove from the analysis, for example `--exclude-animals JaxA2`
- `--pool-leiden`: pools the included Leidens into one curve per animal, while still stratifying by `dataset×theta_bin×leiden`
- `--write-by-unit`: writes `usage*_by_unit.csv` with unit = `dataset×roi×ccf_adjusted` (used for weighted error bars)

Current excitatory rerun example (`manual_layer=1`, pooled Leiden `5 6 7`, `Usage_7`, exclude `JaxA2`, renormalize by excluding `Usage_5` and `Usage_6`):

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/usage6_brdue_fraction_model.py \
  --h5ad nvme/cnmf_all_progenitors/umap_with_usages.k9.dt0.1.h5ad \
  --usage-parquet nvme/cnmf_all_progenitors/usage_norm.k9.dt0.1.parquet \
  --usage-col Usage_7 \
  --usage-renorm-exclude-cols Usage_5 Usage_6 \
  --include-leiden 5 6 7 \
  --obs-eq-filter manual_layer=1 \
  --exclude-animals JaxA2 \
  --pool-leiden \
  --write-by-unit \
  --outdir scripts/_out/usage7_all_excit_manual_layer1_567_k9_named_noJaxA2_q5_renorm_excl56
```

Note: the cNMF parquet joins directly to `nvme/cnmf_all_progenitors/umap_with_usages.k9.dt0.1.h5ad`. The plain `~/nvme/all_excit.h5ad` did not match this parquet key space as-is.

Outputs in `--outdir`:

- `usage*_by_animal.csv`: per-animal curves (used for gray per-animal lines)
- `usage*_by_unit.csv` (when `--write-by-unit`): per-unit predictions + weights for error bars
- `usage*_meta.csv`: simple across-animal summaries
- `call_audit_by_dataset.csv`: per-dataset BrdU/EdU prevalence in the analysis subset (useful for threshold drift diagnostics)
- `run.log`: progress + parameters

Stage 2: plot results:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/plot_usage6_brdue_fraction_model.py \
  --indir scripts/_out/usage6_run
```

This writes:

- `usage*_f_hat.png`
- `usage*_inv_f_hat.png`
- `usage*_Ts_over_dt.png`
- `usage*_pE_hat.png` (if present in inputs)
- `usage*_Tc_over_dt.png` (if present in inputs)

The plotter auto-detects the `usage*_by_animal.csv` stem from `--indir`, uses matching `usage*_by_unit.csv` if present, and labels the x-axis from the recorded `usage_col`. X-axis bin labels are shown as integers. If `usage*_by_unit.csv` exists, plots use **dataset×roi×ccf_adjusted-weighted** unit heterogeneity bands (weighted median with 10–90% bands; not confidence intervals), plus per-animal bands. For robustness, low-support unit×bin rows can be treated as missing via `--min-unit-trials` (default 20).

If you want hour/minute-scaled `T_S`/`T_C`, pass the pulse lag (e.g. 90 minutes):

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/plot_usage6_brdue_fraction_model.py \
  --indir scripts/_out/usage6_run \
  --delta-t-minutes 90
```

This also writes `usage*_Ts_minutes.png` and `usage*_Tc_minutes.png`.

Stage 3: fit a hierarchical summary model across units (unit = `dataset×roi×ccf_adjusted`):

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/hierarchical_usage6_mixedlm.py \
  --indir scripts/_out/usage6_run \
  --outcome tc_over_dt
```

For unit-level uncertainty, a block bootstrap is available via `--bootstrap-units-within-animal N`, and low-support unit×bin rows can be dropped via `--min-trials-b1` / `--min-trials-all`.

By default this treats animal as a random effect (MixedLM) and writes `hier_mixedlm_summary*.txt` and `hier_mixedlm_fixed_effects*.csv` into `--indir` (or `--outdir` if provided).

To treat animal as a fixed effect, add `--animal-fixed`:

```sh
CONDA_NO_PLUGINS=true conda run -n seq python scripts/brdu_regression/hierarchical_usage6_mixedlm.py \
  --indir scripts/_out/usage6_run \
  --outcome tc_over_dt \
  --animal-fixed
```

This writes `hier_animal_fixed_summary_*.txt`, `hier_animal_fixed_coefs_*.csv`, and `hier_animal_fixed_pred_curve_by_animal_*.csv`.

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

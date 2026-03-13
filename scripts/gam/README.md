# INM-Aware GAM (R/mgcv)

This folder contains a minimal, coupling-aware model for ventricular-zone INM data:

1. Fit the INM coupling curve `m(theta)` via a cyclic GAM: `x ~ s(theta, bs="cc")`.
2. Fit per-gene negative-binomial GAMs with `mgcv`, using:
   - `s(theta, bs="cc")` (cyclic cell-cycle component)
   - `s(r_um, bs="cs")` (depth component with shrinkage; `r_um` comes from principal-curve `r` in microns)
   - `s(AP_um, ML_um, bs="ts")` (planar spatial component with shrinkage)
   - `ti(AP_um, ML_um, r_um, d=c(2,1), bs=c("ts","cs"))` (3D spatial interaction with shrinkage)
   - `ti(r_um, theta, bs=c("cs","cc"))` (depth × cycle interaction; shrinkage in `r_um`)
   - optional `batch` (parametric factor) for pooled multi-dataset panels
   - optional `--no-theta` mode to remove `s(theta)` and `ti(r_um,theta)` (spatial-only per-gene model)

Run commands from repo root (`/home/chaichontat/fishtools2`). In this repo we typically use:

```bash
CONDA_NO_PLUGINS=true conda run -n seq <command>
```

for both Python and R entry points (e.g. `python ...` or `Rscript ...`).

## Generate Synthetic Test Data

```bash
Rscript scripts/gam/simulate_inm_panel.R /tmp/inm_synth 1
```

Outputs:
- `/tmp/inm_synth/cells.tsv` (`cell_id`, `x`, `r_um`, `AP_um`, `ML_um`, `theta`, `s`)
- `/tmp/inm_synth/counts.tsv` (`cell_id` + gene columns)
- `/tmp/inm_synth/truth.tsv` (gene labels: null/cycle/spatial/interaction)

## Fit The Model

```bash
Rscript scripts/gam/fit_inm_panel.R /tmp/inm_synth
```

Writes `/tmp/inm_synth/fit_results.tsv` with per-gene p-values for `s(r_um)`, `s(theta)`, `ti(r_um,theta)`, plus spatial `AP_um/ML_um` terms. If the output TSV already exists, the script resumes and only fits missing genes.

It also saves per-gene `mgcv` fit objects to `fits_rds/` next to the output TSV (one `.rds` per gene), so you can inspect coefficients and basis via `coef(fit)` and `predict(fit, type="lpmatrix")`.

### Spatial-only fits (`--no-theta`)

To drop the theta terms from the per-gene NB GAM (removes `s(theta)` and `ti(r_um,theta)`), use:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --no-theta
```

This sets `p_cycle`, `p_interaction` (and their `log_p_*`) to `NA`, and `cycle_amp_link`/`gating_index_link` to `NA` in the fit-results TSV.

### Parallelism (`--threads`)

`fit_inm_panel.R` parallelizes **across genes**. `--threads N` means “fit up to N genes concurrently”.
By default each per-gene `mgcv::bam()` uses **1 thread** (to avoid oversubscription); use `--bam-threads M` to set `mgcv::bam(nthreads = M)`.

### Basis choice (`--basis`)

By default we use standard bases for the large smooth terms:
- `bs='tp'` for `s(AP_um,ML_um)` and the AP/ML part of the tensor interaction
- `bs='cr'` for `s(r_um)` and the `r_um` parts of the tensor interactions

Default behavior:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --basis standard
```

If you explicitly want shrinkage bases:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --basis shrink
```

**Note:** `--basis shrink` (`ts/cs`) is usually slower than `--basis standard` (`tp/cr`) because there are more smoothing/penalty parameters to optimize.

### Variant Without EdU/BrdU Positivity Covariates

If `cells.tsv` contains `brdu_pos`/`edu_pos` you can explicitly disable those covariates and fit the same smooth decomposition without them:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --no-pos
```

### Optional per-gene diagnostics (`--diagnostics`)

By default, `fit_inm_panel.R` writes mgcv diagnostics during fitting (per-gene `summary()`, `gam.check()`, `k.check()`, EDF table, concurvity table).

To disable diagnostics:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --no-diagnostics
```

To explicitly enable (and optionally restrict to “hits”), use:

```bash
Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --no-pos --diagnostics
```

With `--diagnostics`, diagnostics are written only for “hits” where `min(p_spatial, p_cycle, p_interaction) <= 0.05`. Use `--diagnostics-p 0.01` to tighten, or `--diagnostics-all` to write diagnostics for every fitted gene.

## Where `cells.tsv` Comes From (and Cell Filtering)

The R GAM fitter/plotter read a simple "panel" directory with:
- `cells.tsv`: must contain `cell_id`, `x`, `r_um`, `AP_um`, `ML_um`, `theta`, `s`
- `counts.tsv`: `cell_id` + one column per gene (counts)
- optional `truth.tsv` (synthetic labels)

`fit_inm_panel.R` subsets/reorders `counts.tsv` to match `cells.tsv` via `cell_id`.

`fit_inm_panel.R` does not apply an additional in-script `r_um` cutoff.

### Optional EdU/BrdU positivity covariates

If `cells.tsv` also contains `brdu_pos` and `edu_pos` (0/1), the GAM includes them as parametric covariates plus their interaction:
- `brdu_pos + edu_pos + brdu_pos:edu_pos`

The output `fit_results.tsv` will include `p_brdu_pos`, `p_edu_pos`, and `p_brdu_edu`.

### Optional batch term for pooled panels

If you pool cells from multiple datasets/ROIs into one panel directory, you can include a parametric batch term.
If `cells.tsv` contains a `source` column, the fitter uses it as a `batch` factor:
- `... + batch + ...`

Additionally, when `batch` is available the INM coupling curve `m(theta)` is fit *per batch* (diagnostic only).

For real data, `brdu_pos`/`edu_pos` are typically computed from `obs.brdu_mean`/`obs.edu_mean` as:
- `log_brdu_mean = log1p(brdu_mean)`
- `log_edu_mean = log1p(edu_mean)`
then thresholded using `*.brdu_edu_thresholds.json` (see `scripts/gam/export_panel_from_princurve_h5ad.py`).

### Synthetic panels

`cells.tsv` is generated by the synthetic simulator:

```bash
Rscript scripts/gam/simulate_inm_panel.R /tmp/inm_synth 1
```

It writes:
- `cell_id = 1..n_cells`
- `theta ~ Uniform(0, 2π)`
- `x = m(theta) + noise` where `m(theta)` is a smooth periodic “INM coupling” curve (generated from a low-order Fourier series in the simulator)
- `s` as a lognormal size factor

### Real data in this repo

For real datasets we start from a principal-curve annotated AnnData like:
- `/home/chaichontat/fishtools2/working/20251230_JaxA4_Sag6/analysis/output/ccf-transforms/4/4.syn.annotated.princurve.h5ad`

Naming convention for these inputs:
- `*.x.princurve.h5ad` means the file contains multiple sub-ROIs.
- When selecting files to fit, include all principal-curve files that **do not** end with `bad` (e.g. keep `*.cortex.princurve.h5ad`, skip `*.cortexbad.princurve.h5ad`).

This file is produced by `scripts/princurve/find_princurve.py` and contains:
- `obs['t_local']`: local principal-curve coordinate (`obsm['principal'][:, 0]`)
- `obs['t_all']`: whole-ribbon coordinate (`1 - t_local`)
- `obs['t_neomeso']`: neocortex+mesocortex masked coordinate from per-mask endpoint intervals
- `obsm['principal']` (columns: `t`, `r`) where `t == t_local`
- optional `obsm['principal_r_signed']`

For downstream (t, r) work in this folder we typically export a TSV from that `.princurve.h5ad`:
- `scripts/gam/plot_princurve_cortex.py` writes `cells_tr.tsv` (e.g. `cell_id`, `t`, `r`, plus selected `obs` columns; it can also infer `theta` via tricycle projection).

To build GAM-ready `cells.tsv`/`counts.tsv` directly from principal-curve `.h5ad`, use:
- `scripts/gam/export_panel_from_princurve_h5ad.py` (single dataset)
- `scripts/gam/export_panel_from_princurve_h5ad_multi.py` (pooled multi-dataset)
- `scripts/gam/export_panel_from_pooled_h5ad.py` (single pooled h5ad that already contains multiple datasets/ROIs)

### Recommended Export + Run (single dataset)

1) Optional coordinate export (`cells_tr.tsv`) for quick QC of principal-curve coordinates:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_princurve_cortex.py \
  <input.princurve.h5ad> \
  --out <coord_out_dir> \
  --region-col ccf_adjusted --region cortex \
  --t-min 0.1 --t-max 0.75 --r-min 0.6 \
  --infer-theta-tricycle
```

2) Export GAM panel files (`cells.tsv`, `counts.tsv`, `panel_meta.json`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/export_panel_from_princurve_h5ad.py \
  <input.princurve.h5ad> \
  --out-dir <panel_dir> \
  --region-col ccf_adjusted --region cortex \
  --t-min 0.1 --t-max 0.75 --r-min 0.6 \
  --genes all
```

3) Fit and check GAM:

```bash
CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/fit_inm_panel.R <panel_dir>
CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/check_gam_diagnostics.R <panel_dir>
```

### Recommended Export + Run (single pooled h5ad, e.g. `~/nvme/wip.h5ad`)

This variant is for one large pooled AnnData that already contains per-cell `dataset`/`roi` metadata and tricycle angle in `obs.tricycle`.
It writes GAM-ready `cells.tsv`/`counts.tsv` with:
- `source = <dataset>.<roi>`
- `batch = <dataset>`
- unique `cell_id` values (safe even if obs names are duplicated)

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/export_panel_from_pooled_h5ad.py \
  /home/chaichontat/nvme/wip.h5ad \
  --out-dir <panel_dir> \
  --region-col ccf_adjusted --region cortex,cortex2 \
  --dataset-col dataset --roi-col roi \
  --theta-col tricycle \
  --counts-layer raw \
  --t-min 0.1 --t-max 0.75 --r-min 0.6 \
  --genes all
```

Then fit:

```bash
CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/fit_inm_panel.R <panel_dir> <out_tsv> --no-pos
```

If you want positivity covariates in `cells.tsv`, pass thresholds during export:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/export_panel_from_pooled_h5ad.py \
  /home/chaichontat/nvme/wip.h5ad \
  --out-dir <panel_dir> \
  --log-brdu-threshold <thr_log_brdu> \
  --log-edu-threshold <thr_log_edu>
```

### Cycle OT transition fit on pooled h5ad

For a direct, non-GAM view of radial transport across tricycle phase, fit an
entropic OT transition model between adjacent `theta` bins:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/fit_cycle_ot_transition.py \
  /home/chaichontat/nvme/wip.h5ad \
  --out-dir scripts/_out/gam_cycle_ot_wip \
  --r-col r_um --theta-col tricycle \
  --r-min-um 0 --r-max-um 300 \
  --n-theta-bins 72 --n-r-bins 120
```

Writes:
- `model.npz` (theta/r binning, histogram, and per-theta transition kernels)
- `summary.tsv` (fit parameters + sanity checks such as max row-stochasticity deviation)
- `hist_r_theta.png` (cell composition in `r_um × theta`)
- `drift_heatmap.png` (`E[r_next - r_current]` by `theta` and current `r`)
- `mean_drift_by_theta.png` (occupancy-weighted mean drift per theta bin)

### Composition heatmaps: x vs r_um (including BrdU/EdU state facets)

To quickly visualize cell composition as a 2D histogram in `r_um` vs an x-axis column
(typically tricycle theta), use:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_tricycle/plot_theta_by_rum.py \
  /home/chaichontat/nvme/wip.h5ad \
  --out-dir /tmp/plot_theta_by_rum \
  --r-col r_um --r-min 0 --r-max 300 \
  --theta-col tricycle \
  --dataset-col dataset \
  --basename theta_by_rum_lt300
```

Notes:
- If `--theta-col` is `tricycle` or `theta`, values are treated as an angle (wrapped to `[0, 2pi)`)
  and the x-axis uses `0, pi/2, pi, 3pi/2, 2pi` ticks.
- Otherwise the x-axis is treated as a generic numeric column (no wrapping, auto-ranged),
  e.g. `--theta-col log_edu_mean` or `--theta-col log_brdu_mean`.

Outputs four PNGs under `--out-dir`:
- `<basename>_heatmap.png`
- `<basename>_ranked_heatmap.png`
- `<basename>_heatmap_faceted.png`
- `<basename>_ranked_heatmap_faceted.png`

#### Facet by BrdU/EdU state (none, brdu_only, edu_only, dual)

If your AnnData has boolean `obs.brdu_pos` and `obs.edu_pos`, you can facet the panels
by their 4-state combination (instead of `--dataset-col`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_tricycle/plot_theta_by_rum.py \
  /home/chaichontat/nvme/wip.h5ad \
  --out-dir /tmp/plot_theta_by_rum \
  --facet-brdu-edu --brdu-col brdu_pos --edu-col edu_pos \
  --theta-col tricycle --r-col r_um --r-min 0 --r-max 300 \
  --basename theta_by_rum_brdu_edu_state_lt300
```

### Multi-dataset panels

If you want to **fit one model on cells pooled across multiple principal-curve `.h5ad` files** (e.g. two ROIs / experiments), export a single combined panel directory with:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/export_panel_from_princurve_h5ad_multi.py \\
  <a.princurve.h5ad> <b.princurve.h5ad> \\
  --out-dir <panel_dir> \\
  --region-col ccf_adjusted --region cortex \\
  --t-min 0.1 --t-max 0.75 --r-min 0.6 \\
  --genes all
```

This keeps `cells.tsv`/`counts.tsv` in the same format expected by `fit_inm_panel.R` (and includes a `source` column for provenance).

After export, run the pooled fit:

```bash
CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/fit_inm_panel.R <panel_dir>
```

For the Sag6+Sag5 pooled panel used in this repo, use Sag5-specific filtering/flip settings:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/export_panel_from_princurve_h5ad_multi.py \
  /home/chaichontat/fishtools2/working/20251230_JaxA4_Sag6/analysis/output/ccf-transforms/4/4.syn.annotated.princurve.h5ad \
  /home/chaichontat/fishtools2/working/20251229_JaxA4_Sag5/analysis/output/ccf-transforms/3/3.syn.annotated.princurve.h5ad \
  --out-dir /home/chaichontat/fishtools2/working/20251230_JaxA4_Sag6/analysis/output/ccf-transforms/4/gam_panel_allgenes_trfilt_rmaxnorm_pos_sag6_4_plus_sag5_3_sag5t02_flipx \
  --region-col ccf_adjusted --region cortex \
  --t-min 0.1 --t-max 0.75 --r-min 0.6 \
  --t-min-override "3.syn.annotated=0.2" \
  --flip-x-for "3.syn.annotated" \
  --genes all
```

This applies the same rules used for pooled GAM fitting:
- Sag6: `t in [0.1, 0.75]`, `x = r / r_max(t)`, keep `x > 0.6`
- Sag5: `t in [0.2, 0.75]`, `x = 1 - (r / r_max(t))`, keep `x > 0.6`

#### “Highest r is 1” normalization (rmaxnorm)

Some downstream plots/segmentations use a strict per-`t` upper-envelope normalization:

`r_norm(t) = r / r_max(t)`

so the local maximum radius is ~1 for each `t` slice. See:
- `scripts/gam/segment_two_bands.py` (`r_tmax_norm = r / r_max(t)`)
- `scripts/gam/plot_*_by_r_theta.R` and `scripts/gam/plot_label_by_rmaxnorm.R` (`calc_r_norm_strict()`)

## Visualize

```bash
Rscript scripts/gam/plot_inm_panel.R /tmp/inm_synth
```

Writes:
- `coupling.png` (x vs theta + fitted `m(theta)` and residual diagnostics)
- `pvals.png` (-log10 p-value summaries per component; uses `truth.tsv` if present)
- `surface_<gene>.png` for representative cycle/spatial/interaction genes (if `fit_results.tsv` exists)

### Python plotting for significant genes

To render per-gene fitted plots in Python/matplotlib (including AP/ML fitted heatmaps):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_significant_gams.py <panel_dir> [alpha] [out_dir]
```

Implementation notes: `scripts/gam/PLOTTING.md`.
Outputs per significant gene under `<out_dir>` (default: `<panel_dir>/plots_gam_significant_py`):
- `fit_ap_ml.png` (fitted mean on an `AP_um × ML_um` grid at fixed `r_um=0`, `theta=0`)
- `fit_r.png` / `fit_theta.png` when the corresponding component p-values are significant
- `fit_r_theta.png` when `p_interaction < alpha`

#### Interpreting `fit_ap_ml.png` (biologist-friendly)

By default `fit_ap_ml.png` uses `--apml-surface effect`, which plots the spatial smooth contribution
`s(AP_um,ML_um) + ti(AP_um,ML_um,r0)` on the **link scale** (additive in the linear predictor).

For NB/Poisson-style models with a log link, you can interpret this as a **relative enrichment/depletion** map:
differences correspond to log fold-changes in expected counts (holding other covariates at reference values).

To present this as a fold-change map with ratio units (e.g. “0.25× … 4×”):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_significant_gams.py <panel_dir> [alpha] [out_dir] \
  --apml-surface effect --link-ratio --link-ratio-range 0.25 4
```

This keeps the same underlying link-scale effect, but labels the colorbar in **× fold-change** units for readability.

Native-projection surface renders (`*_native_proj.png`) are written without axes and include a 500 μm scale bar for slides.

Command used for the current `gam_all_neurons` AP/MLR native triptych render (`r_min`, `r_med`, `r_p90`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_significant_gams.py \
  _out/gam_runs/gam_all_neurons/panel \
  0.05 \
  _out/gam_runs/gam_all_neurons/plots_significant_apmlr_amp0p2_v4_masked \
  --fit-results _out/gam_runs/gam_all_neurons/fit_results.tsv \
  --fits-dir _out/gam_runs/gam_all_neurons/fits_rds__fit_results \
  --genes-file _out/gam_runs/gam_all_neurons/apml_amp0p2_genes.txt \
  --plot-apmlr-interaction \
  --plot-apml-native-proj \
  --only fit_ap_ml__r_min_med_max_native_proj.png \
  --apml-mu-scale log \
  --apml-mu-vmin -8.0 \
  --apml-mu-vmax -4.0
```

Commands to plot AP/ML fitted `mu` (same run):

```bash
# AP/ML heatmap on native AP_um x ML_um grid
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_significant_gams.py \
  _out/gam_runs/gam_all_neurons/panel \
  0.05 \
  _out/gam_runs/gam_all_neurons/plots_significant_apmlr_amp0p2_v4_masked \
  --fit-results _out/gam_runs/gam_all_neurons/fit_results.tsv \
  --fits-dir _out/gam_runs/gam_all_neurons/fits_rds__fit_results \
  --genes-file _out/gam_runs/gam_all_neurons/apml_amp0p2_genes.txt \
  --plot-apmlr-interaction \
  --only fit_ap_ml.png \
  --apml-surface mu \
  --shrink none \
  --apml-mu-scale log \
  --apml-mu-vmin -8.0 \
  --apml-mu-vmax -4.0

# Native projection version of AP/ML fitted mu
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_significant_gams.py \
  _out/gam_runs/gam_all_neurons/panel \
  0.05 \
  _out/gam_runs/gam_all_neurons/plots_significant_apmlr_amp0p2_v4_masked \
  --fit-results _out/gam_runs/gam_all_neurons/fit_results.tsv \
  --fits-dir _out/gam_runs/gam_all_neurons/fits_rds__fit_results \
  --genes-file _out/gam_runs/gam_all_neurons/apml_amp0p2_genes.txt \
  --plot-apml-native-proj \
  --only fit_ap_ml_native_proj.png \
  --apml-surface mu \
  --shrink none \
  --apml-mu-scale log \
  --apml-mu-vmin -8.0 \
  --apml-mu-vmax -4.0
```

By default, plotting also gates interaction plots by per-gene EDF (skips interactions that are penalized away with `edf ~ 0`).
This requires a `diagnostics_summary.tsv` written next to the fit-results TSV. Generate it with:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/summarize_gam_diagnostics.py <diagnostics_dir> --out <fit_results_dir>/diagnostics_summary.tsv
```

Disable EDF gating with `--no-gate-by-edf`.

### AP/ML clustering workflow (k selection → clustering → plotting)

Use this when you want native-projection cluster plots with one shared colorbar per montage page.
All commands below assume the run root is in `OUT`.

```bash
OUT=_out/gam_runs/gam_all_excit_r300_leiden4
PANEL=${OUT}/panel
CLUSTER_DIR=${OUT}/cluster_apml_top80_tneomeso_cuml_pca
```

0) Generate per-gene predicted-`μ` means (required by `--min-log-mean-mu` and expression sorting):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_mu_distribution.py \
  ${PANEL} \
  --fits-dir ${OUT}/fits_rds__fit_results \
  --fit-results ${OUT}/fit_results.tsv \
  --jobs 32 \
  --restrict-t-neomeso
```

1) Build AP/ML clustering surfaces (all genes passing filters) into the `cuml_pca` folder:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/cluster_apml_patterns.py \
  ${PANEL} \
  --fit-results ${OUT}/fit_results.tsv \
  --fits-dir ${OUT}/fits_rds__fit_results \
  --out-dir ${CLUSTER_DIR} \
  --top-n 0 \
  --rank-by p_apml \
  --max-q-apml 1e-10 \
  --min-amplitude 0.2 \
  --mu-mean-tsv ${OUT}/mu_mean_by_gene.tsv \
  --min-log-mean-mu -7.0 \
  --restrict-t-neomeso
```

2) Determine candidate `k` (isolated outputs; does not write `summary_clustered_k*.tsv`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/optimize_cluster_k.py \
  ${CLUSTER_DIR} \
  --k-min 4 --k-max 12 --pc-k 20
```

This writes only under `${CLUSTER_DIR}/k_optimization/`, including:
- `k_sweep_metrics.tsv`
- `recommendation.tsv`
- `k_metrics.png`
- `k_overall_score.png`
- `labels_by_k/labels_k*.tsv`

3) Run final clustering at the selected `k` (for clustered summaries/diagnostic cluster plots):

```bash
K=6
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/cluster_surfaces_cuml.py \
  --mode apml \
  --surfaces ${CLUSTER_DIR}/surfaces.npz \
  --fit-results ${OUT}/fit_results.tsv \
  --out-dir ${CLUSTER_DIR} \
  --k ${K} \
  --pc-k 20
```

4) Plot per-cluster native-projection montages (all genes, sorted high→low expression):

```bash
K=6
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_cluster_example_surfaces.py \
  ${CLUSTER_DIR} \
  --summary ${CLUSTER_DIR}/k_optimization/labels_by_k/labels_k${K}.tsv \
  --cluster-col cluster_k${K} \
  --fits-dir ${OUT}/fits_rds__fit_results \
  --panel-dir ${PANEL} \
  --n-per-cluster 0 \
  --cluster-jobs 1 \
  --sort-by-expression \
  --mu-mean-tsv ${OUT}/mu_mean_by_gene.tsv \
  --native-proj-apml-surface mu \
  --native-proj-out-suffix _mu \
  --native-proj-mu-vmin -8.0 \
  --native-proj-mu-vmax -4.0 \
  --native-proj-elev-deg -10 \
  --native-proj-azim-deg -110 \
  --force
```

Plot output:
- `${CLUSTER_DIR}/cluster_examples_native_proj/cluster_XX/montage_fit_ap_ml_native_proj_mu*.png`
- `${CLUSTER_DIR}/cluster_examples_native_proj/index.tsv`

Notes:
- Native projection defaults are `elev=-10`, `azim=-110`, `roll=180`, `latlon=true`, `graticule=ijk`.
- `--n-per-cluster 0` plots all genes in each cluster.
- Gene order within each cluster is expression high→low by `mu_mean_by_gene.tsv` (`log_mean_mu`).
- `--force` overwrites existing per-gene and montage PNGs.
- Parallelism is at the cluster level (`--cluster-jobs`): montage rendering requires `plot_significant_gams.py --max-workers=1`.

## Diagnostics (Coupling, Concurvity, EDF)

To validate the fits on real data:

1. `m_hat(theta)` and `r = x - m_hat(theta)` vs `theta` (coupling diagnostics; not used directly in the per-gene model)
2. `concurvity()` on a representative gene model (numerical stability / identifiability)
3. EDF sanity check (EDF near `k` is a warning sign that you are fitting noise)

Run:

```bash
Rscript scripts/gam/check_gam_diagnostics.R <panel_dir> [GENE] [OUT_DIR]
```

## Ordinal pulse-stage GAM (BrdU/EdU)

If `cells.tsv` includes `brdu_pos` and `edu_pos` (0/1), you can fit an ordered-categorical ordinal GAM for the three “in-window” pulse stages:

- 1 = BrdU+ / EdU−
- 2 = BrdU+ / EdU+
- 3 = BrdU− / EdU+

Cells with (BrdU−, EdU−) are dropped for this model.

```bash
Rscript scripts/gam/fit_pulse_stage_ocat.R <panel_dir> [out_dir]
```

Outputs (in `out_dir`, default = `<panel_dir>`):
- `pulse_ocat_predictions.tsv` (per-cell `eta` and `p(stage=k)` for `fit0/fit1/fit2`)
- `pulse_ocat_cutpoints.tsv` (fitted cutpoints in latent-score space)
- `pulse_ocat_fit*.gam.rds` (saved `mgcv` fits)

Writes under `OUT_DIR`:
- `coupling_diagnostics.png`
- `concurvity.tsv` (long format: `kind,row,col,value`)
- `edf.tsv` (with `warn_edf_near_refdf`)

### How To Interpret QC Outputs

These checks are specifically about whether the coupling diagnostics and the per-gene smooth decomposition are behaving as intended.

#### `coupling_diagnostics.png`

This figure has four panels:

1. **Coupling fit (x vs theta):** scatter of `x` vs `theta` with the fitted `m_hat(theta)`.
   - Expect a smooth periodic trend, not a jagged curve that tracks noise.
   - If `cells.tsv` has `source` and you pool datasets, `m_hat(theta)` is fit per batch; strong between-batch differences are a sign you should not force one shared coupling curve.
2. **Residual depth histogram:** distribution of `r = x - m_hat(theta)`; look for pathologies (spikes, heavy tails) that can destabilize NB fits.
3. **Residual vs theta:** scatter of `r` vs `theta`.
  - You want *no obvious periodic banding*. Strong residual cyclicity can indicate that depth/cycle structure is not being cleanly separated in the downstream model.
4. **Residual QQ-plot:** a coarse check for extreme outliers; heavy tails are common in real data, but very strong tails can cause per-gene failures.

In addition to the plot, the script prints:
- `corr(r, cos(theta))` and `corr(r, sin(theta))` (should be near 0)
- `Residual cyclicity check: gam(r ~ s(theta)) p=...`
  - Treat this as a *diagnostic* only. With large `n`, very small p-values can occur for tiny effects; use it alongside the residual-vs-theta plot.

#### `concurvity.tsv`

`mgcv::concurvity()` measures how well each smooth term can be explained by the others (smooth-term collinearity).

- Values near `1` mean the decomposition across smooth terms (e.g. `s(r_um)`, `s(theta)`, `s(AP_um,ML_um)`, `ti(AP_um,ML_um,r_um)`, `ti(r_um,theta)`) is numerically unstable.
- As a rule of thumb: `>0.9` is a warning; `>0.99` means component p-values are often not trustworthy.
- If concurvity is high:
  1. Reduce flexibility (lower interaction `k` first, then marginal `k` values).
  2. Re-check that `theta` is wrapped and that pooled datasets use per-batch coupling (`source`).
  3. Verify dataset-specific transforms (e.g. Sag5 `x` flip) were applied consistently upstream.

#### `edf.tsv`

`edf.tsv` reports `edf` and `Ref.df` per smooth term from `summary(fit)$s.table`, plus:
- `edf_over_refdf`
- `warn_edf_near_refdf` (flag when EDF is close to the basis limit)

If a term repeatedly has EDF very close to its basis limit, it is a strong hint that `k_*` is too large or the model is trying to fit noise. Prefer lowering `k_int` before lowering the marginal `k` values.

### Interpreting Per-Gene Outputs

`fit_results.tsv` contains both component p-values and simple effect sizes on the *link* (log) scale:

- `p_spatial`, `p_cycle`, `p_interaction`: evidence that `s(r_um)`, `s(theta)`, or `ti(r_um,theta)` is non-zero.
- `p_apml`: evidence that `s(AP_um,ML_um)` is non-zero.
- `p_apml_r_um`: evidence that `ti(AP_um,ML_um,r_um)` is non-zero.
  - With large `n`, p-values can be extremely small for tiny effects; they are best used for ranking, not for deciding “biology labels” alone.
- `cycle_amp_link`: `max_theta(eta_hat) - min_theta(eta_hat)` at a representative `r`.
- `spatial_grad_link`: `eta_hat(r_hi) - eta_hat(r_lo)` at fixed `theta`.
- `gating_index_link`: `(cycle amplitude at r_hi) - (cycle amplitude at r_lo)`.

Because these are on the link scale, `exp(effect_size)` is a rough multiplicative change in expected counts (holding offset and covariates fixed).

#### Note: tiny negative p-values

In some `mgcv` builds, extremely small smooth-term p-values can appear as tiny negative numbers due to numerical roundoff. This repo clamps p-values to `[0,1]` when extracting them from fits; for any legacy output files you can repair them with:

```bash
Rscript scripts/gam/clamp_fit_results_pvals.R <fit_results.tsv>
```

## Smoke Test

```bash
Rscript scripts/gam/synthetic_smoke_test.R
```

## Simplex Topic Model (Logistic-Normal / ALR)

This mode is for modeling **topic loadings** (e.g. cNMF `Usage_*`) as a composition that must sum to `1`.
It fits Gaussian `bam()` models on ALR coordinates `log(u_k/u_ref)` and reconstructs per-topic loadings with a softmax so outputs are always on the simplex.

Important caveats (reviewer-facing):
- The model is fit on **log-ratios**: each fitted response is `z_k = log(u_k/u_ref)`, so effects are always **relative to the reference topic** (and therefore relative to the rest after closure).
- This is `K-1` **separate univariate** Gaussian GAM fits (one per ALR component). We do not model cross-topic covariance, and results are **not invariant** to the ALR reference choice.
- We apply an `eps` floor then re-close to the simplex before taking logs. This avoids `log(0)` but changes the estimand; treat `eps` as a hyperparameter and check sensitivity.
- The plotted simplex loadings are a **plug-in** back-transform: `u_hat(x) = softmax(z_hat(x))`. This is not exactly `E[u|x]` under a logistic-normal (nonlinearity/Jensen).
- `--exclude-random-effects` means “set `s(animal)` and `s(ab)` to 0 in prediction”, not “marginalize/average over animals/batches”.

Inputs:
- `${PANEL}/cells.tsv` with at least: `cell_id`, `r_um`, `AP_um`, `ML_um`, `batch` (and `theta` unless you pass `--no-theta`).
  - `r_um`, `AP_um`, `ML_um` must be finite for all rows used in the fit.
  - If `cells.tsv` does not contain an explicit `animal` column, the fitter will parse it from `batch` using a `JaxA\\d+` regex (and will error if it cannot).
- A `usage_norm.*.tsv` where the first column is the cell id (must match `cells.tsv:cell_id`) and the remaining columns include `Usage_1..Usage_K` (values in `[0,1]`).

Fit:

```bash
PANEL=_out/gam_panel__cnmf_topics_k9_dt0.1__rg_sweep__20260303
USAGE=_out/cnmf_all_progenitors/usage_norm.k9.dt0.1.tsv

CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/fit_inm_simplex_panel.R \
  ${PANEL} \
  --usage-tsv ${USAGE} \
  --eps 1e-4 \
  --alr-ref auto \
  --r-max 400 \
  --threads 6 \
  --bam-threads 1 \
  --basis standard \
  --k-uv 15
```

Plot native projection:

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_simplex_native_proj.py \
  ${PANEL} \
  --exclude-random-effects \
  --out-dir ${PANEL}/plots_native_proj_simplex_u \
  --latlon \
  --graticule ijk \
  --elev-deg -10 \
  --azim-deg -110 \
  --roll-deg 180
```

Plot outputs:
- Native projection (one PNG per program + montage): `${PANEL}/plots_native_proj_simplex_u/`
- 2D slices (one PNG per program, with `AP_um×r_um` and `ML_um×r_um` panels): `${PANEL}/plots_apmlr_simplex_u/`
  - By default, the `r` axis is capped at the fitted `r_max` recorded in `simplex_meta.json` (to avoid showing extrapolation past the fitted range).

Notes:
- `--r-max` filters the input cells to `r_um <= r_max` *before fitting* (useful if coverage is poor at high `r`).
- `--exclude-random-effects` subtracts `s(animal)` and `s(ab)` from the ALR link predictions before simplex inversion.
  - Use this when you want a population-level spatial pattern instead of conditioning on a single reference `animal`/`ab` level.
- `--marginalize-animal` (plotting) averages simplex loadings across animal levels (uniform weights).
  - This keeps `s(animal)` but excludes `s(ab)` to avoid conditioning on a specific batch.
  - Incompatible with `--exclude-random-effects`.
- Optional: pass `--label-tsv PATH` to name topics in plot titles. The TSV must contain `program` and either `curated_label` or `label`.

Outputs under `${PANEL}`:
- `fit_results.simplex.tsv`
- `simplex_meta.json`
- `fits_rds__fit_results_simplex/ALR_P*_vs_P*.gam.rds`

### ILR Variant (Reference-Free Coordinates)

If you want to avoid choosing an ALR reference topic, you can fit in ILR (isometric log-ratio) coordinates.
This still fits `K-1` separate Gaussian GAMs, but uses an orthonormal basis in clr space (pivot ILR).

Fit:

```bash
CONDA_NO_PLUGINS=true conda run -n seq Rscript scripts/gam/fit_inm_simplex_panel_ilr.R \
  ${PANEL} \
  --usage-tsv ${USAGE} \
  --eps 1e-4 \
  --r-max 400 \
  --threads 6 \
  --bam-threads 1 \
  --basis standard \
  --k-uv 15
```

Plot (note `--meta`):

```bash
CONDA_NO_PLUGINS=true conda run -n seq python scripts/gam/plot_simplex_native_proj.py \
  ${PANEL} \
  --meta ${PANEL}/simplex_meta_ilr.json \
  --exclude-random-effects \
  --out-dir ${PANEL}/plots_native_proj_simplex_u__ilr \
  --apmlr-out-dir ${PANEL}/plots_apmlr_simplex_u__ilr \
  --latlon \
  --graticule ijk \
  --elev-deg -10 \
  --azim-deg -110 \
  --roll-deg 180
```

ILR outputs under `${PANEL}`:
- `fit_results.simplex_ilr.tsv`
- `simplex_meta_ilr.json`
- `fits_rds__fit_results_simplex_ilr/ILR_C*.gam.rds`

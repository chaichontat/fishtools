# Methods: BrdU-first / EdU-second interaction phenotypes (panel-constrained)

This document defines the **mechanistic endpoints** and **interaction phenotypes** used in our BrdU-first / EdU-second analysis.

Canonical input: `~/nvme/all_progenitors.h5ad` (targeted panel).

## 1) Data, cohort, and conventions

- Analysis compartment: `all_progenitors2.h5ad` contains the analysis population; use `--leiden-col manual_annotation` and treat `obs['manual_annotation']` as the cluster field (e.g. `apical`, `intermediate`).
- Pulse labels (measurement layer): current GLM runs use **threshold-logistic soft calls** from per-unit thresholds in `adata.uns` (`p=0.5` at threshold; default `p=0.01/0.99` at threshold `±1` log unit). The GMM-based soft-call path is retained as a legacy option.
  Legacy GMM path: for each channel (`x = log_edu_mean` and `x = log_brdu_mean`) fit a `K`-component Gaussian mixture in a latent intensity space `u`, with per-batch nuisance parameters:
  - `x = α_dataset + β_dataset * u`, and `u | k,dataset ~ Normal(μ_k, γ_dataset * σ_k^2)`
  - mixing proportions `π_k` are shared across datasets (global prevalence), while `α/β/γ` capture run-specific intensity shifts.
  - “positive” is defined as the upper tail of components in `u`. The canonical fit uses a parsimonious setting to avoid a single very-broad tail component:
    - `K=6` components per channel
    - variance tied across components (`--batchaware1d-variance tied`) to avoid a very broad “tail” component eating into the main mass
    - exclude exact zeros from the fit (`--edu-zero-policy exclude`, `--brdu-zero-policy exclude`), and force `P(pos)=0` when `x==0` at inference
    - EdU+ taken as the top-2 highest-mean components (`--edu-pos-topk 2`)
    - BrdU+ taken as the **highest-mean** component (`--brdu-pos-topk 1`).
  To avoid per-dataset artifacts where the implied `p≈0.5` boundary drifts into the main mass, we optionally apply a **floor-only** per-dataset logit-shift calibration that can only raise thresholds (never lower them):
  - `--calibrate-to-xmin --edu-xmin 4.5 --brdu-xmin 6.5`
  This yields per-cell marginals:
  - `pE_i = P(E=1 | log_edu_mean_i, dataset_i)`
  - `pB_i = P(B=1 | log_brdu_mean_i, dataset_i)`
  We then form an **approximate joint posterior** by conditional independence:
  - `pi_11 = pB * pE`, `pi_10 = pB * (1-pE)`, `pi_01 = (1-pB) * pE`, `pi_00 = (1-pB) * (1-pE)`.
  These `pi_*` are stored in the GMM cache (canonical: `~/nvme/all.gmm.npz`, fit on `~/nvme/all.h5ad`) keyed by `cell_id = dataset + ":" + obs_name`, so it aligns to any downstream `.h5ad` subset.
  Manual hard calls (`brdu_pos`, `edu_pos`) are used only for QC benchmarking (agreement tables vs the legacy threshold calls). They are not used to calibrate the canonical soft-call posteriors.
- Grouping variables:
  - `dataset = obs["dataset"]` (string).
  - `animal`: parsed from dataset string as `JaxA#`.
  - `orientation`: inferred from dataset string (`Sag` if contains `"Sag"`, `Coro` if contains `"Coro"`, else `Unknown`).
- Spatial coordinates:
  - `obsm["AP_ML_um"]` in microns (upstream naming may be swapped).
  - **Explicit convention used by our scripts:** we always treat:
    - `AP_um = obsm["AP_ML_um"][:, 0]`
    - `ML_um = obsm["AP_ML_um"][:, 1]`
    regardless of upstream label semantics.
  - 1D spatial conditioning uses AP for Sag datasets and ML for Coro (and `Unknown`) datasets.

## 2) Tricycle phase (theta) and phase bins (matching strata)

We compute a tricycle angle `θ` using `neuroRef.csv` (`symbol`, `pc1.rot`, `pc2.rot`):

1. restrict neuroRef genes to those present in `adata.var_names`
2. within each dataset: mean-center the reference-gene expression and project to (`pc1`, `pc2`)
3. define `θ = atan2(pc2, pc1)` (shared circle, `(-π, π]`)

We then discretize `θ` into `theta_bin` for matched strata in one of two modes:

- `quantile` (legacy default): quantile bins within each `dataset×leiden` (shared edges reused across endpoints).
- `angle`: fixed absolute angular bins (12 equal-angle slices of `[0,2π)` after mapping to that range).

`angle` makes “theta_bin=k” refer to the same geometric phase interval across datasets; `quantile` is retained as a sensitivity baseline.

## 3) Gene gates (current): dataset-wise Pearson residuals + theta residualization

All per-gene predictors are binary gates `G ∈ {0,1}`. Gates must behave sensibly for a **zero-inflated targeted panel**, so we use Pearson residuals rather than OLS on log1p counts.

### 3.1 Raw inputs

- Raw counts are taken from `adata.layers["raw"]` when present, else from `adata.X`.
- **Gene symbol normalization:** `Kctd16-*` isoform features (e.g. `Kctd16-206`, `Kctd16-211`) are summed into a single `Kctd16` raw-count vector before gate construction and all downstream endpoints. We do not report isoform-separated `Kctd16-*` calls.
- `obs["total_counts"]` is required for expectations.

### 3.2 Dataset-wise Pearson residuals (NB variance stabilization)

Within each dataset `d` and gene `g`:

- raw count: `x_{ig}`
- total counts: `t_i`
- dataset gene frequency: `p_{dg} = (Σ_i x_{ig}) / (Σ_i t_i)`
- expected mean: `μ_{ig} = t_i * p_{dg}`
- NB variance: `Var(x_{ig}) = μ_{ig} + μ_{ig}^2 / θ_nb`
- Pearson residual: `r_{ig} = (x_{ig} - μ_{ig}) / sqrt(Var(x_{ig}))`

Defaults:

- `θ_nb = 100`
- clip to `[-10, +10]`

### 3.3 Residualize against tricycle phase within dataset

To remove a (linearized) phase-position component before thresholding, we regress within each dataset:

`r_{ig} = a_g + b_g sin(θ_i) + c_g cos(θ_i) + ε_{ig}`

and use `ε_{ig}` as the gate score.

### 3.4 Gate threshold (per-dataset quantile)

For each dataset and gene:

- Define `k_d = ceil((1−q) * n_d)` within each dataset `d` (where `n_d` is the number of cells in `d`).
- Set `G_{ig}=1` for the **top `k_d`** cells by `ε_{ig}` within `d`. To make the gate **exact-size and deterministic** under ties (common in zero-inflated panels), we add an infinitesimal, **stable** tie-breaker: `score_i = ε_{ig} + eps_tie * j_i`, where `j_i` is a deterministic per-cell jitter derived from a stable cell identifier (e.g. `obs_names` hash) and `eps_tie` is tiny (e.g. `1e-9`). This breaks ties without changing the ordering of non-tied scores.

`q` is gene-specific (from the Mode A “best-q” scan table or from an explicit sensitivity run).
Concrete mapping: `q=0.95` means “gate+ is the top 5% of cells per dataset” (after θ residualization).

**Design constraint:** gates are *not* defined within microscopic `dataset×theta_bin×leiden` slices. Phase/type matching belongs in the **model strata**, not in the gate. For zero-inflated targeted panels, forcing “top 5% per theta bin” manufactures false “gene+” cells in bins where the gene is truly off.

## 4) Mechanistic endpoints from the full 2×2 BrdU/EdU table

Each cell belongs to one of four quadrants. To avoid confusion with the BrdU indicator `B`, we label quadrants by their `(B,E)` bits:

- `q11`: Dual (`B=1`, `E=1`)
- `q10`: BrdU-only (`B=1`, `E=0`)
- `q01`: EdU-only (`B=0`, `E=1`)
- `q00`: Double-negative (`B=0`, `E=0`)

We estimate four matched conditional ORs for each gate:

### E1: within BrdU+ (`q11` vs `q10`)

`OR_E1 = OR_{GE | B=1} = odds(E=1 | G=1, B=1, matched) / odds(E=1 | G=0, B=1, matched)`

### E3: within BrdU− (`q01` vs `q00`)

`OR_E3 = OR_{GE | B=0}`

E3 is the internal “general EdU propensity / entry / calling” axis: if E3 is large, the gate predicts EdU+ even without BrdU conditioning.

### E2 and E4 (mirror conditionals; diagnostics)

- `OR_E2 = OR_{GB | E=1}` (within EdU+)
- `OR_E4 = OR_{GB | E=0}` (within EdU−)

## 5) Interaction phenotypes: what we treat as “BrdU-conditioned kinetics-like” signal

### 5.1 Primary interaction (IntE / logIOR)

On the log scale:

- `IntE = log(OR_E1) - log(OR_E3)`

On the OR scale:

- `IOR = OR_E1 / OR_E3`

Interpretation:

- `IntE < 0` means gene+ has **lower** BrdU-conditioned persistence/progression than expected from its general EdU propensity.
- `IntE > 0` means gene+ has **higher** BrdU-conditioned persistence/progression than expected from its general EdU propensity.

### 5.2 Mirror interaction (IntB; diagnostic)

- `IntB = log(OR_E2) - log(OR_E4)`

IntB is used as a robustness check; it is often noisier because the EdU− cohort can be small in some strata.

### 5.3 Probability-scale interaction (ceiling-aware)

Odds ratios can look extreme when `P(E=1 | B=1, …)` is near saturation. We therefore also report an additive interaction on the probability scale using the same matched strata:

- `Δ1 = P(E=1 | G=1, B=1, matched) − P(E=1 | G=0, B=1, matched)`
- `Δ0 = P(E=1 | G=1, B=0, matched) − P(E=1 | G=0, B=0, matched)`
- `ΔINT = Δ1 − Δ0`

Pooling (probability scale): within each stratum `s = dataset × theta_bin × leiden` and each BrdU slice `b∈{0,1}`, we form a stratum-level contrast
`Δ_b(s) = p̂(E=1|G=1,B=b,s) − p̂(E=1|G=0,B=b,s)` (with small-count smoothing), and then pool within each dataset by a **trial-count weight**
`w_b(s) = n_{s,B=b,G=0} + n_{s,B=b,G=1}` (number of cells contributing to that BrdU slice in that stratum). Strata without a within-`B` gate contrast (missing either `G=0` or `G=1`) get weight `0`. The pooled estimate is the weighted mean `Σ_s w_b(s)·Δ_b(s) / Σ_s w_b(s)`. We implement this as per-dataset numerator/denominator contributions so that uncertainty can be computed by **dataset bootstrap** with shared resamples (same resampling unit as MH ORs). For publishable cohort-level statements we additionally summarize `Δ1/Δ0/ΔINT` at the **animal** level (e.g., LOAO held-out-animal estimates with a t-based CI across animals).

Interpretation and “what is a real kinetics-like gene” criteria are in `scripts/brdu_regression/biology-analysis.md` (kept out of this methods document).

## 6) Estimators and uncertainty

### 6.1 Legacy MH endpoint tables (retired)

Earlier versions of this project used Mantel–Haenszel (MH) pooled OR tables for fast endpoint screening. We no longer run MH for panel selection or inference; all current screening and reporting is GLM-based using soft-label expected-likelihood.

- `strata = dataset × theta_bin × leiden`

Uncertainty is computed by **dataset bootstrap**, resampling datasets with shared resamples so endpoint differences preserve covariance. (ORs use Haldane–Anscombe correction; probability-scale contrasts use the same dataset bootstrap with small-count smoothing.)

### 6.2 Grouped-binomial fixed-effect GLM (auditfix v2; interaction coefficient)

IntE can also be estimated directly as the interaction coefficient:

`logit P(E=1) = α_stratum + β_G G + β_B B + β_GB (G×B)`

with `strata = dataset×theta_bin×leiden`.

**Important workflow note:** do not start with this GLM on a large gene list. Use the screening pipeline first (Mode A scan + MH/interaction summaries + confound flags) to narrow to a short candidate set, then apply the GLM estimator as a confirmation / sensitivity step on that shortlist.

Interpretation:

- `β_G` corresponds to `log(OR_E3)`
- `β_G + β_GB` corresponds to `log(OR_E1)`
- `β_GB` is `IntE` (logIOR)

The grouped-binomial implementation (aggregating counts by stratum and `(G,B)`) is likelihood-equivalent to cell-level Bernoulli FE GLM and avoids sparse-table continuity correction artifacts.

## 7) Spatial mechanisms: composition vs within-location effect

Spatial structure can induce interaction signals via:

1. **Composition (spatial confounding):** gate prevalence differs across regions with different baseline BrdU→EdU behavior.
2. **Within-location effect:** interaction persists after coarse spatial conditioning.

We distinguish them by computing:

- marginal: `strata = dataset×theta_bin×leiden`
- space-conditioned: `strata = dataset×theta_bin×leiden×spatial_bin`

and reporting:

- `δ_marg = IntE_marginal`
- `δ_space = IntE_space_conditioned`
- `δ_comp = δ_marg - δ_space`

## 8) Reference implementations (scripts)

- Gate scan / best-q: `scripts/scan_ts_brdu_first_gate_mode_a.py`
- Pulse measurement layer (batch-aware GMM; writes `cache/per_cell_qc.npz`): `scripts/gam/qc_batch_aware_brdu_edu_gmm.py`
- Legacy hard-call MH phase-consistency + quadrant validation (comparison only): `scripts/mh_phase_consistency_mode_a.py`, `scripts/mh_quadrant_validation.py`
- Consultant exec summary tables: `scripts/make_consultant_exec_summary.py`
- Theta-bin mode sensitivity (quantile vs angle): `scripts/verify_theta_bin_mode.py`
- Leiden gate-vs-type diagnostics: `scripts/leiden_confound_diagnostics.py`
- GLM interaction estimator and spatial-adjusted variant: `scripts/auditfix2_stratum_glm.py`, `scripts/auditfix2_spatial_adjusted_dataset_sliced_glm.py`

## 9) How To Run (End-to-End)

All commands below assume you are in the repo root and use the `seq` conda env.

```bash
export CONDA_NO_PLUGINS=true
H5AD=~/nvme/all_progenitors2.h5ad
TRC=neuroRef.csv
OUT=scripts/_out/brdu_pipeline
DATE=$(date +%Y%m%d)
CLUSTER_COL=manual_annotation
CLUSTER_A=apical
CLUSTER_B=intermediate
TAG=manual_annotation
SOFT_MODE=soft_logistic_thresholds
SOFT_LOG_BRDU=log_brdu_mean
SOFT_LOG_EDU=log_edu_mean
SOFT_THR_UNS_KEY=brdu_edu_thresholds_by_dataset_roi_ccf_adjusted
SOFT_SPAN=1
SOFT_P_HI=0.99
```

### 9.0a Soft pulse mode (threshold-logistic; no `gmm.npz`)

Current GLM scripts support soft pulse calls directly from your per-unit thresholds in `adata.uns`, using a threshold-centered logistic rule:

- `p=0.5` at the threshold,
- `p=0.01` at `threshold - 1` log unit,
- `p=0.99` at `threshold + 1` log unit.

This corresponds to `--soft-span 1 --soft-p-hi 0.99`.

### 9.0 Baseline continuity by cluster (no gating; recommended context)

This is the baseline continuity, not gene-gated: `f_B1(cluster) = P(E=1 | B=1, cluster)` summarized at the **animal** level.

```bash
conda run -n seq python scripts/brdu_regression/baseline_continuity_by_leiden.py \
  --h5ad "$H5AD" \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_A" "$CLUSTER_B" \
  --include-b0 \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/baseline_continuity_${DATE}"
```

Primary output: `.../baseline_summary_by_leiden.csv` (plus per-animal table `.../baseline_by_animal.csv`).

### 9.1 Mode A discovery scan (best-q per gene)

```bash
conda run -n seq python scripts/scan_ts_brdu_first_gate_mode_a.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}" \
  --phase-matched --phase-matched-method residual
```

Primary output: `$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/ts_scan_gate_mode_a_combined_bestq.csv`

### 9.1b Screening principle (read this before running any GLM)

The intended workflow is:

1. **Screen** genome-wide (or panel-wide) with Mode A + MH summaries to find candidates and identify obvious confounds (Leiden markers, entry/propensity genes, sex/stress axes).
2. **Only then** run the grouped-binomial stratum FE GLM (and any spatially-conditioned variants) on a small shortlist for confirmation and for stability checks.

This keeps iteration time short and avoids spending compute on genes that are clearly “entry/propensity-dominated” (large E3) or essentially type markers.

### 9.2 Animal robustness filter for non-neuroRef genes

```bash
conda run -n seq python scripts/brdu_regression/panel_crossfit_bestq_glm.py \
  --h5ad "$H5AD" --tricycle-ref-csv "$TRC" \
  --pulse-call-mode "$SOFT_MODE" \
  --soft-log-brdu-col "$SOFT_LOG_BRDU" \
  --soft-log-edu-col "$SOFT_LOG_EDU" \
  --soft-threshold-uns-key "$SOFT_THR_UNS_KEY" \
  --soft-span "$SOFT_SPAN" \
  --soft-p-hi "$SOFT_P_HI" \
  --pulse-pmax-min 0.0 \
  --pulse-min-eff-mass 3 \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_A" \
  --theta-bin-mode quantile \
  --q-grid 0.8 0.9 0.95 \
  --select-metric glm_abs_beta_gxb \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_A}_${DATE}_support"

conda run -n seq python scripts/brdu_regression/panel_crossfit_bestq_glm.py \
  --h5ad "$H5AD" --tricycle-ref-csv "$TRC" \
  --pulse-call-mode "$SOFT_MODE" \
  --soft-log-brdu-col "$SOFT_LOG_BRDU" \
  --soft-log-edu-col "$SOFT_LOG_EDU" \
  --soft-threshold-uns-key "$SOFT_THR_UNS_KEY" \
  --soft-span "$SOFT_SPAN" \
  --soft-p-hi "$SOFT_P_HI" \
  --pulse-pmax-min 0.0 \
  --pulse-min-eff-mass 3 \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_B" \
  --theta-bin-mode quantile \
  --q-grid 0.8 0.9 0.95 \
  --select-metric glm_abs_beta_gxb \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_B}_${DATE}_support"
```

Primary output per run: `.../panel_crossfit_by_gene.csv` (rankable table with stability + evaluability QC; `q_selected_mode` is the per-gene tuned `q`).

Notes:

- Run one Leiden at a time (avoid oversaturating CPU); do not rely on parallel runs for throughput tracking.
- Defaults include gene×dataset support filtering; override only if you know you want to allow sparse genes:
  - `--support-min-detected-gatepos`, `--support-min-frac-detected-in-gatepos`, `--support-min-n-datasets-kept`, `--no-support-filter`.

### 9.3 Shortlist characterization (primary; per cluster, then optional joint refit)

```bash
conda run -n seq python scripts/brdu_regression/shortlist_glm_by_animal_meta.py \
  --h5ad "$H5AD" --tricycle-ref-csv "$TRC" \
  --pulse-call-mode "$SOFT_MODE" \
  --soft-log-brdu-col "$SOFT_LOG_BRDU" \
  --soft-log-edu-col "$SOFT_LOG_EDU" \
  --soft-threshold-uns-key "$SOFT_THR_UNS_KEY" \
  --soft-span "$SOFT_SPAN" \
  --soft-p-hi "$SOFT_P_HI" \
  --pulse-pmax-min 0.0 \
  --pulse-min-eff-mass 3 \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_A" \
  --panel-screen-by-gene-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_A}_${DATE}_support/panel_crossfit_by_gene.csv" \
  --top-k 15 \
  --theta-exclude-tested-genes \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_${CLUSTER_A}_${DATE}"

conda run -n seq python scripts/brdu_regression/shortlist_glm_by_animal_meta.py \
  --h5ad "$H5AD" --tricycle-ref-csv "$TRC" \
  --pulse-call-mode "$SOFT_MODE" \
  --soft-log-brdu-col "$SOFT_LOG_BRDU" \
  --soft-log-edu-col "$SOFT_LOG_EDU" \
  --soft-threshold-uns-key "$SOFT_THR_UNS_KEY" \
  --soft-span "$SOFT_SPAN" \
  --soft-p-hi "$SOFT_P_HI" \
  --pulse-pmax-min 0.0 \
  --pulse-min-eff-mass 3 \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_B" \
  --panel-screen-by-gene-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_B}_${DATE}_support/panel_crossfit_by_gene.csv" \
  --top-k 15 \
  --theta-exclude-tested-genes \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_${CLUSTER_B}_${DATE}"
```

Optional joint heterogeneity-friendly refit (two clusters in one per-animal model, cluster-specific slopes; fixed `q` per gene from the screen table used for selection):

```bash
conda run -n seq python scripts/brdu_regression/shortlist_glm_by_animal_meta.py \
  --h5ad "$H5AD" --tricycle-ref-csv "$TRC" \
  --pulse-call-mode "$SOFT_MODE" \
  --soft-log-brdu-col "$SOFT_LOG_BRDU" \
  --soft-log-edu-col "$SOFT_LOG_EDU" \
  --soft-threshold-uns-key "$SOFT_THR_UNS_KEY" \
  --soft-span "$SOFT_SPAN" \
  --soft-p-hi "$SOFT_P_HI" \
  --pulse-pmax-min 0.0 \
  --pulse-min-eff-mass 3 \
  --leiden-col "$CLUSTER_COL" \
  --include-leiden "$CLUSTER_A" "$CLUSTER_B" \
  --joint-leiden-slopes \
  --joint-contrast-leiden "$CLUSTER_B" "$CLUSTER_A" \
  --panel-screen-by-gene-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_A}_${DATE}_support/panel_crossfit_by_gene.csv" \
  --top-k 15 \
  --theta-exclude-tested-genes \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_joint_${CLUSTER_B}_from_${CLUSTER_A}_${DATE}"
```

Primary outputs: `.../shortlist_glm_meta.csv` (meta summaries) and `.../shortlist_glm_by_animal.csv` (per-animal values; includes `Δ1/Δ0/ΔINT` and `T_S/Δt` companions).

### 9.3b Consultant-facing panel report (GLM; cluster-stratified)

This produces a single markdown report that summarizes the **current** cluster-stratified GLM screen results and (optionally) the shortlist and joint refit.

```bash
conda run -n seq python scripts/brdu_regression/make_consultant_leiden_stratified_report.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --bestq-csv "$OUT/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv" \
  --gate-method residual \
  --out-csv "$OUT/ts_scan_gate_mode_a_phase_matched/non_neuroref_by_animal_effects.csv"
```

### 9.3 Phase-bin consistency (select top-N genes with their best q)

```bash
conda run -n seq python scripts/mh_phase_consistency_mode_a.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --scan-csv "$OUT/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2" \
  --n-genes 50 \
  --strata dataset_theta_leiden \
  --exclude-genes Cnksr2
```

Quadrant validation (MH endpoints; optional QC on a small gene list):

```bash
conda run -n seq python scripts/mh_quadrant_validation.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2/mh_phase_consistency_top_genes.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2" \
  --theta-bin-mode quantile
```

Primary output: `.../quadrant_validation.csv`

### 9.5 Leiden confounding diagnostics (gate↔Leiden association + naive vs matched effects)

Example below uses the Leiden 14 shortlist table as a `gene,q` list; substitute the Leiden 15 shortlist (`.../shortlist_glm_meta_leiden15_${DATE}/shortlist_glm_meta.csv`) as needed.

```bash
conda run -n seq python scripts/leiden_confound_diagnostics.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_leiden14_${DATE}/shortlist_glm_meta.csv" \
  --theta-bin-mode quantile \
  --out-csv "$OUT/ts_scan_gate_mode_a_phase_matched/leiden_confound_diagnostics_no_cnksr2.csv"
```

### 9.6 Minimal validation scorecard (replication/sensitivity/permutation)

```bash
conda run -n seq python scripts/validate_candidates.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_leiden14_${DATE}/shortlist_glm_meta.csv" \
  --theta-bin-mode quantile \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/gene_validation_suite_no_cnksr2"
```

Primary output: `.../gene_validation_scorecard.csv`

### 9.6b Optional: posterior-confidence sensitivity (replaces B/E threshold sweeps)

Because inference now uses posterior quadrants instead of hard calls, sensitivity is run by filtering on posterior certainty (`p_max`) and re-running the same soft-label pipeline:

- all cells (`--pulse-pmax-min 0.0`)
- moderate confidence (`--pulse-pmax-min 0.8`)
- high confidence (`--pulse-pmax-min 0.9`)

using a fixed pulse-call configuration (here: threshold-logistic from `adata.uns`). This tests robustness to ambiguous pulse calls without changing the gate or GLM model.

### 9.6c Optional: spatial decomposition (failure-modes #16/#17)

To separate “composition” from “within-location” effects without mixing in support/weighting artifacts from changing strata, run the support-aligned spatial decomposition:

```bash
conda run -n seq python scripts/brdu_regression/spatial_decomposition_standardized.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_leiden14_${DATE}/shortlist_glm_meta.csv" \
  --theta-exclude-tested-genes \
  --leiden 6,14,15 \
  --bin-um 500 \
  --orientation-stratify \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/spatial_decomp_top50_bin500"
```

Primary output: `.../spatial_decomposition_standardized.csv` (includes raw `delta_comp_raw` and support-aligned `delta_comp_std_support`).

### 9.7 Consultant executive summary table (adds `rd_*` / `ΔINT` columns)

```bash
conda run -n seq python scripts/make_consultant_exec_summary.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --theta-bin-mode quantile \
  --quadrant-csv "$OUT/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2/quadrant_validation.csv" \
  --scorecard-csv "$OUT/ts_scan_gate_mode_a_phase_matched/gene_validation_suite_no_cnksr2/gene_validation_scorecard.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50"
```

Primary output: `.../consultant_exec_summary.csv`

### 9.7b Optional: confirm a shortlist with the stratum FE GLM (not a screening step)

Run this only for a small candidate list (e.g., 10–50 genes) after reviewing `consultant_exec_summary.csv`.

```bash
conda run -n seq python scripts/auditfix2_stratum_glm.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50/consultant_exec_summary.csv" \
  --n-genes 30 \
  --q 0.95 \
  --exclude-genes Cnksr2 \
  --outdir "$OUT/brdu_persistence_auditfix_v2_stratum_glm"
```

Note: the GLM script currently uses a single `--q` for all genes; for gene-specific q confirmation prefer the MH-based tables (`mh_quadrant_validation.py` / `make_consultant_exec_summary.py`) which respect per-gene q from the best-q scan.

### 9.8 Consultant 6-gene text reports (panel-constrained)

```bash
conda run -n seq python scripts/make_consultant_gene_reports.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --theta-bin-mode quantile \
  --q-csv "$OUT/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv" \
  --animal-csv "$OUT/ts_scan_gate_mode_a_phase_matched/non_neuroref_by_animal_effects.csv" \
  --outdir "$OUT/consultant_gene_reports"
```

### 9.9 Theta-bin mode verification (quantile vs angle; optional)

```bash
conda run -n seq python scripts/make_consultant_exec_summary.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --theta-bin-mode angle \
  --quadrant-csv "$OUT/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2/quadrant_validation.csv" \
  --scorecard-csv "$OUT/ts_scan_gate_mode_a_phase_matched/gene_validation_suite_no_cnksr2/gene_validation_scorecard.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50_anglebins"

conda run -n seq python scripts/verify_theta_bin_mode.py \
  --quantile-csv "$OUT/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50/consultant_exec_summary.csv" \
  --angle-csv "$OUT/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50_anglebins/consultant_exec_summary.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/theta_bin_mode_verify_top50"
```
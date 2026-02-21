# Methods: BrdU-first / EdU-second interaction phenotypes (panel-constrained)

This document defines the **mechanistic endpoints** and **interaction phenotypes** used in our BrdU-first / EdU-second analysis.

Canonical input: `~/nvme/all_progenitors.h5ad` (targeted panel).

## 1) Data, cohort, and conventions

- Analysis compartment: `all_progenitors.h5ad` contains progenitors; in practice `obs["leiden"] ∈ {6, 14, 15}` for this dataset.
- Binary pulse labels:
  - `B = obs["brdu_pos"]` (0/1).
  - `E = obs["edu_pos"]` (0/1).
- Grouping variables:
  - `dataset = obs["dataset"]` (string).
  - `animal`: parsed from dataset string as `JaxA#`.
  - `orientation`: inferred from dataset string (`Sag` if contains `"Sag"`, else `Coro` if contains `"Coro"`).
- Spatial coordinates:
  - `obsm["AP_ML_um"]` in microns (note upstream AP/ML labels are swapped).
  - throughout we use:
    - `AP_um = obsm["AP_ML_um"][:, 0]`
    - `ML_um = obsm["AP_ML_um"][:, 1]`
  - 1D spatial conditioning uses AP for Sag datasets and ML for Coro datasets.

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
- Set `G_{ig}=1` for the **top `k_d`** cells by `ε_{ig}` within `d`, using deterministic tie-breaking (stable jitter) so the achieved gate fraction is well-defined under ties / zero inflation.

`q` is gene-specific (from the Mode A “best-q” scan table or from an explicit sensitivity run).

**Important design constraint (Fix #2):** gates are *not* defined within microscopic `dataset×theta_bin×leiden` slices. Phase/type matching belongs in the **model strata**, not in the gate. For zero-inflated targeted panels, forcing “top 5% per theta bin” manufactures false “gene+” cells in bins where the gene is truly off.

## 4) Mechanistic endpoints from the full 2×2 BrdU/EdU table

Each cell belongs to one of four quadrants:

- A: Dual (`B=1`, `E=1`)
- B: BrdU-only (`B=1`, `E=0`)
- C: EdU-only (`B=0`, `E=1`)
- D: Double-negative (`B=0`, `E=0`)

We estimate four matched conditional ORs for each gate:

### E1: within BrdU+ (Dual vs BrdU-only)

`OR_E1 = OR_{GE | B=1} = odds(E=1 | G=1, B=1, matched) / odds(E=1 | G=0, B=1, matched)`

### E3: within BrdU− (EdU-only vs Double-negative)

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

### 5.3 Probability-scale interaction (Fix #5; ceiling-aware)

Odds ratios can look extreme when `P(E=1 | B=1, …)` is near saturation. We therefore also report an additive interaction on the probability scale using the same matched strata:

- `Δ1 = P(E=1 | G=1, B=1, matched) − P(E=1 | G=0, B=1, matched)`
- `Δ0 = P(E=1 | G=1, B=0, matched) − P(E=1 | G=0, B=0, matched)`
- `ΔINT = Δ1 − Δ0`

We pool these contrasts trial-weighted within each dataset and compute 95% CIs via dataset bootstrap (same resampling unit as the MH ORs).

Interpretation and “what is a real kinetics-like gene” criteria are in `scripts/brdu_regression/biology-analysis.md` (kept out of this methods document).

## 6) Estimators and uncertainty

### 6.1 Mantel–Haenszel (MH) endpoint tables (fast; phase/type matched)

We compute E1–E4 and IntE/IntB via MH pooled ORs with strict strata:

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
- Phase-consistency and quadrant validation (MH endpoints): `scripts/mh_phase_consistency_mode_a.py`, `scripts/mh_quadrant_validation.py`
- Consultant exec summary tables: `scripts/make_consultant_exec_summary.py`
- Theta-bin mode sensitivity (quantile vs angle): `scripts/verify_theta_bin_mode.py`
- Leiden gate-vs-type diagnostics: `scripts/leiden_confound_diagnostics.py`
- GLM interaction estimator and spatial-adjusted variant: `scripts/auditfix2_stratum_glm.py`, `scripts/auditfix2_spatial_adjusted_dataset_sliced_glm.py`

## 9) How To Run (End-to-End)

All commands below assume you are in the repo root and use the `seq` conda env.

```bash
export CONDA_NO_PLUGINS=true
H5AD=~/nvme/all_progenitors.h5ad
TRC=neuroRef.csv
OUT=scripts/_out/brdu_pipeline
```

### 9.1 Mode A discovery scan (best-q per gene)

```bash
conda run -n seq python scripts/scan_ts_brdu_first_gate_mode_a.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched" \
  --phase-matched --phase-matched-method residual
```

Primary output: `$OUT/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv`

### 9.1b Screening principle (read this before running any GLM)

The intended workflow is:

1. **Screen** genome-wide (or panel-wide) with Mode A + MH summaries to find candidates and identify obvious confounds (Leiden markers, entry/propensity genes, sex/stress axes).
2. **Only then** run the grouped-binomial stratum FE GLM (and any spatially-conditioned variants) on a small shortlist for confirmation and for stability checks.

This keeps iteration time short and avoids spending compute on genes that are clearly “entry/propensity-dominated” (large E3) or essentially type markers.

### 9.2 Animal robustness filter for non-neuroRef genes

```bash
conda run -n seq python scripts/report_non_neuroref_effects_by_animal.py \
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
  --animal-csv "$OUT/ts_scan_gate_mode_a_phase_matched/non_neuroref_by_animal_effects.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2" \
  --n-genes 50 \
  --strata dataset_theta_leiden \
  --exclude-genes Cnksr2
```

Primary output: `.../mh_phase_consistency_top_genes.csv` (columns: `gene,q`)

### 9.4 Quadrant validation (E1–E4, IntE/IntB) + probability-scale `ΔINT`

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

```bash
conda run -n seq python scripts/leiden_confound_diagnostics.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2/mh_phase_consistency_top_genes.csv" \
  --theta-bin-mode quantile \
  --out-csv "$OUT/ts_scan_gate_mode_a_phase_matched/leiden_confound_diagnostics_no_cnksr2.csv"
```

### 9.6 Minimal validation scorecard (replication/sensitivity/permutation)

```bash
conda run -n seq python scripts/validate_candidates.py \
  --h5ad "$H5AD" \
  --tricycle-ref-csv "$TRC" \
  --genes-csv "$OUT/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2/mh_phase_consistency_top_genes.csv" \
  --theta-bin-mode quantile \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched/gene_validation_suite_no_cnksr2"
```

Primary output: `.../gene_validation_scorecard.csv`

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

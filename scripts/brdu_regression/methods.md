# Methods: BrdU-first / EdU-second interaction phenotypes (panel-constrained)

This document defines the **mechanistic endpoints** and **interaction phenotypes** used in our BrdU-first / EdU-second analysis.

Canonical input: `~/nvme/all_progenitors2.h5ad` (targeted panel; includes `obs['manual_annotation']`).

## Executive overview: estimands and reading guide

**Goal.** Identify gene-defined transcriptional states that are associated with **S-phase continuity across a fixed lag Δt** (BrdU at `t=0`, EdU at `t=Δt`), while separating that from a broader **EdU propensity / S-entry** program.

**Notation.** `B` and `E` are latent BrdU/EdU labels with per-cell joint posteriors
`(pi_00, pi_01, pi_10, pi_11) = P(B=b,E=e | intensities)`, and `G` is a gene-high gate. Within matched strata
`s = dataset × leiden × theta_bin` (and optionally `× spatial_bin`), each cell falls in a 2×2 table:

```
                E=0        E=1
B=0 (BrdU−)     q00        q01
B=1 (BrdU+)     q10        q11
```

**Probability-scale endpoints (mechanistic axes).**

- **Continuity / retention over Δt:**
  `ρ_g = P(E=1 | B=1, G=g, matched)` and `Δ1 = ρ_1 − ρ_0`.
- **Recruitment / entry over the window:**
  `η_g = P(E=1 | B=0, G=g, matched)` and `Δ0 = η_1 − η_0`.

**Specificity diagnostic (interaction).**

- **Additive:** `ΔINT = Δ1 − Δ0`
- **Odds-scale:** `IntE = log(OR_E1) − log(OR_E3)`, where `OR_E1` is the `G`–`E` odds ratio within `B=1` and `OR_E3` is the same within `B=0`.

Interpretation:

- `Δ1` (or E1 / `OR_E1`) is the “elongation/retention marker” readout.
- `IntE` / `ΔINT` asks whether that retention association is **BrdU-conditioned** (continuity-specific) versus simply mirroring general EdU propensity (`Δ0`/E3).

**Estimation layer (what is treated as primary).**

- **QC / fast screening:** gene×dataset support QC and probability-scale endpoints (`Δ1/Δ0/ΔINT`) from the same soft-label GLM layer (plus RD anchors for quick sanity checks).
- **Primary estimand / inference:** grouped-binomial stratum fixed-effect GLM
  `logit P(E=1) = α_s + β_G G + β_B B + β_GB (G×B)` fit on **effective successes/trials** from posterior quadrants:
  `n_eff(s,G,B)` and `y_eff(s,G,B)` (expected complete-data likelihood).
  Mapping is unchanged: `β_G ≈ log(OR_E3)`, `β_G + β_GB ≈ log(OR_E1)`, `β_GB = IntE`.
- **Overlap support (identifiability):** replace “all four (G,B) cells exist” with effective-mass support
  `n_eff(s,G,B) ≥ m` for all four cells (default `m=3`).

**Replication unit.** Treat `animal` as the independence unit. Panel screens use LOAO cross-fit; shortlist reporting uses per-animal fits/meta summaries.

Reading guide: Sections 1–5 define inputs, gates, strata, and endpoints; section 6 covers estimators/uncertainty; section 9 is the runbook.

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

**θ leakage sensitivity (Fix #12):** since θ is computed from gene expression, gate construction can be entangled with θ if a candidate gene is in (or tightly correlated with) the reference gene set. For shortlist sensitivity runs, exclude tested genes from the θ reference set via `--theta-exclude-tested-genes` (or `--theta-exclude-genes ...`) in `mh_quadrant_validation.py` / `make_consultant_exec_summary.py` (and related shortlist scripts).

## 2.1 Single-Δt interpretation: “transit” vs “entry” vs phase positioning

The experiment has a **single time lag (Δt)** between BrdU and EdU pulses. With a single Δt, an apparent “BrdU-conditioned EdU effect” could in principle reflect:

- true **S-phase retention/transit** differences over Δt, or
- differences in **EdU entry propensity** (general EdU calling), or
- **phase positioning within S** (early vs late) rather than true transit kinetics.

Our primary design choice is to report a **phase- and state-conditioned estimand**: all comparisons are within matched `dataset × theta_bin × leiden` strata, i.e. within the same **phase-position slice** (θ) and **cell-state slice** (Leiden). This reduces sensitivity to phase-position confounding, but it is not “ground truth” in a causal sense: conditioning on θ changes the question being asked.

In particular, if a gene-defined state (gate `G`) causally shifts phase position (θ), then θ matching will condition away that pathway. In that case, the θ-matched interaction is best interpreted as **“BrdU-conditioned EdU association at fixed phase position and cell state”**, while the no-θ-matching analysis corresponds to a **marginal mixture** over phase positions.

For shortlist/narrative genes we also include an explicit ablation:

- **Remove θ matching** (drop `theta_bin` from strata; match only `dataset × leiden`): effects can inflate or change if phase-position differences contribute (as confounding and/or mediation).
- **Keep θ matching** (default): transit/retention markers should persist if the phenotype is not explained by phase positioning alone.

We therefore phrase the claim conservatively as: **“predicts S-phase retention over Δt within matched phase position and cell state”**, not “changes S-phase duration.”

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

**Important design constraint (Fix #2):** gates are *not* defined within microscopic `dataset×theta_bin×leiden` slices. Phase/type matching belongs in the **model strata**, not in the gate. For zero-inflated targeted panels, forcing “top 5% per theta bin” manufactures false “gene+” cells in bins where the gene is truly off.

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

## 4.1 Continuity (retention) vs recruitment (entry) over a single Δt

The paired readout is (i) transcriptional state (`G`) plus (ii) a two-timepoint indicator of being in S-phase over a fixed lag `Δt` (BrdU first, EdU second). Within a matched stratum (`dataset × theta_bin × leiden`), two conditional probabilities are directly interpretable as short-horizon flux:

- **S-phase continuity / retention (elongation-relevant):**
  - `ρ(G) = P(E=1 | B=1, G, matched)`
  - probability-scale effect: `Δ1 = P(E=1 | G=1, B=1, matched) − P(E=1 | G=0, B=1, matched)`
  - odds-scale companion: E1 / `OR_E1`.
- **S-phase recruitment / entry over the window:**
  - `η(G) = P(E=1 | B=0, G, matched)`
  - probability-scale effect: `Δ0 = P(E=1 | G=1, B=0, matched) − P(E=1 | G=0, B=0, matched)`
  - odds-scale companion: E3 / `OR_E3` (aka “E3”).

If the biological goal is “S-phase elongation markers,” the cleanest operational headline readout is **`Δ1` (or equivalently E1/`OR_E1`)**, i.e. the within-`B=1` continuity effect after matching for cell state and phase position. Interaction terms (below) then serve as diagnostics for continuity-specificity versus global recruitment/occupancy programs.

## 5) Continuity-specificity interactions (BrdU-conditioned beyond EdU propensity)

### 5.1 Interaction (IntE / logIOR)

On the log scale:

- `IntE = log(OR_E1) - log(OR_E3)`

On the OR scale:

- `IOR = OR_E1 / OR_E3`

Interpretation:

- `IntE < 0` means gene+ has **lower** BrdU-conditioned persistence/progression than expected from its general EdU propensity.
- `IntE > 0` means gene+ has **higher** BrdU-conditioned persistence/progression than expected from its general EdU propensity.

**E3 is not an exclusion criterion by default:** `E3` (≈ GLM `β_G`) is the BrdU−/EdU axis (general EdU propensity / entry / calling). We always report it, but we do not exclude genes solely because `E3` is non-null. A “slow/extended S-like” state can plausibly affect both BrdU+ retention (E1) and BrdU−→EdU+ propensity (E3). When `E3` is large, interpret the phenotype as **mixed** rather than “pure BrdU-conditioned retention.”

### 5.1.1 Interpreting sign and the observed negative skew in panel screening

In our convention, `IntE < 0` means the gate’s EdU association is **more negative / less positive among BrdU+** than among BrdU− after matching (`dataset × theta_bin × leiden`). Operationally, this means the gate’s association with the continuity probability `ρ(G)` is smaller than (or opposite to) its association with the recruitment probability `η(G)` on the odds scale.

We no longer use Mantel–Haenszel (MH) interactions for panel screening; all panel-wide selection/evaluation is GLM-based (soft-label expected-likelihood). We still track the **sign distribution** of the GLM interaction `β_G×B` and always report `Δ1` and `Δ0` alongside `ΔINT`/`IntE` so that global skew does not hide interpretable E1/E3 patterns.

Operationally, we therefore include these QC checks alongside any panel-wide ranking table:

- **Report sign distribution** of `β_G×B` and list the strongest positive-tail genes (to confirm the analysis can produce both signs).
- **Report E1 and E3 separately** (or in GLM terms, report `β_G` and `β_G×B`, and the implied BrdU+ logOR `β_E1 = β_G + β_G×B`) so “all-negative IntE” does not hide that E1 or E3 can still be positive/negative in biologically interpretable ways.
- For shortlist/narrative genes, optionally run a **gate-direction sensitivity** (top‑k vs bottom‑k) to verify that sign behavior is not an artifact of always selecting “high residual expression” cells.

**Why can `β_G×B` be broadly negative in a high-continuity cluster?** Under the GLM parameterization
`logit P(E=1) = α_s + β_G·G + β_B·B + β_G×B·(G×B)`, the gene log-OR among BrdU− is `β_G`, while the gene log-OR among BrdU+ is `β_E1 = β_G + β_G×B`. Therefore `β_G×B = β_E1 − β_G`: a negative `β_G×B` often indicates the gene is **more predictive of EdU positivity in BrdU−** (entry/occupancy) than in BrdU+ (continuity). This is common when `P(E=1|B=1,matched)` is near a ceiling while `P(E=1|B=0,matched)` is rare-event, so the same absolute risk change can correspond to a much larger change in log-odds in `B=0` than in `B=1`. In that regime, a negative interaction logOR is not contradictory to high baseline continuity; it is a reminder to prioritize probability-scale `Δ1` (and report `Δ0`) for mechanistic reading.

### 5.1.2 “Elongation marker” vs “continuity specificity” (what the interaction is for)

When the biological question is “which transcriptional states mark **prolonged S** over a fixed `Δt`,” we treat the within-initial-S continuity effect as the headline:

- **Elongation marker readout:** `Δ1` (and/or E1 / `OR_E1`)

The interaction (`IntE` / GLM `β_G×B`) is then a **continuity-specificity diagnostic**:

- **Continuity specificity:** `ΔINT = Δ1 − Δ0` (or on the odds scale, `IntE = log(OR_E1) − log(OR_E3)`)
  - asks whether the continuity association in `B=1` is specific to the initial-S cohort versus part of a broader recruitment/occupancy program in `B=0`.

Importantly, a gene can have **positive `Δ1` (higher continuity)** while still having **negative `ΔINT`/`IntE`** if `Δ0`/E3 is even larger. For narrative genes we therefore report `Δ1` and `Δ0` alongside interaction diagnostics rather than interpreting interaction sign alone.

### 5.2 Mirror interaction (IntB; diagnostic)

- `IntB = log(OR_E2) - log(OR_E4)`

IntB is used as a robustness check; it is often noisier because the EdU− cohort can be small in some strata.

### 5.3 Probability-scale interaction (Fix #5; ceiling-aware)

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

The **key idea we retain** from that era is the *support-aligned overlap restriction*: do not let strata with missing `(G,B)` support drive interaction summaries. In the current GLM implementation this is enforced via an **effective-mass overlap threshold** `n_eff(s,G,B) ≥ m` for all four `(G,B)` cells (configured by `--pulse-min-eff-mass`).

### 6.2 Grouped-binomial fixed-effect GLM (auditfix v2; interaction coefficient)

IntE can also be estimated directly as the interaction coefficient:

`logit P(E=1) = α_stratum + β_G G + β_B B + β_GB (G×B)`

with `strata = dataset×theta_bin×leiden`.

**Important workflow note:** do not start with a *full* GLM characterization run on a large gene list. Use the screening pipeline first (Mode A scan + **Leiden-stratified GLM LOAO screen with support filtering**) to narrow to a short candidate set, then apply the richer GLM reporting layer (including `Δ1/Δ0/ΔINT` and `T_S/Δt` companions) on that shortlist.

### 6.3 Confirmatory inference within cohort (animal LOAO cross-fit)

For **publishable** claims we treat `animal` as the independence / resampling unit (datasets within an animal are not independent). Our confirmatory workflow therefore uses **leave-one-animal-out (LOAO)** cross-fit:

For each held-out animal `a`:

1. **Tune `q` on training animals** (`≠ a`) by maximizing a pre-specified metric across a small grid `q_grid` (default `[0.8, 0.9, 0.95]`).
   - Default selector: `max_q |β_G×B,GLM(train)|` (aligns selection to the primary estimand; robust in rare-outcome regimes where MH+cc can be biased).
2. **Evaluate on held-out animal** using the selected `q`:
   - Panel-wide: stratum-FE grouped-binomial GLM `β_G×B` on held-out animal (plus `Δ1/Δ0/ΔINT` companions when needed).

We then aggregate the held-out-animal results across LOAO folds (5 animals in this cohort).

- **Panel-wide screen:** treat LOAO GLM results as **prioritization + stability** (mean/SD/sign-consistency across animals, plus q-selection stability). With `n_animals=5`, parametric p-values from “t-test across folds” are too fragile to be the basis of a publishable panel-wide discovery claim.
- **Shortlist / narrative:** treat LOAO GLM (`β_G×B`, and explicitly `β_E1`/`Δ1`) as the primary reporting layer and summarize at the **animal** level (see note below).

Reference implementation: `scripts/brdu_regression/panel_crossfit_bestq_glm.py` (GLM-only).

### 6.4 Cluster-stratified analysis runs (required)

Because `cluster` is part of the matching strata (`dataset×theta_bin×cluster`), panel screens intended for narrative interpretation should be run **separately per cluster** and then compared. Mixing clusters in a single screen can elevate genes that are simply other-cluster markers (or reflect population-definition issues).

Implementation detail: the scripts historically used `leiden` as the cluster label; we now support arbitrary cluster annotations via the CLI option `--leiden-col <obs_column>` (default `leiden`). The `--include-leiden/--exclude-leiden` filters always refer to values of `obs[--leiden-col]`.

`panel_crossfit_bestq_glm.py` supports cluster filtering via:

- `--leiden-col manual_annotation --include-leiden apical` (run within a single manual cluster)
- `--leiden-col manual_annotation --exclude-leiden no` (exclude a cluster)

In the current `~/nvme/all_progenitors2.h5ad`, `obs['manual_annotation']` is the intended cluster field for the consultant packet (e.g. `apical`, `intermediate`).

**Shortlist scope (decision):** the core narrative shortlist is drawn from the two progenitor clusters of interest (e.g. `apical` and `intermediate`) and is **not** pooled with other groups.

### 6.4.1 Optional: joint two-cluster shortlist refit (heterogeneity-friendly; equivalent to separate fits)

For shortlist/narrative genes, we optionally run a **single joint per-animal GLM** that includes *both* Leiden 14 and 15 cells (same gate definition and same stratum FE structure), but allows **Leiden-specific slopes** for `G`, `B`, and `G×B`.

Specifically, with `stratum = dataset × theta_bin × leiden`, we fit within each animal:

`logit P(E=1) = α_stratum + Σ_k [ β_G^(k)·(G·I_k) + β_B^(k)·(B·I_k) + β_G×B^(k)·(G·B·I_k) ]`,

where `k` ranges over the included Leiden labels (typically `{14,15}`) and `I_k = 1[leiden=k]`.

Notes:

- We do **not** include a separate Leiden main effect because Leiden is already absorbed by the stratum intercepts.
- Allowing `β_G^(k)` and `β_B^(k)` to vary by Leiden is important: constraining them to be shared can distort `β_G×B^(k)` when baseline EdU propensity (`E3`) and BrdU effects differ across states.
- This joint model is likelihood-equivalent to fitting separate per-Leiden models, but it enables **paired within-animal contrasts** (e.g. `β_G×B^(15) − β_G×B^(14)`) for heterogeneity diagnostics without relying on a huge-n Wald test.

Implementation: `scripts/brdu_regression/shortlist_glm_by_animal_meta.py --joint-leiden-slopes --leiden-col manual_annotation --include-leiden apical intermediate`.

Outputs: this joint refit reports, per Leiden and as paired within-animal contrasts (15−14), the primary retention readouts:

- `Δ1`, `Δ0`, `ΔINT`
- `p_b1_g0`, `p_b1_g1` and the derived `T_S/Δt` companions
- Cluster-specific gate evaluability QC (datasets kept, within-gate detection, within-gate expression evidence) to help distinguish biology from “one cluster is sparse/noisy”.

### 6.5 Gene evaluability / expression support filtering (recommended defaults)

To reduce false “hits” driven by extremely sparse expression (where a top‑k gate necessarily contains mostly zeros), we apply per gene×dataset evaluability criteria at the chosen `q`:

- `n_detected_gate1 = Σ 1[x_raw > 0 and G=1]` ≥ `--support-min-detected-gatepos` (default `1`)
- `frac_detected_in_gate1 = n_detected_gate1 / Σ 1[G=1]` ≥ `--support-min-frac-detected-in-gatepos` (default `0.0`)
- require at least `--support-min-n-datasets-kept` datasets to pass (default `5`); otherwise treat the gene×q as non-evaluable in that fold.

For traceability, `panel_crossfit_bestq_glm.py` also emits QC columns in `panel_crossfit_by_gene.csv` (at `q_selected_mode`), including:

- `qc_frac_x_gt0`
- `qc_median_x_gatepos`, `qc_p90_x_gatepos`
- `qc_frac_detected_in_gatepos`
- `qc_ds_frac_detected_in_gatepos_min/median/max`
- `qc_support_n_datasets_kept_gatepos`

**Tightened shortlist selection (decision):** for shortlist/narrative gene selection we apply stronger evaluability/evidence filters than the panel-wide defaults:

- require broader support: e.g. `qc_support_n_datasets_kept_gatepos ≥ 10` (configurable)
- require within-gate expression evidence: `qc_median_x_gatepos ≥ 1` **or** `qc_p90_x_gatepos ≥ 2`
- exclude separation-prone genes: `animal_sd_beta_GxB` above a threshold (e.g. `> 1.0`)

Reference implementation: `scripts/brdu_regression/shortlist_glm_by_animal_meta.py` (shortlist selection options `--shortlist-*`).

**Small-`n_animals` inference note:** with only 5 animals, asymptotic GLM SEs can be tiny (many cells per animal) and inverse-variance `z` tests can look overconfident. For shortlist/narrative claims, use the **held-out-animal GLM estimates** and summarize them at the animal level (equal-weight mean, t-based CI/p-value across the 5 held-out animals). `panel_crossfit_bestq_glm.py` emits both the IVW meta summary (`meta_*`) and the animal-level t summary (`animal_*`); treat the animal-level summary as the publishable inference unit.

**Runtime / logging note:** `panel_crossfit_bestq_glm.py` logs progress at the gene and block level. For long runs, `--log-folds` enables per-fold (held-out animal) progress lines so it is obvious where time is being spent.

**Permutation p-values (when used):** report Monte Carlo permutation p-values as `(b+1)/(n_perm+1)` where `b` is the number of permuted statistics at least as extreme as observed, to avoid `p=0` artifacts.

**Permutation calibration for the panel screen (optional):** to calibrate the *screening algorithm as used* (including best-`q` selection), use `scripts/brdu_regression/permutation_calibration_panel_crossfit_mh.py`, which reruns the LOAO q-selection inside each permutation replicate under a stratum-preserving random-labeling null. If this calibration is run on a **post-selected top-k** gene subset, any BH/FDR values are **within that subset only** and must not be interpreted as panel-wide FDR control across the full gene panel.

**Permutation resolution note:** finite `n_perm` implies a minimum attainable nonzero p-value of `1/(n_perm+1)`. Panel-wide BH across `m` genes requires p-values fine enough to resolve `≈ 0.05/m` (for the most significant hits), so small `n_perm` top-k permutation runs are intended as QC/calibration for a shortlist rather than panel-wide discovery inference.

Interpretation:

- `β_G` corresponds to `log(OR_E3)`
- `β_G + β_GB` corresponds to `log(OR_E1)`
- `β_GB` is `IntE` (logIOR)

**Elongation/retention endpoints (decision):** for the “S-phase retention over Δt” story, we explicitly report the BrdU+ gate effect:

- **E1 (odds scale):** `β_E1 = β_G + β_G×B` (i.e., `log(OR_E1)`), as the primary elongation/retention marker readout.
- **Δ1 (probability scale):** `Δ1 = P(E=1|G=1,B=1,matched) − P(E=1|G=0,B=1,matched)` as the reader-facing effect size on the probability scale.

We retain `β_G×B` (interaction / continuity specificity) and `β_G` (E3 axis) as mechanistic context rather than exclusion criteria.

**Relation to the standard `T_S` convention (`BrdU+EdU+ / BrdU+`):** under the common field convention, within an analysis group we define:

- `f = BrdU+EdU+ / BrdU+ = P(E=1 | B=1)`
- `T_S/Δt ≈ 1/(1−f)` (so `T_S ≈ Δt/(1−f)`), for `f<1`.

This is a standard approximation under short pulses and negligible re-entry within `Δt`; treat it as a convention-based summary (especially when `f→1`).

In the shortlist GLM layer we therefore report (within matched strata, trial-weighted g-computation under the fitted GLM):

- `p_b1_g0 = P(E=1 | B=1, G=0, matched)` and `p_b1_g1 = P(E=1 | B=1, G=1, matched)`
- the corresponding dimensionless `T_S/Δt` summaries:
  - `T_S/Δt (G=0) = 1/(1−p_b1_g0)`
  - `T_S/Δt (G=1) = 1/(1−p_b1_g1)`
  - `Δ(T_S/Δt) = (T_S/Δt)(G=1) − (T_S/Δt)(G=0)`

Because this transform is nonlinear and can become unstable when `f→1`, we treat `Δ1` as the primary retention readout and use `T_S/Δt` as a standard-convention companion.

The grouped-binomial implementation (aggregating counts by stratum and `(G,B)`) is likelihood-equivalent to cell-level Bernoulli FE GLM and avoids sparse-table continuity correction artifacts.

**Legacy MH vs GLM divergence QC (optional; comparison only):** earlier versions compared the primary GLM interaction (`β_GB`) to an overlap-restricted MH interaction (`IntE_overlap`). We do **not** run MH in the current soft-call workflow; this check applies only if you also generate the legacy MH tables for comparison. In `scripts/brdu_regression/make_publishable_shortlist_table.py` (and any executive-summary tables built from it), the divergence fields include:

- `abs_glm_minus_mh_intE_overlap = |β_GB − IntE_overlap|`
- `glm_mh_intE_overlap_sign_mismatch`
- `flag_glm_vs_mh_overlap_divergent` (defaults to sign mismatch or `abs_glm_minus_mh_intE_overlap > 0.2`; configurable via `--glm-mh-divergence-abs`)

Interpretation: a divergence flag means the interaction estimate is **estimator-sensitive**. Common causes are (i) strong stratum-level heterogeneity that violates the GLM’s shared `β_B` structure, and/or (ii) different weighting/sparsity behavior between MH pooling and GLM likelihood. Treat these genes as “needs deeper modeling / stratified review,” not as stable kinetics-like calls from a single number.

In particular, the MH interaction is computed as a **difference of two separately pooled log-ORs** (E1 pooled within `B=1`, E3 pooled within `B=0`), which generally will not exactly equal the single-model interaction parameter under non-collapsibility/heterogeneity. Additionally, MH requires a continuity correction in sparse strata (`--cc`, default Haldane–Anscombe `0.5`), while the GLM uses the binomial likelihood; with rare outcomes these can differ materially even when the underlying signal is the same.

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

**Standardization note (Fix #16/#17):** a naive `δ_comp = δ_marg − δ_space` can mix in estimator artifacts because changing strata changes support/weights (and `spatial_bin` is orientation-dependent). For spatial follow-up on a shortlist, use `scripts/brdu_regression/spatial_decomposition_standardized.py`, which reports both the raw difference and a support-aligned variant that computes the marginal estimand after restricting to fine spatial strata with B-overlap; optionally report results stratified by orientation.

## 8) Reference implementations (scripts)

- Gate scan / best-q: `scripts/scan_ts_brdu_first_gate_mode_a.py`
- Pulse measurement layer (batch-aware GMM; writes `cache/per_cell_qc.npz`): `scripts/gam/qc_batch_aware_brdu_edu_gmm.py`
- Legacy hard-call MH phase-consistency + quadrant validation (comparison only): `scripts/mh_phase_consistency_mode_a.py`, `scripts/mh_quadrant_validation.py`
- Consultant exec summary tables: `scripts/make_consultant_exec_summary.py`
- Panel-wide LOAO GLM screen (cluster-stratified via `--leiden-col`): `scripts/brdu_regression/panel_crossfit_bestq_glm.py`
- Shortlist GLM meta + optional joint 14+15 refit: `scripts/brdu_regression/shortlist_glm_by_animal_meta.py`
- Consultant-facing cluster-stratified panel report: `scripts/brdu_regression/make_consultant_leiden_stratified_report.py`
- Theta-bin mode sensitivity (quantile vs angle): `scripts/verify_theta_bin_mode.py`
- Leiden gate-vs-type diagnostics: `scripts/leiden_confound_diagnostics.py`
- GLM interaction estimator and spatial-adjusted variant: `scripts/auditfix2_stratum_glm.py`, `scripts/auditfix2_spatial_adjusted_dataset_sliced_glm.py`
- Legacy B/E calling threshold sweep (retired under mixture model): `scripts/brdu_regression/sensitivity_be_thresholds.py`
- Legacy hard-call spatial decomposition (MH + RD; comparison only): `scripts/brdu_regression/spatial_decomposition_standardized.py`
- Cluster-level baseline continuity (no gating): `scripts/brdu_regression/baseline_continuity_by_leiden.py`
- Shortlist inference sensitivity suite runner: `scripts/brdu_regression/run_shortlist_inference_suite.py`

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

1. **Screen** panel-wide with Mode A + **cluster-stratified LOAO GLM** (plus support/evaluability filters) to find candidates and identify obvious confounds (type markers, sparsity-driven “hits”, instability across animals).
2. **Only then** run the richer shortlist reporting layer (per-animal GLM meta outputs including `β_E1/Δ1` and, when needed, `T_S/Δt` companions; plus the optional joint 14+15 heterogeneity refit).

This keeps iteration time short and avoids spending compute on genes that are essentially type markers or that fail basic expression/evaluability evidence in the gate. We do **not** exclude genes solely for having a non-null E3; E3 is used to *classify* phenotypes rather than filter them out.

### 9.2 Panel-wide LOAO GLM screen (primary; run per cluster)

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
  --leiden-col "$CLUSTER_COL" \
  --baseline-leiden-labels "$CLUSTER_A" "$CLUSTER_B" \
  --cluster-a-name "Apical" \
  --cluster-b-name "Intermediate" \
  --cluster-a-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_A}_${DATE}_support/panel_crossfit_by_gene.csv" \
  --cluster-b-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/panel_crossfit_glm_bestq_${CLUSTER_B}_${DATE}_support/panel_crossfit_by_gene.csv" \
  --shortlist-a-meta-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_${CLUSTER_A}_${DATE}/shortlist_glm_meta.csv" \
  --shortlist-b-meta-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_${CLUSTER_B}_${DATE}/shortlist_glm_meta.csv" \
  --joint-ab-meta-csv "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/shortlist_glm_meta_joint_${CLUSTER_B}_from_${CLUSTER_A}_${DATE}/shortlist_glm_meta.csv" \
  --outdir "$OUT/ts_scan_gate_mode_a_phase_matched_${TAG}/consultant_panel_report_glm_bestq_${CLUSTER_COL}_${CLUSTER_A}_${CLUSTER_B}_${DATE}"
```

Primary outputs: `.../consultant_panel_report.md` and `.../consultant_panel_report_update_only.md` (if `--updates-only` is used).

### 9.4 Legacy MH-only top-N selection and MH quadrant tables (QC only)

These steps are retained as MH endpoint/QC utilities. They are **not** the primary panel-wide screen under rare-EdU regimes.

Phase-bin consistency top-N (MH endpoints; outputs `gene,q` list):

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

**θ ablations (Fix #13):** to test θ entanglement, rerun with:

- `--no-gate-theta-residualize` (gate residualization off, θ matching on)
- `--no-theta-matching` (θ matching off, gate residualization on/off)

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

This is a **legacy MH-centric** executive-summary table builder. For the current GLM-first workflow, prefer the cluster-stratified report in **9.3b**.

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

If you ran additional MH cross-fit diagnostics (legacy), you can attach them here via `--crossfit-bestq-by-gene-csv ...`.

For final-shortlist review, the table reports **`intE_primary`** as the stratum-FE GLM interaction coefficient (`glm_beta_GxB`). If legacy MH tables are also provided, it can optionally include MH-vs-GLM comparison fields (e.g. `*_overlap`), but MH is not part of the current soft-call workflow.

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

Legacy utility; requires older “non_neuroref_by_animal_effects.csv” inputs. Prefer the GLM cluster-stratified report (9.3b) and/or the shortlist GLM meta tables.

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

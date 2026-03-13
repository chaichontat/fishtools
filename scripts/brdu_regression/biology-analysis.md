# BrdU -> EdU (BrdU-first) biological interpretation: mechanism-aware endpoints

This document is a companion to `scripts/brdu_regression/methods.md`. It explains the biological meaning of the **four BrdU/EdU quadrants** and the derived interaction phenotype, and it defines what we mean by a “real kinetics-like gene” in a way that is testable with this dataset.

## Executive overview: what the dual pulse supports

This dual-pulse design measures, over a fixed lag `Δt`, whether a transcriptional state is associated with:

- **continuity/retention**: being in S at `t=0` (BrdU+) and still in S at `t=Δt` (EdU+), and/or
- **recruitment/entry**: entering S between pulses (EdU+ among BrdU− cells),

after conditioning on cell state (Leiden) and phase position (`θ` bin). The analysis is **associational** and should be described as “predicts/associates with continuity over `Δt`,” not as a causal change in S-phase time.

Operationally, we report two axes:

- the **BrdU+ continuity effect** (`Δ1` / E1) as the retention/elongation marker readout, and
- the **interaction** (`IntE` / `ΔINT`) as a continuity-specificity diagnostic relative to the BrdU− entry/propensity axis (E3 / `Δ0`).

In this project, “kinetics-like” means a reproducible **BrdU-conditioned continuity phenotype** in the 2×2 table—not a full kinetic model of the cell cycle.

## Statistical footing and limits (read before narrative claims)

What is statistically well-founded in the current design:

- **Clear estimands under explicit conditioning:** effects are defined within `dataset×leiden×theta_bin` strata, i.e. association of EdU positivity at `t=Δt` with a transcriptional gate **at fixed cell state and phase-position slice**.
- **Two-axis decomposition:** reporting both `Δ1` (BrdU+ continuity/retention) and `Δ0` (BrdU− EdU propensity / entry axis) prevents over-interpreting a single interaction number as “kinetics”.
- **Replication unit:** treating `animal` as the independence unit and using LOAO for screening stability is the right response to within-animal pseudo-replication.
- **Probability-scale companions:** OR interactions can be unstable under saturation/rare events; `Δ1/Δ0/ΔINT` provide interpretable effect sizes and reduce estimator pathologies.

Core limitations that materially constrain inference:

1. **Single `Δt` limits mechanistic identification:** retention/transit, entry propensity, and within‑S positioning can map to the same 2×2 table. θ matching reduces (but does not eliminate) positioning confounding and also changes the estimand from marginal to conditional.
2. **θ is expression-derived:** conditioning on θ is a choice of conditional estimand, but can block θ-mediated pathways (mediation) and can in principle introduce collider-like artifacts; θ ablations are required for narrative genes.
3. **Rare events / sparse support:** when EdU is rare in `B=0`, OR-based E3/IntE can be dominated by sparse strata and continuity corrections; prefer `Δ1` and model-implied `p_b1_g*`, and use overlap restriction as OR QC.
4. **Fixed-effects logistic is a modeling choice:** `β_G×B` is a weighting-dependent average under slope heterogeneity, and with sparse strata can be separation-prone; treat per-animal summaries/heterogeneity diagnostics as first-class outputs.
5. **Selection/multiplicity:** with `n_animals=5`, “cell-level” p-values are not a publishable discovery layer for panel-wide screening; the credible evidence is between-animal replication and pre-specified robustness checks.

High-leverage practice (project conventions):

- **Primary effect size:** make `Δ1` (plus `p_b1_g0/p_b1_g1`) the headline retention readout; treat `β_G×B` / `IntE` as a continuity-specificity diagnostic.
- **Separation sensitivity (shortlist):** compare to a penalized estimator (e.g. Firth) per animal as a flag for “infinite logOR” artifacts (not a replacement for the primary report; implement as a sensitivity check).
- **Avoid absolute `T_S` language under θ matching:** keep `T_S/Δt` as a conventional rescaling of retention probability, not a literal duration estimate without additional assumptions or multiple `Δt`.

## 1) What can we claim from this assay?

BrdU is administered first (`t=0`) and EdU second (`t=Δt`). In the idealized model:

- BrdU+ marks cells in S at `t=0`.
- EdU+ marks cells in S at `t=Δt`.

In real targeted-panel data, the BrdU/EdU table also reflects:

- entry into S between pulses (and exit from S),
- analogue availability windows,
- and label calling / incorporation artifacts.

Therefore, when we use the word “kinetics” here, we mean:

> a reproducible **S-phase continuity / retention over Δt** phenotype in the dual-pulse table, which under the standard field convention supports an operational estimate of **S-phase time (T_S)** from `BrdU+EdU+ / BrdU+` (with the usual assumptions noted below).

## 2) The four quadrants and what each conditional endpoint diagnoses

Let `B` and `E` be the latent BrdU/EdU pulse states, with per-cell quadrant posteriors
(`pi_00`, `pi_01`, `pi_10`, `pi_11`) provided by the mixture model; and let `G` be a gene-high gate. Hard calls
(`brdu_pos`, `edu_pos`) are retained only for QC/debug comparisons.

Each cell has a latent quadrant (unknown); the mixture model provides posterior mass over quadrants. For exposition, we label quadrants by their `(B,E)` bits:

- `q11`: Dual (`B=1`, `E=1`)
- `q10`: BrdU-only (`B=1`, `E=0`)
- `q01`: EdU-only (`B=0`, `E=1`)
- `q00`: Double-negative (`B=0`, `E=0`)

We estimate four matched conditional ORs (phase/type matched; see `methods.md` for the exact strata and gate definition):

### E1: within BrdU+ (`q11` vs `q10`)

`OR_E1 = OR_{GE | B=1}`

Interpretation:

- compares gate+ vs gate− among cells BrdU+ at `t=0`
- sensitive to persistence/progression of the BrdU+ cohort (and BrdU-specific calling)

### E3: within BrdU− (`q01` vs `q00`)

`OR_E3 = OR_{GE | B=0}`

Interpretation:

- “general EdU propensity / S-entry / EdU calling” axis
- if E3 is large, the gate is strongly associated with EdU+ even when BrdU=0, so BrdU+ effects are likely mixed with entry/propensity

### E2 and E4 (mirror conditionals; diagnostics)

- `OR_E2 = OR_{GB | E=1}` (within EdU+)
- `OR_E4 = OR_{GB | E=0}` (within EdU−)

Mirror conditionals can disagree with E1/E3; we treat them as diagnostics for label asymmetry and support issues, not as hard constraints.

## 2.1 Continuity vs recruitment: what the dual pulse directly gives

The assay is best understood as a pairing of:

1. a transcriptional state (`G`), and
2. a two-timepoint indicator of being in S-phase over a fixed lag `Δt` (`B` at `t=0`, `E` at `t=Δt`),

within matched cell state (Leiden) and matched cell-cycle position (θ bin).

Within a matched stratum, two conditional probabilities are directly interpretable as short-horizon flux:

- **Continuity / retention over Δt (elongation-relevant):** `ρ(G) = P(E=1 | B=1, G, matched)`
- **Recruitment / entry over the window:** `η(G) = P(E=1 | B=0, G, matched)`

Operationally, these correspond to the within-stratum contrasts:

- continuity effect: `Δ1 = P(E=1 | G=1, B=1) − P(E=1 | G=0, B=1)` (odds-scale companion: E1 / `OR_E1`)
- recruitment effect: `Δ0 = P(E=1 | G=1, B=0) − P(E=1 | G=0, B=0)` (odds-scale companion: E3 / `OR_E3`)

If the biological goal is “S-phase elongation markers,” `Δ1`/E1 is the clean headline readout; interaction terms are then used to classify whether that continuity association is continuity-specific versus part of a broader recruitment/occupancy program.

### 2.1.1 Relation to the standard `T_S` convention (`BrdU+EdU+ / BrdU+`)

The standard double-pulse convention treats:

- `BrdU+EdU+ / BrdU+` as the **retention fraction** over the pulse interval `Δt`:
  - `f = P(E=1 | B=1, matched)` (within the analysis population / stratum definition).

Under the usual assumption that S-phase positions are approximately uniform at `t=0` within the compared group, this yields:

- `T_S/Δt ≈ 1/(1−f)`, i.e. `T_S ≈ Δt/(1−f)` (for `f<1`).

In the GLM reporting layer, we explicitly output these quantities on the probability scale for interpretability:

- `p_b1_g0` = model-implied `P(E=1 | B=1, G=0, matched)` (gate− retention fraction).
- `p_b1_g1` = model-implied `P(E=1 | B=1, G=1, matched)` (gate+ retention fraction).
- `Δ1 = p_b1_g1 − p_b1_g0` (continuity / retention effect).

We also report the corresponding **dimensionless** S-phase time contrasts (in units of `Δt`):

- `T_S/Δt (G=0) = 1/(1−p_b1_g0)`
- `T_S/Δt (G=1) = 1/(1−p_b1_g1)`
- `Δ(T_S/Δt) = (T_S/Δt)(G=1) − (T_S/Δt)(G=0)`

Practical note: this `T_S` transform is nonlinear and can become unstable when `f→1`, so we treat `Δ1` as the primary retention readout and use `T_S/Δt` as a standard-convention companion.

## 3) The mechanism-aware scalar: IntE (BrdU-conditioned beyond EdU propensity)

One useful scalar diagnostic is the interaction:

- `IntE = log(OR_E1) − log(OR_E3)` (aka `logIOR`)

Interpretation:

- subtracts the BrdU− EdU-propensity axis (E3) from the BrdU+ conditional (E1)
- it is the tightest single-number summary of “BrdU-conditioned persistence/progression” available from the four quadrants

Sign conventions:

- `IntE < 0`: gene+ has lower BrdU-conditioned persistence/progression than expected from its general EdU propensity.
- `IntE > 0`: gene+ has higher BrdU-conditioned persistence/progression than expected from its general EdU propensity.

Because ORs can look extreme when `P(E=1 | B=1, …)` is near saturation, we also report a probability-scale interaction:

- `ΔINT = (P(E=1|G=1,B=1)−P(E=1|G=0,B=1)) − (P(E=1|G=1,B=0)−P(E=1|G=0,B=0))`

This makes “ceiling effects” explicit and is used as a companion interpretability metric (not a replacement for IntE/logIOR).

Biological reading (important):

- Throughout, when we say a gene “changes EdU positivity,” we mean a **within-stratum association contrast** (not a causal effect): compare two otherwise matched cells (same Leiden and same θ bin), one **gene-high** (`G=1`, in the gate) and one **gene-low** (`G=0`, out of gate), and ask how their probability of being EdU+ differs at the EdU pulse.
  - `Δ0` answers this contrast **within BrdU− cells**: does the gene-high state mark cells that are more/less likely to become EdU+ over the interval (entry/propensity axis)?
  - `Δ1` answers the same contrast **within BrdU+ cells**: among cells already in S at BrdU, does the gene-high state mark cells that are more/less likely to still be EdU+ at `t=Δt` (continuity axis)?
- `ΔINT ≈ 0` does **not** mean “no effect” — it means the gate’s EdU association is **not BrdU-history dependent** (`Δ1 ≈ Δ0`).
  - This is consistent with a **cell-state** program whose association with EdU positivity is similar in BrdU− and BrdU+ cohorts (i.e. it does not specifically depend on being BrdU+ at `t=0`).
  - It is **not** a diagnostic for “general S-phase marker”: a cell-cycle–linked gene can yield either `ΔINT≈0` or `ΔINT≠0` depending on how BrdU partitions time-since-S (stage distribution) and on ceiling/floor effects.
- `ΔINT > 0` indicates the gene’s EdU association is **more BrdU+-specific** (`Δ1 > Δ0`), consistent with a continuity/retention-skewed signature.
- `ΔINT < 0` indicates the gene’s EdU association is **more BrdU−-specific** (`Δ0 > Δ1`), consistent with an entry/propensity-skewed signature.

## 4) Mechanism signatures (how we classify genes)

These patterns are used for interpretation and prioritization:

### 4.1 “Entry/propensity-dominated” (not persistence-specific)

Typical pattern:

- E3 is large (often significant), sometimes comparable to E1.

Interpretation:

- gene gate is a marker of cycling propensity, S-entry timing, and/or EdU calling.

### 4.2 “BrdU-conditioned persistence/progression-like” (more kinetics-like)

Typical pattern:

- IntE is significant,
- E3 is near-null (small and/or not significant),
- ideally IntB matches IntE sign.

Interpretation:

- harder to explain as general EdU propensity; more consistent with a BrdU-conditioned structural effect in the dual-pulse table.

Note: this “E3 near-null” signature is useful when the goal is to isolate a **pulse-history–specific** phenotype beyond general EdU propensity. It is *not* required for “slow/extended S-like” interpretations, where the state can plausibly affect EdU positivity even among BrdU− cells.

### 4.3 “Mixed / asymmetric”

Typical pattern:

- E3 is large and IntE is also nonzero, and/or mirror metrics disagree strongly.

Interpretation:

- multiple pathways affect the joint table (entry, persistence/progression, calling asymmetry).
  These are still valid phenotypes, but should not be described as “pure persistence”.

## 5) What we mean by “real kinetics-like gene” (operational criteria)

In this dataset, we prioritize “real kinetics-like” candidates by requiring:

1. **Interaction exists (primary estimand):** the stratum-FE GLM interaction `β_G×B` excludes 0 and is stable across animals (LOAO / per-animal fits).
2. **E3 is characterized (not excluded):** report E3 (BrdU−/EdU axis) alongside the interaction (`β_G×B` / `IntE`). Large E3 implies a **mixed** phenotype (entry/propensity and/or extended S occupancy/calling), which can still be compatible with “slow/extended S-like” interpretations.
3. **Not just cell type:** the gate is not essentially a Leiden classifier.
4. **Spatial robustness (if claiming within-location effect):** space-adjusted IntE retains direction/magnitude.

Passing these criteria supports: “gene-defined subpopulation has a BrdU-conditioned interaction phenotype consistent with altered persistence/progression,” without implying causality.

### 5.1 Explicit filtering / prioritization steps (what we actually do)

We separate *panel-wide screening* from *shortlist characterization* to keep the narrative coherent and to avoid overclaiming with `n_animals=5`.

#### Stage A: Panel-wide screen (ranking + stability; animal-aware; Leiden-stratified)

0. **Gene universe (pre-specified):** restrict to **non-neuroRef genes** (exclude the θ reference set) to avoid θ leakage in confirmatory screening.
1. **Leiden-specific LOAO cross-fit screens:** run panel screens **separately within Leiden** (core narrative scope: Leiden 14 and 15; Leiden 6 is QC/reference).
2. **Screen outputs (per gene):** rank by replicate-aware summaries (mean/SD and sign-consistency across held-out animals) for the chosen primary screen metric.

We keep overlap/effective-support metrics as optional QC/sensitivity, but the panel-wide sign distribution of odds-scale interactions is not treated as a biological summary when EdU is rare in `B=0` (see `methods.md`).

This stage is used for **prioritization** and “reproducible directionality” checks, not as the final publishable p-value layer.

#### Stage B: Shortlist characterization (publishable estimand table)

3. **Select a manageable shortlist** from Stage A (e.g., top K by stability + magnitude).
4. **Refit the primary estimand (GLM):** for each shortlist gene at the chosen `q`:
   - fit the stratum-FE grouped-binomial GLM *within each animal* and extract `β_G×B` (primary) and `β_G` (E3 axis),
   - meta-analyze across animals (fixed-effect as default; random-effects τ² reported as a heterogeneity diagnostic).
   - report the retention/elongation readout explicitly on the probability scale (`p_b1_g0`, `p_b1_g1`, `Δ1`) and, where helpful for field readers, the derived `T_S/Δt` contrasts.
   - run the required inference sensitivity suite for narrative genes (θ ablations, q-curve, gate-direction check, and estimator-sensitivity anchors) via `scripts/brdu_regression/run_shortlist_inference_suite.py`.
5. **Narrative filtering (example used in current draft table):**
   - **sign agreement:** `sign(β_G×B)` is consistent between the panel screen and the per-animal meta refit,
   - **E3 (entry/propensity axis):** report `β_G` (≈ `log(OR_E3)`) and use it to *label* phenotypes (retention-specific vs mixed), not to exclude genes. A state that “slows/extends S” can plausibly affect both BrdU+ retention (E1) and BrdU−→EdU+ propensity (E3), so non-null E3 is not a disqualifier.
   - **heterogeneity check:** τ² for `β_G×B` small (used as a QC flag; not a hard rule yet).

Genes with large `|β_G|` are categorized as **mixed / propensity-involved** rather than “pure persistence/progression-like”, but are not excluded from the narrative shortlist by default.

## 6) Spatial interpretation: composition vs within-location effect

Spatial structure can affect interaction phenotypes in two ways:

1. **Composition / spatial confounding:** gate prevalence differs across regions with different baseline BrdU→EdU regimes.
2. **Within-location effect:** interaction persists after coarse spatial conditioning.

We therefore report both:

- `δ_marg` (no explicit spatial conditioning)
- `δ_space` (space-conditioned)
- `δ_comp` (composition component): conceptually `δ_marg − δ_space`, but note that changing strata changes support/weights; for shortlist follow-up we prefer the support-aligned decomposition reported by `scripts/brdu_regression/spatial_decomposition_standardized.py` (see `delta_comp_raw` vs `delta_comp_std_support`).

## 7) Where the mechanism tables live (current runs)

Canonical mechanism-aware summaries:

- Legacy hard-call MH quadrant table (for comparison only): `scripts/_out/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2/quadrant_validation.csv`
- Executive summary table (endpoints, interactions, sparsity diagnostics, confound flags): `scripts/_out/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50/consultant_exec_summary.csv`
- Gate definition parameters (`q` per gene; Pearson residual settings are in scripts): `scripts/_out/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv`
- Legacy hard-call MH panel screen (for comparison only): `scripts/_out/ts_scan_gate_mode_a_phase_matched/panel_crossfit_bestq_mh_non_neuroref_20260221_v2_stability/panel_crossfit_by_gene.csv`
- Panel-wide GLM best-q screens (Leiden-stratified; 2026-02-22): `scripts/_out/ts_scan_gate_mode_a_phase_matched/panel_crossfit_glm_bestq_leiden14_20260222_support/panel_crossfit_by_gene.csv` and `scripts/_out/ts_scan_gate_mode_a_phase_matched/panel_crossfit_glm_bestq_leiden15_20260222_support/panel_crossfit_by_gene.csv`
- Shortlist GLM-by-animal meta (Leiden-stratified; includes `Δ1/Δ0/ΔINT`): `scripts/_out/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_leiden14_20260222/shortlist_glm_meta.csv` and `scripts/_out/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_leiden15_20260222/shortlist_glm_meta.csv`
- Joint Leiden 14+15 shortlist refit (paired within-animal heterogeneity; includes `p_b1_g0/p_b1_g1` and `T_S/Δt` contrasts): `scripts/_out/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_joint1415_from_leiden14_20260222/shortlist_glm_meta.csv` and `scripts/_out/ts_scan_gate_mode_a_phase_matched/shortlist_glm_meta_joint1415_from_leiden15_20260222/shortlist_glm_meta.csv`

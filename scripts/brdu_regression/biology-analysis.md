# BrdU -> EdU (BrdU-first) biological interpretation: mechanism-aware endpoints

This document is a companion to `scripts/brdu_regression/methods.md`. It explains the biological meaning of the **four BrdU/EdU quadrants** and the derived interaction phenotype, and it defines what we mean by a “real kinetics-like gene” in a way that is testable with this dataset.

## 1) What can we claim from this assay?

BrdU is administered first (`t=0`) and EdU second (`t=Δt`). In the idealized model:

- BrdU+ marks cells in S at `t=0`.
- EdU+ marks cells in S at `t=Δt`.

In real targeted-panel data, the BrdU/EdU table also reflects:

- entry into S between pulses (and exit from S),
- analogue availability windows,
- and label calling / incorporation artifacts.

Therefore, when we use the word “kinetics” here, we mean:

> a reproducible **BrdU-conditioned persistence/progression phenotype** in the dual-pulse table, not a guaranteed literal S-phase duration estimate.

## 2) The four quadrants and what each conditional endpoint diagnoses

Let `B=brdu_pos`, `E=edu_pos`, and `G` be a gene-high gate.

Each cell is in one quadrant:

- A: Dual (`B=1`, `E=1`)
- B: BrdU-only (`B=1`, `E=0`)
- C: EdU-only (`B=0`, `E=1`)
- D: Double-negative (`B=0`, `E=0`)

We estimate four matched conditional ORs (phase/type matched; see `methods.md` for the exact strata and gate definition):

### E1: within BrdU+ (Dual vs BrdU-only)

`OR_E1 = OR_{GE | B=1}`

Interpretation:

- compares gate+ vs gate− among cells BrdU+ at `t=0`
- sensitive to persistence/progression of the BrdU+ cohort (and BrdU-specific calling)

### E3: within BrdU− (EdU-only vs Double-negative)

`OR_E3 = OR_{GE | B=0}`

Interpretation:

- “general EdU propensity / S-entry / EdU calling” axis
- if E3 is large, the gate is strongly associated with EdU+ even when BrdU=0, so BrdU+ effects are likely mixed with entry/propensity

### E2 and E4 (mirror conditionals; diagnostics)

- `OR_E2 = OR_{GB | E=1}` (within EdU+)
- `OR_E4 = OR_{GB | E=0}` (within EdU−)

Mirror conditionals can disagree with E1/E3; we treat them as diagnostics for label asymmetry and support issues, not as hard constraints.

## 3) The mechanism-aware scalar: IntE (BrdU-conditioned beyond EdU propensity)

The key phenotype is the interaction:

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

### 4.3 “Mixed / asymmetric”

Typical pattern:

- E3 is large and IntE is also nonzero, and/or mirror metrics disagree strongly.

Interpretation:

- multiple pathways affect the joint table (entry, persistence/progression, calling asymmetry).
  These are still valid phenotypes, but should not be described as “pure persistence”.

## 5) What we mean by “real kinetics-like gene” (operational criteria)

In this dataset, we prioritize “real kinetics-like” candidates by requiring:

1. **Interaction exists:** IntE CI excludes 0 and is stable across datasets/animals.
2. **Not just entry/calling:** E3 is small and/or not significant.
3. **Not just cell type:** the gate is not essentially a Leiden classifier.
4. **Spatial robustness (if claiming within-location effect):** space-adjusted IntE retains direction/magnitude.

Passing these criteria supports: “gene-defined subpopulation has a BrdU-conditioned interaction phenotype consistent with altered persistence/progression,” without implying causality.

## 6) Spatial interpretation: composition vs within-location effect

Spatial structure can affect interaction phenotypes in two ways:

1. **Composition / spatial confounding:** gate prevalence differs across regions with different baseline BrdU→EdU regimes.
2. **Within-location effect:** interaction persists after coarse spatial conditioning.

We therefore report both:

- `δ_marg` (no explicit spatial conditioning)
- `δ_space` (space-conditioned)
- `δ_comp = δ_marg − δ_space` (composition component)

## 7) Where the mechanism tables live (current runs)

Canonical mechanism-aware summaries:

- Quadrant endpoints + IntE/IntB (MH): `scripts/_out/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2/quadrant_validation.csv`
- Executive summary table (endpoints, interactions, sparsity diagnostics, confound flags): `scripts/_out/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2_top50/consultant_exec_summary.csv`
- Gate definition parameters (`q` per gene; Pearson residual settings are in scripts): `scripts/_out/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv`

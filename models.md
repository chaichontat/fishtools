# Models

## Hierarchical GLMM for `Ts` in top-20% `pyUCell` cells

This model is used for the `vzsvz.h5ad` analysis with:

- `manual_layer == '1'`
- Leiden `0/1/3`
- `JaxA2` excluded
- `top20` defined as the top 20% of the `pyUCell` score for `Tnc`, `Ptprz1`, `Hes5`, `Vim-`, `Pax6-`, `Eomes-`, `Nsg2-`, `Dcx-`

The fit is performed only on `BrdU+` cells.

### Response

For each `BrdU+` cell `i`:

- `dual_i = 1` if the cell is `BrdU+ EdU+`
- `dual_i = 0` if the cell is `BrdU+ EdU-`

So the modeled probability is:

- `P(dual_i = 1 | BrdU_i = 1) = P(EdU_i = 1 | BrdU_i = 1)`

### Model

The hierarchical binomial logistic mixed model is:

`logit P(dual_i = 1) = beta0 + beta1 * top20_i + b_animal[i]`

with:

- `top20_i = 1` for cells in the top-20% signature subset
- `top20_i = 0` otherwise
- `b_animal ~ Normal(0, sigma_animal^2)`

Interpretation:

- `beta1` is the effect of belonging to the top-20% signature subset
- `b_animal` is an animal-specific random intercept
- `sigma_animal` captures between-animal heterogeneity in baseline `BrdU+ -> EdU+` probability
- the current fit allows animal-specific baseline shifts only; it does not include an animal-specific random slope for `top20`

### What is reported

The current output table reports:

- an approximate Wald test for `beta1`
- the corresponding odds ratio `exp(beta1)`
- conditional `Ts` predictions for `top20 = 0` and `top20 = 1`, evaluated at random intercept `b = 0`
- marginalized `Ts` predictions for `top20 = 0` and `top20 = 1`, integrated over the animal random-intercept distribution

The conditional `Ts` values are obtained by transforming:

- `f_rest = logistic(beta0)`
- `f_top20 = logistic(beta0 + beta1)`

and then applying:

- `Ts = Δt * f / (1 - f)`

The marginalized `Ts` values instead use:

- `f_x = E_b[ logistic(beta0 + beta1 * x + b) ]`
- `b ~ Normal(0, sigma_animal^2)`

before applying:

- `Ts = Δt * f_x / (1 - f_x)`

### Mapping to `Ts`

Let:

- `f = P(EdU+ | BrdU+) = P(dual = 1 | BrdU+)`

Then:

- `Ts = Δt * f / (1 - f)`

For the current analyses:

- `Δt = 1.5` hours

So the mixed model tests whether the top-20% signature subset has a higher `BrdU+` dual-label fraction after accounting for animal-to-animal variation, and the fitted fractions are then converted to `Ts`.

### Caveats

- After excluding `JaxA2`, there are only 4 animals in this analysis.
- With so few animals, a cell-level random-intercept logistic model can produce extremely small p-values because the fit is driven by many cells plus parametric assumptions.
- The hierarchical GLMM should therefore be treated as model-based partial-pooling evidence, not as definitive replicate-level confirmation.
- The paired animal sign-flip analysis remains the stricter animal-level check when the goal is replicate-based inference.
- Because the model assumes a common `top20` effect across animals, inference can be sensitive if the effect is heterogeneous across animals.

### Current output

The current fitted output table is:

- `scripts/_out/pyucell_top20_ts_vzsvz_manual_layer1_leiden013_paired/hierarchical_glmm_test.csv`

## Pooled AP/ML Multinomial Model For `Ts` / `Tc`

This is the model used for the pooled AP/ML effect-size figures and sagittal-line renders in:

- [scripts/fit_apml_multinomial_animal_meta.py](/home/chaichontat/fishtools2/scripts/fit_apml_multinomial_animal_meta.py)
- [scripts/plot_ts_tc_vs_sagittal_t_pooled.py](/home/chaichontat/fishtools2/scripts/plot_ts_tc_vs_sagittal_t_pooled.py)
- [scripts/plot_ts_tc_native_with_sagittal_line.py](/home/chaichontat/fishtools2/scripts/plot_ts_tc_native_with_sagittal_line.py)

### Data used

The pooled fit is built from `all_excit.h5ad`, with `obsm` overwritten from `~/nvme/obsm.h5ad`.

The current plotted subsets are:

- `manual_layer = 1`
- `manual_layer = 5, Eomes > 1`

Common filters:

- Leiden `0..8`
- `JaxA2` excluded
- AP/ML coordinates taken from `obsm["AP_ML_um"]`

### Response

Each cell is assigned to one of four mutually exclusive states:

- `unlabeled`
- `brdu_only`
- `edu_only`
- `dual`

where:

- `brdu_only`: `BrdU+ EdU-`
- `edu_only`: `BrdU- EdU+`
- `dual`: `BrdU+ EdU+`

### Model

The pooled AP/ML surface is fit with a baseline-category multinomial logit model, using `unlabeled` as the reference state:

`log P(y_i = c) / P(y_i = unlabeled) = beta_c0 + beta_c_AP * AP_i + beta_c_ML * ML_i`

for `c in {brdu_only, edu_only, dual}`.

Notes:

- `AP_i` and `ML_i` are in millimeters
- this pooled effect-size fit does **not** include `animal` terms
- the per-animal multinomial fits in the same script are used separately as a robustness check, not as the main pooled effect-size estimate

### Mapping to `Ts` and `Tc`

Let the fitted multinomial probabilities at an AP/ML location be:

- `p_BO = P(brdu_only)`
- `p_D = P(dual)`

Then the derived quantities are:

- `Ts = Δt * p_D / p_BO`
- `Tc = Ts / p_D = Δt / p_BO`

For the current analyses:

- `Δt = 1.5` hours

So the pooled `Ts` and `Tc` plots are not fit directly as separate responses. They are deterministic transforms of the fitted multinomial state probabilities.

### What is plotted

The same pooled AP/ML multinomial fit is evaluated at arbitrary query points:

- on the native AP/ML surface
- along sagittal slice lines after those lines are mapped into AP/ML space

Support clipping is a display restriction only. The model is fit once on the filtered cells, then evaluated on the requested AP/ML query coordinates and masked to the chosen support region for plotting.

### Uncertainty

Pointwise uncertainty bands are obtained by:

- drawing multinomial coefficients from the fitted `MNLogit` covariance
- recomputing the multinomial probabilities at each query point
- transforming those draws into `Ts` and `Tc`
- reporting the `2.5%` and `97.5%` quantiles

So the `Ts` / `Tc` intervals are propagated from the joint multinomial fit rather than from separate independent GLMs.

## Animal-Level Multinomial Robustness Check

This is the replicate-level robustness analysis used to check whether the pooled AP/ML effect has consistent direction across animals.

It uses the same multinomial state definition and the same AP/ML coordinates as the pooled model, but it changes the estimand:

- the pooled model asks for one common effect-size surface across all cells
- the robustness path asks whether animal-specific effects tend to agree

### Per-animal fit

For each animal separately, a 2D baseline-category multinomial logit model is fit over:

- `unlabeled`
- `brdu_only`
- `edu_only`
- `dual`

with `unlabeled` as the reference state and AP/ML as predictors:

`log P(y_i = c) / P(y_i = unlabeled) = beta_c0^(animal) + beta_c_AP^(animal) * AP_i + beta_c_ML^(animal) * ML_i`

for `c in {brdu_only, edu_only, dual}`.

This is implemented in [scripts/fit_apml_multinomial_animal_meta.py](/home/chaichontat/fishtools2/scripts/fit_apml_multinomial_animal_meta.py).

### Derived quantities

Within each animal, the fitted multinomial probabilities are transformed the same way as in the pooled model:

- `Ts = Δt * p_D / p_BO`
- `Tc = Δt / p_BO`

### Animal-level summaries

The current robustness figures do not report the raw multinomial coefficients directly. Instead, each animal fit is summarized into AP and ML contrasts or slopes for:

- `brdu_only_frac`
- `edu_only_frac`
- `dual_frac`
- `Ts_hours`
- `Tc_hours`

In the current summary plots, the displayed quantity is an on-support per-mm slope:

- evaluate the fitted metric between that animal’s `q10` and `q90` support along one axis
- hold the other axis at that animal’s median
- divide the fitted change by the support span

So these animal points are support-aware derived slopes, not direct multinomial coefficient estimates.

### Across-animal combination

After fitting each animal separately, the animal-level summaries are combined across animals with random-effects meta-analysis.

This means:

- each animal contributes one effect estimate and one within-animal uncertainty estimate
- the black summary error bar reflects between-animal heterogeneity as well as within-animal uncertainty
- animals are not pooled simply by cell count in this step

This robustness path is therefore closer to equal-animal inference than the pooled cell-level effect-size model.

### Interpretation

Use this section as a robustness check, not as the main effect-size estimate:

- pooled multinomial model: best for the main AP/ML effect-size surface
- animal-level multinomial meta-analysis: best for checking sign consistency and heterogeneity across animals

If the two disagree, that usually indicates animal heterogeneity or orientation/support differences rather than a contradiction in the fitting code.

# Model Notes

## Usage_7 Attribution Model

The `Usage_7` attribution analysis for `manual_layer=1` uses a pooled cell-level multinomial model over four label states:

- `unlabeled`
- `BrdU-only`
- `EdU-only`
- `dual`

The fitted model is:

```text
state ~ AP + ML + Usage_7
```

where:

- `AP` and `ML` are the flattened spatial coordinates from `obsm['AP_ML_um']`
- `Usage_7` is a cell-level continuous covariate
- `JaxA2` is excluded from the fitted population

The model is fit at the cell level, not on bins.

## Derived Quantities

After fitting the multinomial, the state probabilities are converted to:

```text
Ts = Δt * p_dual / p_brdu_only
Tc = Ts / p_dual
```

with `Δt = 1.5 h`.

So `Ts` and `Tc` are not fit directly. They are deterministic transforms of the fitted multinomial probabilities.

## Spatial Attribution

The attribution analysis is counterfactual.

First, a smooth spatial `Usage_7(AP, ML)` field is estimated from the existing simplex GAM output in:

[`scripts/_out/gam/vzsvz_manual_layer1_simplex_20260322_142901`](/home/chaichontat/fishtools2/scripts/_out/gam/vzsvz_manual_layer1_simplex_20260322_142901)

That GAM field is marginalized over:

- `r_um`
- `theta`
- animal

and cached before reuse in the attribution renderers.

At each spatial location, the multinomial is evaluated twice:

1. Factual:

```text
T_factual(AP, ML) = T(AP, ML, Usage_7(AP, ML))
```

2. Counterfactual baseline:

```text
T_base(AP, ML) = T(AP, ML, u_ref)
```

where `u_ref` is a low in-support baseline, currently taken from a low quantile of the populated-bin GAM `Usage_7` values (for example q05 or q10).

The plotted attribution is:

```text
ΔT(AP, ML) = T_factual(AP, ML) - T_base(AP, ML)
```

for both `Ts` and `Tc`.

## Interpretation

This is not an exclusion analysis and not a causal estimate. It is a model-based decomposition:

- hold `AP` and `ML` fixed
- replace the spatial `Usage_7` field with a low baseline
- measure how much model-implied `Ts` or `Tc` changes

The current multinomial is additive only:

```text
state ~ AP + ML + Usage_7
```

There are no `AP:Usage_7` or `ML:Usage_7` interaction terms, so the model assumes the effect of `Usage_7` is spatially constant.

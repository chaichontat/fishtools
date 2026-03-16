#!/usr/bin/env Rscript

# Canonical simplex fitter entrypoint.
# We now default to the ILR implementation so callers do not need to pick
# between parallel ALR/ILR scripts.

source("scripts/gam/fit_inm_simplex_panel_ilr.R")

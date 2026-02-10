#!/usr/bin/env Rscript

source("scripts/gam/inm_gam.R")
source("scripts/gam/simulate_inm_panel.R")

set.seed(1)
sim <- simulate_inm_panel(
  n_cells = 900,
  n_null = 3,
  n_cycle = 3,
  n_spatial = 3,
  n_interaction = 3,
  seed = 1,
  inm_sigma_r = 0.06,
  nb_size = 10
)

coupling <- fit_inm_coupling_gam(sim$cells$x, sim$cells$theta, k_theta = 8)
if (coupling$r2 < 0.7) stop(sprintf("Expected strong INM coupling, got R2=%.3f", coupling$r2))

results <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  theta = sim$cells$theta,
  sf = sim$cells$s
)

merged <- merge(results, sim$truth, by = "gene", all.x = TRUE, sort = FALSE)

ok <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg)
}

min_p <- function(p) suppressWarnings(min(p, na.rm = TRUE))
max_p <- function(p) suppressWarnings(max(p, na.rm = TRUE))

ok(min_p(merged$p_cycle[merged$kind == "cycle"]) < 1e-6, "Cycle genes should have strong cycle signal.")
ok(min_p(merged$p_spatial[merged$kind == "spatial"]) < 1e-6, "Spatial genes should have strong spatial signal.")
ok(min_p(merged$p_interaction[merged$kind == "interaction"]) < 1e-6, "Interaction genes should have strong interaction signal.")

# Null genes should not look significant across any component in this synthetic setting.
ok(max_p(merged$p_cycle[merged$kind == "null"]) > 1e-3, "Null genes cycle p-values unexpectedly small.")
ok(max_p(merged$p_spatial[merged$kind == "null"]) > 1e-3, "Null genes spatial p-values unexpectedly small.")
ok(max_p(merged$p_interaction[merged$kind == "null"]) > 1e-3, "Null genes interaction p-values unexpectedly small.")

theta_shift <- sim$cells$theta + 4 * pi
results_shift <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  theta = theta_shift,
  sf = sim$cells$s
)
merged_shift <- merge(results_shift, sim$truth, by = "gene", all.x = TRUE, sort = FALSE)
ok(min_p(merged_shift$p_cycle[merged_shift$kind == "cycle"]) < 1e-6, "Wrapped-theta cycle genes should have strong cycle signal.")
ok(min_p(merged_shift$p_spatial[merged_shift$kind == "spatial"]) < 1e-6, "Wrapped-theta spatial genes should have strong spatial signal.")
ok(min_p(merged_shift$p_interaction[merged_shift$kind == "interaction"]) < 1e-6, "Wrapped-theta interaction genes should have strong interaction signal.")

ok(all(c("cycle_amp_link", "spatial_grad_link", "gating_index_link") %in% names(results)),
   "Expected effect size columns in fit_panel output.")

res_list <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  theta = sim$cells$theta,
  sf = sim$cells$s,
  return = "list"
)
ok(is.list(res_list) && all(c("table", "coupling", "r", "theta") %in% names(res_list)),
   "Expected fit_panel(return='list') to return table+coup+r+theta.")
ok(is.data.frame(res_list$table), "Expected res_list$table to be a data.frame.")

cat("OK: synthetic smoke test passed.\n")

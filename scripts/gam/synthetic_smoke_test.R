#!/usr/bin/env Rscript

source("scripts/gam/inm_gam.R")
source("scripts/gam/simulate_inm_panel.R")

set.seed(1)
sim <- simulate_inm_panel(
  n_cells = 250,
  n_null = 1,
  n_cycle = 1,
  n_spatial = 1,
  n_interaction = 1,
  seed = 1,
  inm_sigma_r = 0.06,
  nb_size = 10
)

coupling <- fit_inm_coupling_gam(sim$cells$x, sim$cells$theta, k_theta = 8)
if (coupling$r2 < 0.7) stop(sprintf("Expected strong INM coupling, got R2=%.3f", coupling$r2))

results <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  r_um = sim$cells$r_um,
  AP_um = sim$cells$AP_um,
  ML_um = sim$cells$ML_um,
  theta = sim$cells$theta,
  sf = sim$cells$s,
  k_uv = 8,
  k_uvr = 8,
  k_r_uvr = 3,
  k_rtheta = c(4, 6)
)

merged <- merge(results, sim$truth, by = "gene", all.x = TRUE, sort = FALSE)

ok <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg)
}

min_p <- function(p) suppressWarnings(min(p, na.rm = TRUE))
max_p <- function(p) suppressWarnings(max(p, na.rm = TRUE))

ok(min_p(merged$p_cycle[merged$kind == "cycle"]) < 1e-4, "Cycle genes should have strong cycle signal.")
ok(min_p(merged$p_spatial[merged$kind == "spatial"]) < 1e-4, "Spatial genes should have strong spatial signal.")
ok(min_p(merged$p_interaction[merged$kind == "interaction"]) < 1e-4, "Interaction genes should have strong interaction signal.")

# Null genes should not look significant across any component in this synthetic setting.
ok(max_p(merged$p_cycle[merged$kind == "null"]) > 1e-3, "Null genes cycle p-values unexpectedly small.")
ok(max_p(merged$p_spatial[merged$kind == "null"]) > 1e-3, "Null genes spatial p-values unexpectedly small.")
ok(max_p(merged$p_interaction[merged$kind == "null"]) > 1e-3, "Null genes interaction p-values unexpectedly small.")

theta_shift <- sim$cells$theta + 4 * pi
results_shift <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  r_um = sim$cells$r_um,
  AP_um = sim$cells$AP_um,
  ML_um = sim$cells$ML_um,
  theta = theta_shift,
  sf = sim$cells$s,
  k_uv = 8,
  k_uvr = 8,
  k_r_uvr = 3,
  k_rtheta = c(4, 6)
)
merged_shift <- merge(results_shift, sim$truth, by = "gene", all.x = TRUE, sort = FALSE)
ok(min_p(merged_shift$p_cycle[merged_shift$kind == "cycle"]) < 1e-4, "Wrapped-theta cycle genes should have strong cycle signal.")
ok(min_p(merged_shift$p_spatial[merged_shift$kind == "spatial"]) < 1e-4, "Wrapped-theta spatial genes should have strong spatial signal.")
ok(min_p(merged_shift$p_interaction[merged_shift$kind == "interaction"]) < 1e-4, "Wrapped-theta interaction genes should have strong interaction signal.")

ok(all(c("cycle_amp_link", "spatial_grad_link", "gating_index_link") %in% names(results)),
   "Expected effect size columns in fit_panel output.")

res_list <- fit_panel(
  counts = sim$counts,
  x = sim$cells$x,
  r_um = sim$cells$r_um,
  AP_um = sim$cells$AP_um,
  ML_um = sim$cells$ML_um,
  theta = sim$cells$theta,
  sf = sim$cells$s,
  k_uv = 8,
  k_uvr = 8,
  k_r_uvr = 3,
  k_rtheta = c(4, 6),
  return = "list"
)
ok(is.list(res_list) && all(c("table", "coupling", "r_um", "theta") %in% names(res_list)),
   "Expected fit_panel(return='list') to return table+coup+r_um+theta.")
ok(is.data.frame(res_list$table), "Expected res_list$table to be a data.frame.")

# Regression: diagnostics script should run and emit concurvity/EDF tables.
tmp_panel <- file.path(tempdir(), sprintf("inm_synth_panel_%d", as.integer(Sys.time())))
dir.create(tmp_panel, showWarnings = FALSE, recursive = TRUE)
cells_path <- file.path(tmp_panel, "cells.tsv")
counts_path <- file.path(tmp_panel, "counts.tsv")

cells_out <- data.frame(
  cell_id = seq_len(nrow(sim$cells)),
  x = sim$cells$x,
  r_um = sim$cells$r_um,
  AP_um = sim$cells$AP_um,
  ML_um = sim$cells$ML_um,
  theta = sim$cells$theta,
  s = sim$cells$s,
  stringsAsFactors = FALSE
)
write.table(cells_out, file = cells_path, sep = "\t", row.names = FALSE, quote = FALSE)

g1 <- colnames(sim$counts)[[1]]
counts_out <- data.frame(cell_id = seq_len(nrow(sim$cells)), stringsAsFactors = FALSE)
counts_out[[g1]] <- sim$counts[, 1]
write.table(counts_out, file = counts_path, sep = "\t", row.names = FALSE, quote = FALSE)

diag_out <- file.path(tmp_panel, "diagnostics_out")
dir.create(diag_out, showWarnings = FALSE, recursive = TRUE)
cmd <- c("scripts/gam/check_gam_diagnostics.R", tmp_panel, g1, diag_out)
status <- suppressWarnings(system2("Rscript", cmd, stdout = TRUE, stderr = TRUE))
if (!file.exists(file.path(diag_out, "coupling_diagnostics.png"))) stop("Expected coupling_diagnostics.png")
if (!file.exists(file.path(diag_out, "edf.tsv"))) stop("Expected edf.tsv")
cv_path <- file.path(diag_out, "concurvity.tsv")
if (!file.exists(cv_path)) stop("Expected concurvity.tsv")
cv_head <- read.delim(cv_path, nrows = 1, stringsAsFactors = FALSE, check.names = FALSE)
ok(all(c("kind", "row", "col", "value") %in% names(cv_head)),
   "Expected concurvity.tsv to contain kind/row/col/value columns.")

cat("OK: synthetic smoke test passed.\n")

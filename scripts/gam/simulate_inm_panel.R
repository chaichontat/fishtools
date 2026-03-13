#!/usr/bin/env Rscript

simulate_inm_panel <- function(
  n_cells = 800,
  n_null = 4,
  n_cycle = 4,
  n_spatial = 4,
  n_interaction = 4,
  seed = 1,
  inm_sigma_r = 0.05,
  nb_size = 8,
  base_log_mu = -7.5
) {
  set.seed(seed)

  theta <- runif(n_cells, 0, 2 * pi)

  # Periodic INM coupling curve m(theta); apical is near theta ~ 0.
  m_theta <- 0.5 +
    0.25 * cos(theta) +
    0.08 * sin(theta) +
    0.06 * cos(2 * theta) -
    0.03 * sin(2 * theta)

  r_true <- rnorm(n_cells, mean = 0, sd = inm_sigma_r)
  x <- m_theta + r_true

  # Size factors; kept modest to yield sparse-ish counts.
  s <- rlnorm(n_cells, meanlog = log(1500), sdlog = 0.25)

  # Additional covariates matching real panels.
  AP_um <- runif(n_cells, min = 0, max = 2000)
  ML_um <- runif(n_cells, min = 0, max = 2000)

  make_gene <- function(kind, idx) {
    gene <- paste0(kind, "_", idx)

    eta <- rep(base_log_mu, n_cells)

    if (kind %in% c("cycle", "interaction")) {
      # Strong 1st-harmonic oscillation.
      eta <- eta + 0.9 * cos(theta - 0.6)
    }

    if (kind %in% c("spatial", "interaction")) {
      # Monotone gradient along residual depth r.
      eta <- eta + 6.0 * r_true
    }

    if (kind == "interaction") {
      # Gating: cycle amplitude increases with r (position-dependent oscillation).
      eta <- eta + (10.0 * r_true) * cos(theta - 0.6)
    }

    mu <- s * exp(eta)
    y <- rnbinom(n_cells, size = nb_size, mu = mu)

    list(gene = gene, kind = kind, y = y)
  }

  genes <- list()
  for (i in seq_len(n_null)) genes[[length(genes) + 1]] <- make_gene("null", i)
  for (i in seq_len(n_cycle)) genes[[length(genes) + 1]] <- make_gene("cycle", i)
  for (i in seq_len(n_spatial)) genes[[length(genes) + 1]] <- make_gene("spatial", i)
  for (i in seq_len(n_interaction)) genes[[length(genes) + 1]] <- make_gene("interaction", i)

  counts <- do.call(cbind, lapply(genes, function(g) g$y))
  colnames(counts) <- vapply(genes, function(g) g$gene, character(1))

  cells <- data.frame(
    cell_id = seq_len(n_cells),
    x = x,
    r_um = r_true,
    AP_um = AP_um,
    ML_um = ML_um,
    theta = theta,
    s = s,
    stringsAsFactors = FALSE
  )

  truth <- data.frame(
    gene = vapply(genes, function(g) g$gene, character(1)),
    kind = vapply(genes, function(g) g$kind, character(1)),
    stringsAsFactors = FALSE
  )

  list(cells = cells, counts = counts, truth = truth)
}

write_panel_tsv <- function(out_dir, sim) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write.table(sim$cells, file = file.path(out_dir, "cells.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

  counts_df <- data.frame(cell_id = sim$cells$cell_id, sim$counts, check.names = FALSE)
  write.table(counts_df, file = file.path(out_dir, "counts.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

  write.table(sim$truth, file = file.path(out_dir, "truth.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
if (sys.nframe() == 0) {
  if (length(args) < 1) {
    cat("Usage: Rscript scripts/gam/simulate_inm_panel.R OUT_DIR [SEED]\n")
    quit(status = 2)
  }

  out_dir <- args[[1]]
  seed <- if (length(args) >= 2) as.integer(args[[2]]) else 1L

  sim <- simulate_inm_panel(seed = seed)
  write_panel_tsv(out_dir, sim)

  cat(sprintf("Wrote synthetic panel to %s\n", out_dir))
}

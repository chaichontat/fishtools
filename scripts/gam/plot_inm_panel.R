#!/usr/bin/env Rscript

source("scripts/gam/inm_gam.R")

read_panel_tsv <- function(in_dir) {
  cells <- read.delim(file.path(in_dir, "cells.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  counts_df <- read.delim(file.path(in_dir, "counts.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  fit_path <- file.path(in_dir, "fit_results.tsv")
  fit_results <- if (file.exists(fit_path)) {
    read.delim(fit_path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    NULL
  }
  truth_path <- file.path(in_dir, "truth.tsv")
  truth <- if (file.exists(truth_path)) {
    read.delim(truth_path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    NULL
  }

  if (!all(c("cell_id", "x", "theta", "s") %in% names(cells))) {
    stop("cells.tsv must contain: cell_id, x, theta, s")
  }
  if (!("cell_id" %in% names(counts_df))) stop("counts.tsv must contain cell_id.")

  counts_df <- counts_df[match(cells$cell_id, counts_df$cell_id), , drop = FALSE]
  counts_mat <- as.matrix(counts_df[, setdiff(names(counts_df), "cell_id"), drop = FALSE])
  storage.mode(counts_mat) <- "numeric"

  list(cells = cells, counts = counts_mat, fit_results = fit_results, truth = truth)
}

safe_log10p <- function(p) {
  p <- as.numeric(p)
  p[p <= 0] <- .Machine$double.xmin
  -log10(p)
}

plot_coupling <- function(out_png, x, theta, coupling) {
  theta_grid <- seq(0, 2 * pi, length.out = 801)
  theta_grid <- theta_grid[-length(theta_grid)]
  theta_plot <- wrap_theta(theta)
  m_grid <- as.numeric(predict(coupling$fit, newdata = data.frame(theta = theta_grid), type = "response"))

  png(out_png, width = 1400, height = 900, res = 160)
  op <- par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.3, 1.2))
  on.exit({ par(op); dev.off() }, add = TRUE)

  col_pts <- rgb(0, 0, 0, alpha = 0.08)
  plot(theta_plot, x, pch = 16, cex = 0.5, col = col_pts, xlab = "theta (radians)", ylab = "x (depth)",
       main = sprintf("INM coupling (R2=%.3f)", coupling$r2))
  lines(theta_grid, m_grid, col = "#d62728", lwd = 2.5)

  hist(coupling$r_hat, breaks = 40, col = "grey80", border = "white",
       xlab = "r = x - m(theta)", main = "Residual depth distribution")

  plot(theta_plot, coupling$r_hat, pch = 16, cex = 0.5, col = col_pts,
       xlab = "theta (radians)", ylab = "r", main = "Residual vs theta")
  abline(h = 0, col = "grey40", lwd = 1)

  qqnorm(coupling$r_hat, pch = 16, cex = 0.5, col = col_pts, main = "Residual QQ-plot")
  qqline(coupling$r_hat, col = "grey40", lwd = 1.2)
}

plot_pvals <- function(out_png, fit_results, truth) {
  df <- fit_results
  if (!is.null(truth)) {
    df <- merge(df, truth, by = "gene", all.x = TRUE, sort = FALSE)
  }
  if (!("kind" %in% names(df))) df$kind <- "unknown"

  png(out_png, width = 1400, height = 900, res = 160)
  op <- par(mfrow = c(1, 3), mar = c(7.5, 4.2, 2.0, 1.0))
  on.exit({ par(op); dev.off() }, add = TRUE)

  kinds <- unique(df$kind)
  kinds <- kinds[order(kinds)]

  bxp <- function(values, title) {
    vals_by_kind <- split(values, df$kind)
    vals_by_kind <- vals_by_kind[kinds]
    boxplot(vals_by_kind, las = 2, main = title, ylab = "-log10(p)", col = "grey85", border = "grey35")
    abline(h = 2, col = "#d62728", lty = 2) # ~ p=0.01
  }

  bxp(safe_log10p(df$p_spatial), "Spatial: s(r)")
  bxp(safe_log10p(df$p_cycle), "Cycle: s(theta)")
  bxp(safe_log10p(df$p_interaction), "Interaction: ti(r,theta)")
}

pick_representative_genes <- function(fit_results, truth) {
  df <- fit_results
  if (!is.null(truth)) df <- merge(df, truth, by = "gene", all.x = TRUE, sort = FALSE)

  pick_min <- function(kind, col) {
    sub <- df
    if ("kind" %in% names(sub)) sub <- sub[sub$kind == kind, , drop = FALSE]
    if (nrow(sub) == 0) return(NA_character_)
    sub <- sub[order(sub[[col]]), , drop = FALSE]
    as.character(sub$gene[[1]])
  }

  list(
    interaction = pick_min("interaction", "p_interaction"),
    cycle = pick_min("cycle", "p_cycle"),
    spatial = pick_min("spatial", "p_spatial")
  )
}

plot_gene_surface <- function(out_png, gene, y, s, r, theta) {
  fit <- fit_gene_gam(y = y, sf = s, r = r, theta = theta)

  r_seq <- as.numeric(quantile(r, probs = seq(0.02, 0.98, length.out = 60)))
  theta_seq <- seq(0, 2 * pi, length.out = 90)

  grid <- expand.grid(r = r_seq, theta = theta_seq)
  grid$sf <- median(s)

  mu <- as.numeric(predict(fit, newdata = grid, type = "response"))
  z <- matrix(mu, nrow = length(r_seq), ncol = length(theta_seq), byrow = FALSE)
  z_plot <- log10(z + 1)

  png(out_png, width = 1400, height = 900, res = 160)
  op <- par(mfrow = c(1, 1), mar = c(4.5, 4.5, 2.5, 1.2))
  on.exit({ par(op); dev.off() }, add = TRUE)

  image(
    x = r_seq,
    y = theta_seq,
    z = z_plot,
    xlab = "r = x - m(theta)",
    ylab = "theta (radians)",
    main = sprintf("Fitted surface: %s (log10(mu+1))", gene),
    col = hcl.colors(64, "YlOrRd", rev = FALSE)
  )
  contour(x = r_seq, y = theta_seq, z = z_plot, add = TRUE, drawlabels = FALSE, col = "grey20")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript scripts/gam/plot_inm_panel.R IN_DIR [OUT_DIR]\n")
  quit(status = 2)
}

in_dir <- args[[1]]
out_dir <- if (length(args) >= 2) args[[2]] else in_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

panel <- read_panel_tsv(in_dir)
cells <- panel$cells

coupling <- fit_inm_coupling_gam(x = cells$x, theta = cells$theta, k_theta = 8)
r <- coupling$r_hat

plot_coupling(
  out_png = file.path(out_dir, "coupling.png"),
  x = cells$x,
  theta = cells$theta,
  coupling = coupling
)

if (!is.null(panel$fit_results)) {
  plot_pvals(
    out_png = file.path(out_dir, "pvals.png"),
    fit_results = panel$fit_results,
    truth = panel$truth
  )

  reps <- pick_representative_genes(panel$fit_results, panel$truth)
  genes <- unique(unlist(reps, use.names = FALSE))
  genes <- genes[!is.na(genes)]

  if (length(genes) > 0) {
    gene_names <- colnames(panel$counts)
    for (gene in genes) {
      idx <- which(gene_names == gene)
      if (length(idx) != 1) next
      plot_gene_surface(
        out_png = file.path(out_dir, paste0("surface_", gene, ".png")),
        gene = gene,
        y = panel$counts[, idx],
        s = cells$s,
        r = r,
        theta = cells$theta
      )
    }
  }
}

cat(sprintf("Wrote plots to %s\n", out_dir))

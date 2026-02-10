#!/usr/bin/env Rscript

source("scripts/gam/inm_gam.R")

read_panel_tsv <- function(in_dir) {
  cells <- read.delim(file.path(in_dir, "cells.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  counts_df <- read.delim(file.path(in_dir, "counts.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
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

  list(cells = cells, counts = counts_mat, truth = truth)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript scripts/gam/fit_inm_panel.R IN_DIR [OUT_TSV]\n")
  quit(status = 2)
}

in_dir <- args[[1]]
out_tsv <- if (length(args) >= 2) args[[2]] else file.path(in_dir, "fit_results.tsv")

panel <- read_panel_tsv(in_dir)

cells <- panel$cells
if (!is.numeric(cells$s) || any(!is.finite(cells$s)) || any(cells$s <= 0)) stop("cells$s must be numeric, finite, and > 0.")

write_qbh <- function(out_tsv) {
  if (!file.exists(out_tsv) || file.info(out_tsv)$size <= 0) {
    stop(sprintf("Cannot compute q-values: missing/empty %s", out_tsv))
  }
  df <- read.delim(out_tsv, stringsAsFactors = FALSE, check.names = FALSE)
  pcols <- grep("^p_", names(df), value = TRUE)
  if (length(pcols) == 0) {
    cat("No p_* columns found; skipping q-value computation.\n")
    return(invisible(NULL))
  }
  for (pc in pcols) {
    p <- df[[pc]]
    ok <- is.finite(p)
    q <- rep(NA_real_, length(p))
    if (any(ok)) {
      q[ok] <- p.adjust(p[ok], method = "BH")
    }
    df[[sub("^p_", "q_", pc)]] <- q
  }
  q_path <- if (grepl("\\.tsv$", out_tsv)) sub("\\.tsv$", ".qbh.tsv", out_tsv) else paste0(out_tsv, ".qbh.tsv")
  write.table(df, file = q_path, sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("Wrote BH q-values (per term across genes) to %s\n", q_path))
  invisible(q_path)
}

batch <- NULL
batch_ref <- NULL
if ("source" %in% names(cells)) {
  batch <- as.factor(cells$source)
  batch_ref <- levels(batch)[[1]]
  cat(sprintf("Using batch term from cells$source (n_levels=%d; ref=%s)\n", nlevels(batch), batch_ref))
}
# Resumable fitting: append per-gene results to OUT_TSV and skip genes already present.
coupling <- fit_inm_coupling_gam(x = cells$x, theta = cells$theta, k_theta = 8, group = batch)
r <- coupling$r_hat

counts <- panel$counts
genes <- colnames(counts)
if (is.null(genes)) genes <- paste0("gene_", seq_len(ncol(counts)))

default_out <- file.path(in_dir, "fit_results.tsv")
fits_dir <- if (identical(out_tsv, default_out)) {
  file.path(dirname(out_tsv), "fits_rds")
} else {
  stem <- sub("\\.tsv$", "", basename(out_tsv))
  file.path(dirname(out_tsv), paste0("fits_rds__", stem))
}
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)

min_nonzero <- 10

brdu_pos <- NULL
edu_pos <- NULL
if (all(c("brdu_pos", "edu_pos") %in% names(cells))) {
  brdu_pos <- as.numeric(cells$brdu_pos)
  edu_pos <- as.numeric(cells$edu_pos)
  if (any(!is.finite(brdu_pos)) || any(!is.finite(edu_pos))) stop("brdu_pos/edu_pos must be finite.")
  if (any(!(brdu_pos %in% c(0, 1))) || any(!(edu_pos %in% c(0, 1)))) stop("brdu_pos/edu_pos must be 0/1.")
  cat("Using positivity covariates: brdu_pos, edu_pos (and brdu_pos:edu_pos)\n")
}

done <- character(0)
if (file.exists(out_tsv) && file.info(out_tsv)$size > 0) {
  existing <- read.delim(out_tsv, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("gene" %in% names(existing))) {
    stop("Cannot resume: existing OUT_TSV is missing a 'gene' column.")
  }
  expected_cols <- c(
    "gene", "inm_r2",
    "p_spatial", "p_cycle", "p_interaction",
    "p_brdu_pos", "p_edu_pos", "p_brdu_edu",
    "cycle_amp_link", "spatial_grad_link", "gating_index_link"
  )
  if (!identical(names(existing), expected_cols)) {
    stop("Cannot resume: existing OUT_TSV columns differ from expected output columns (likely from an older version).")
  }
  done <- unique(as.character(existing$gene))
  done <- done[!is.na(done) & nzchar(done)]
  cat(sprintf("Resuming: found %d genes already in %s\n", length(done), out_tsv))
}

write_header <- !(file.exists(out_tsv) && file.info(out_tsv)$size > 0)
remaining <- sum(!(genes %in% done))
cat(sprintf("Total genes=%d; remaining=%d\n", length(genes), remaining))
if (remaining == 0) {
  cat(sprintf("Nothing to do; %s already contains all genes.\n", out_tsv))
  write_qbh(out_tsv)
  quit(status = 0)
}

for (j in seq_along(genes)) {
  gene <- genes[[j]]
  safe_gene <- gsub("[^A-Za-z0-9._-]+", "_", gene)
  fit_path <- file.path(fits_dir, sprintf("%04d_%s.gam.rds", j, safe_gene))
  already_done <- gene %in% done
  if (already_done && file.exists(fit_path)) next

  t0 <- proc.time()[[3]]
  cat(sprintf("[%d/%d] Fitting %s\n", j, length(genes), gene))
  write_row <- !already_done
  y <- counts[, j]

  if (!is.numeric(y) || any(!is.finite(y)) || any(y < 0)) {
    cat("  invalid y (non-numeric, non-finite, or negative); skipping\n")
    if (write_row) {
      row <- data.frame(
        gene = gene,
        inm_r2 = coupling$r2,
        p_spatial = NA_real_,
        p_cycle = NA_real_,
        p_interaction = NA_real_,
        p_brdu_pos = NA_real_,
        p_edu_pos = NA_real_,
        p_brdu_edu = NA_real_,
        cycle_amp_link = NA_real_,
        spatial_grad_link = NA_real_,
        gating_index_link = NA_real_,
        stringsAsFactors = FALSE
      )
      if (write_header) {
        write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
        write_header <- FALSE
      } else {
        write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
      }
      done <- c(done, gene)
    }
    next
  }

  nonzero <- sum(y > 0)
  if (nonzero < min_nonzero) {
    cat(sprintf("  too sparse (nonzero=%d < %d); skipping\n", nonzero, min_nonzero))
    if (write_row) {
      row <- data.frame(
        gene = gene,
        inm_r2 = coupling$r2,
        p_spatial = NA_real_,
        p_cycle = NA_real_,
        p_interaction = NA_real_,
        p_brdu_pos = NA_real_,
        p_edu_pos = NA_real_,
        p_brdu_edu = NA_real_,
        cycle_amp_link = NA_real_,
        spatial_grad_link = NA_real_,
        gating_index_link = NA_real_,
        stringsAsFactors = FALSE
      )
      if (write_header) {
        write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
        write_header <- FALSE
      } else {
        write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
      }
      done <- c(done, gene)
    }
    next
  }

  fit <- fit_gene_gam(
    y = y,
    sf = cells$s,
    r = r,
    theta = cells$theta,
    brdu_pos = brdu_pos,
    edu_pos = edu_pos,
    batch = batch
  )

  pv <- extract_component_pvals(fit)
  es <- effect_sizes_from_fit(
    fit,
    r = r,
    has_pos = !is.null(brdu_pos) && !is.null(edu_pos),
    has_batch = !is.null(batch),
    batch_ref = batch_ref
  )

  # Save per-gene fit for coefficients/basis inspection.
  fit_save <- fit
  # Strip large per-cell vectors; this keeps coefficients + smooth objects.
  for (k in c("model", "y", "residuals", "fitted.values", "linear.predictors", "prior.weights", "weights", "offset", "data")) {
    if (k %in% names(fit_save)) fit_save[[k]] <- NULL
  }
  tmp_path <- paste0(fit_path, ".tmp")
  saveRDS(fit_save, file = tmp_path)
  ok_rename <- file.rename(tmp_path, fit_path)
  if (!ok_rename) {
    file.copy(tmp_path, fit_path, overwrite = TRUE)
    unlink(tmp_path)
  }

  row <- data.frame(
    gene = gene,
    inm_r2 = coupling$r2,
    p_spatial = pv$p_spatial,
    p_cycle = pv$p_cycle,
    p_interaction = pv$p_interaction,
    p_brdu_pos = pv$p_brdu_pos,
    p_edu_pos = pv$p_edu_pos,
    p_brdu_edu = pv$p_brdu_edu,
    cycle_amp_link = es$cycle_amp_link,
    spatial_grad_link = es$spatial_grad_link,
    gating_index_link = es$gating_index_link,
    stringsAsFactors = FALSE
  )

  if (write_row) {
    if (write_header) {
      write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
      write_header <- FALSE
    } else {
      write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
    }
    done <- c(done, gene)
  }

  dt <- proc.time()[[3]] - t0
  cat(sprintf("  done %s (%.2fs): p_spatial=%.3g p_cycle=%.3g p_interaction=%.3g\n",
              gene, dt, row$p_spatial, row$p_cycle, row$p_interaction))
  cat(sprintf("  saved %s\n", fit_path))
  flush.console()
}

cat(sprintf("Wrote fit results to %s\n", out_tsv))
write_qbh(out_tsv)

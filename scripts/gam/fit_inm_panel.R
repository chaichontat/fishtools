#!/usr/bin/env Rscript

read_panel_tsv <- function(in_dir, require_theta = TRUE) {
  cells <- read.delim(file.path(in_dir, "cells.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  counts_df <- read.delim(file.path(in_dir, "counts.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  truth_path <- file.path(in_dir, "truth.tsv")
  truth <- if (file.exists(truth_path)) {
    read.delim(truth_path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    NULL
  }

  required_cols <- c("cell_id", "x", "s")
  if (isTRUE(require_theta)) required_cols <- c(required_cols, "theta")
  if (!all(required_cols %in% names(cells))) {
    stop(sprintf("cells.tsv must contain: %s", paste(required_cols, collapse = ", ")))
  }
  if (!("cell_id" %in% names(counts_df))) stop("counts.tsv must contain cell_id.")

  counts_df <- counts_df[match(cells$cell_id, counts_df$cell_id), , drop = FALSE]
  counts_mat <- as.matrix(counts_df[, setdiff(names(counts_df), "cell_id"), drop = FALSE])
  storage.mode(counts_mat) <- "numeric"

  list(cells = cells, counts = counts_mat, truth = truth)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript scripts/gam/fit_inm_panel.R IN_DIR [OUT_TSV] [--no-pos] [--no-theta] [--threads N] [--omp-threads N] [--basis shrink|standard] [--k-uv N] [--priority-genes CSV] [--heartbeat-sec N] [--no-diagnostics] [--diagnostics] [--diagnostics-all] [--diagnostics-p P]\n")
  cat("  --threads N controls gene-parallelism (N genes fit concurrently; default: 1); per-gene mgcv threading is forced to 1.\n")
  cat("  --omp-threads N controls OMP_NUM_THREADS (default: 8).\n")
  cat("  --k-uv N controls the AP/ML smooth basis dimension k for s(AP_um, ML_um) (default: 15).\n")
  cat("  --heartbeat-sec N emits periodic 'still running' logs while a chunk is fitting (default: 60).\n")
  cat("  --no-theta removes theta smooth terms (s(theta) and ti(r_um,theta)).\n")
  cat("  Diagnostics are written by default (per gene); use --no-diagnostics to disable.\n")
  quit(status = 2)
}

in_dir <- args[[1]]
out_tsv <- if (length(args) >= 2) args[[2]] else file.path(in_dir, "fit_results.tsv")
fit_error_path <- if (grepl("\\.tsv$", out_tsv)) sub("\\.tsv$", ".errors.tsv", out_tsv) else paste0(out_tsv, ".errors.tsv")
opts <- if (length(args) >= 3) args[3:length(args)] else character(0)
parse_opts <- function(opts) {
  disable_pos <- FALSE
  use_theta <- TRUE
  threads <- 1L
  omp_threads <- 8L
  basis <- "standard"
  k_uv <- 15L
  priority_genes <- c("Eomes", "Nr2f2")
  heartbeat_sec <- 60L
  diagnostics <- TRUE
  diagnostics_all <- TRUE
  diagnostics_p <- 0.05
  i <- 1L
  while (i <= length(opts)) {
    opt <- opts[[i]]
    if (opt == "--no-pos") {
      disable_pos <- TRUE
      i <- i + 1L
      next
    }
    if (opt == "--no-theta") {
      use_theta <- FALSE
      i <- i + 1L
      next
    }
    if (opt == "--no-diagnostics") {
      diagnostics <- FALSE
      diagnostics_all <- FALSE
      i <- i + 1L
      next
    }
    if (opt == "--threads") {
      if (i == length(opts)) stop("--threads requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 1) stop("--threads must be an integer >= 1.")
      threads <- val
      i <- i + 2L
      next
    }
    if (opt == "--omp-threads") {
      if (i == length(opts)) stop("--omp-threads requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 1) stop("--omp-threads must be an integer >= 1.")
      omp_threads <- val
      i <- i + 2L
      next
    }
    if (opt == "--basis") {
      if (i == length(opts)) stop("--basis requires a value: shrink|standard.")
      val <- as.character(opts[[i + 1L]])
      if (!(val %in% c("shrink", "standard"))) stop("--basis must be one of: shrink, standard.")
      basis <- val
      i <- i + 2L
      next
    }
    if (opt == "--k-uv") {
      if (i == length(opts)) stop("--k-uv requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 4) stop("--k-uv must be an integer >= 4.")
      k_uv <- val
      i <- i + 2L
      next
    }
    if (opt == "--priority-genes") {
      if (i == length(opts)) stop("--priority-genes requires a comma-separated list (CSV).")
      val <- as.character(opts[[i + 1L]])
      items <- unlist(strsplit(val, ",", fixed = TRUE), use.names = FALSE)
      items <- trimws(items)
      items <- items[nzchar(items)]
      priority_genes <- unique(as.character(items))
      i <- i + 2L
      next
    }
    if (opt == "--heartbeat-sec") {
      if (i == length(opts)) stop("--heartbeat-sec requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 0) stop("--heartbeat-sec must be an integer >= 0.")
      heartbeat_sec <- val
      i <- i + 2L
      next
    }
    if (opt == "--diagnostics") {
      diagnostics <- TRUE
      diagnostics_all <- FALSE
      i <- i + 1L
      next
    }
    if (opt == "--diagnostics-all") {
      diagnostics <- TRUE
      diagnostics_all <- TRUE
      i <- i + 1L
      next
    }
    if (opt == "--diagnostics-p") {
      if (i == length(opts)) stop("--diagnostics-p requires a numeric value.")
      val <- suppressWarnings(as.numeric(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 0 || val > 1) stop("--diagnostics-p must be a number in [0, 1].")
      diagnostics_p <- val
      diagnostics <- TRUE
      diagnostics_all <- FALSE
      i <- i + 2L
      next
    }
    stop(sprintf("Unknown option: %s", opt))
  }
  list(
    disable_pos = disable_pos,
    use_theta = use_theta,
    threads = threads,
    omp_threads = omp_threads,
    basis = basis,
    k_uv = k_uv,
    priority_genes = priority_genes,
    heartbeat_sec = heartbeat_sec,
    diagnostics = diagnostics,
    diagnostics_all = diagnostics_all,
    diagnostics_p = diagnostics_p
  )
}

parsed <- parse_opts(opts)
disable_pos <- parsed$disable_pos
use_theta <- parsed$use_theta
threads <- parsed$threads
omp_threads <- parsed$omp_threads
basis <- parsed$basis
k_uv <- parsed$k_uv
priority_genes <- parsed$priority_genes
heartbeat_sec <- parsed$heartbeat_sec
diagnostics <- parsed$diagnostics
diagnostics_all <- parsed$diagnostics_all
diagnostics_p <- parsed$diagnostics_p

Sys.setenv(OMP_NUM_THREADS = as.character(as.integer(omp_threads)))
cat(sprintf("Setting OMP_NUM_THREADS=%d\n", as.integer(omp_threads)))

source("scripts/gam/inm_gam.R")

panel <- read_panel_tsv(in_dir, require_theta = isTRUE(use_theta))

	cells <- panel$cells
	if (!is.numeric(cells$s) || any(!is.finite(cells$s)) || any(cells$s <= 0)) stop("cells$s must be numeric, finite, and > 0.")
	  if (!all(c("r_um", "AP_um", "ML_um") %in% names(cells))) {
	  stop("cells.tsv must contain: r_um, AP_um, ML_um (in addition to cell_id, x, s, and theta when theta terms are enabled)")
	}
  r_um_all <- as.numeric(cells$r_um)
  if (any(!is.finite(r_um_all))) stop("cells$r_um must be finite.")
  GAM_GAMMA <- 2.5
  # Additional hard filter: drop rows with non-finite AP/ML coordinates.
  ap_um_all <- as.numeric(cells$AP_um)
  ml_um_all <- as.numeric(cells$ML_um)
  keep_apml <- is.finite(ap_um_all) & is.finite(ml_um_all)
  n_keep_apml <- sum(keep_apml)
  if (!is.finite(n_keep_apml) || n_keep_apml <= 0) stop("No cells remain after requiring finite AP_um and ML_um.")
  if (n_keep_apml < length(keep_apml)) {
    cat(sprintf("Filtering cells: keeping %d/%d with finite AP_um/ML_um\n", n_keep_apml, length(keep_apml)))
    cells <- cells[keep_apml, , drop = FALSE]
    panel$counts <- panel$counts[keep_apml, , drop = FALSE]
    panel$cells <- cells
  }

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
  clamp_p <- function(p) {
    p <- as.numeric(p)
    p[!is.finite(p)] <- NA_real_
    p[p < 0] <- 0.0
    p[p > 1] <- 1.0
    p
  }
  for (pc in pcols) {
    p <- clamp_p(df[[pc]])
    df[[pc]] <- p
    ok <- is.finite(p)
    q <- rep(NA_real_, length(p))
    if (any(ok)) {
      q[ok] <- p.adjust(p[ok], method = "BH")
    }
    df[[sub("^p_", "q_", pc)]] <- q
  }
  q_path <- if (grepl("\\.tsv$", out_tsv)) sub("\\.tsv$", ".qbh.tsv", out_tsv) else paste0(out_tsv, ".qbh.tsv")
  # Avoid `write.table()` padding column names/fields with spaces (breaks downstream TSV readers).
  write_tsv_simple <- function(df, path) {
    con <- file(path, open = "w")
    on.exit(close(con), add = TRUE)
    writeLines(paste(names(df), collapse = "\t"), con = con)
    n <- nrow(df)
    for (i in seq_len(n)) {
      row <- df[i, , drop = FALSE]
      vals <- lapply(row, function(x) {
        if (length(x) != 1) x <- x[[1]]
        if (is.na(x)) return("NA")
        if (is.numeric(x)) return(format(x, digits = 16, scientific = TRUE, trim = TRUE))
        as.character(x)
      })
      writeLines(paste(unlist(vals, use.names = FALSE), collapse = "\t"), con = con)
    }
  }
  write_tsv_simple(df, q_path)
  cat(sprintf("Wrote BH q-values (per term across genes) to %s\n", q_path))
  invisible(q_path)
}

if (!("batch" %in% names(cells))) {
  stop("cells.tsv must contain a 'batch' column for the batch correction model.")
}
batch <- as.factor(cells$batch)
batch_ref <- levels(batch)[[1]]
cat(sprintf("Using batch factor from cells$batch (n_levels=%d; ref=%s)\n", nlevels(batch), batch_ref))

extract_animal_from_dataset <- function(x) {
  s <- as.character(x)
  m <- regexpr("JaxA[0-9]+", s, perl = TRUE)
  out <- ifelse(m > 0, regmatches(s, m), NA_character_)
  out
}

animal_ref <- NULL
if ("animal" %in% names(cells)) {
  animal <- as.factor(cells$animal)
  if (any(is.na(animal))) stop("cells$animal must not contain NA.")
  animal_ref <- levels(animal)[[1]]
  cat(sprintf("Using animal factor from cells$animal (n_levels=%d; ref=%s)\n", nlevels(animal), animal_ref))
} else {
  a0 <- extract_animal_from_dataset(cells$batch)
  if (any(is.na(a0) | !nzchar(a0))) {
    bad <- unique(cells$batch[is.na(a0) | !nzchar(a0)])
    stop(sprintf(
      "Could not parse animal (JaxA*) from cells$batch for dataset(s): %s.",
      paste(bad, collapse = ",")
    ))
  }
  animal <- as.factor(a0)
  animal_ref <- levels(animal)[[1]]
  cat(sprintf("Using animal factor parsed from cells$batch (n_levels=%d; ref=%s)\n", nlevels(animal), animal_ref))
}

batch_model <- batch
batch_ref_model <- batch_ref

  theta_model <- if ("theta" %in% names(cells)) as.numeric(cells$theta) else rep(0.0, nrow(cells))
  if (any(!is.finite(theta_model))) stop("cells$theta must be finite when present.")
  coupling_r2 <- NA_real_
  if (isTRUE(use_theta)) {
	  # Resumable fitting: append per-gene results to OUT_TSV and skip genes already present.
	  coupling_group <- if (!is.null(animal)) animal else batch
	  coupling <- fit_inm_coupling_gam(x = cells$x, theta = theta_model, k_theta = 8, group = coupling_group)
    coupling_r2 <- coupling$r2
  } else {
    cat("Theta disabled via --no-theta: skipping coupling fit and setting inm_r2=NA\n")
  }
	r_um <- as.numeric(cells$r_um)
	AP_um <- as.numeric(cells$AP_um)
	ML_um <- as.numeric(cells$ML_um)
	if (any(!is.finite(r_um))) stop("cells$r_um must be finite.")
	if (any(!is.finite(AP_um)) || any(!is.finite(ML_um))) stop("cells$AP_um/ML_um must be finite.")
	ap_unique <- length(unique(AP_um))
	ml_unique <- length(unique(ML_um))
	if (ap_unique <= 1 || ml_unique <= 1) {
	  cat(sprintf("AP/ML smooth terms disabled: AP_um unique=%d, ML_um unique=%d\n", ap_unique, ml_unique))
	}

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

diagnostics_dir <- NULL
if (isTRUE(diagnostics)) {
  diagnostics_dir <- if (identical(out_tsv, default_out)) {
    file.path(dirname(out_tsv), "diagnostics_gam")
  } else {
    stem <- sub("\\.tsv$", "", basename(out_tsv))
    file.path(dirname(out_tsv), paste0("diagnostics_gam__", stem))
  }
  dir.create(diagnostics_dir, showWarnings = FALSE, recursive = TRUE)
  if (isTRUE(diagnostics_all)) {
    cat(sprintf("Diagnostics enabled for all fitted genes (dir=%s)\n", diagnostics_dir))
  } else {
    cat(sprintf("Diagnostics enabled for hits with min(p_spatial,p_cycle,p_interaction) <= %.3g (dir=%s)\n", diagnostics_p, diagnostics_dir))
  }
}

min_nonzero <- 10

brdu_pos <- NULL
edu_pos <- NULL
if (!disable_pos) {
  if (!all(c("brdu_pos", "edu_pos") %in% names(cells))) {
    stop(
      "Missing positivity covariates in cells.tsv (need brdu_pos and edu_pos). ",
      "Re-export the panel with positivity enabled (requires *.brdu_edu_thresholds.json), ",
      "or pass --no-pos to disable these terms."
    )
  }
  brdu_pos <- as.numeric(cells$brdu_pos)
  edu_pos <- as.numeric(cells$edu_pos)
  if (any(!is.finite(brdu_pos)) || any(!is.finite(edu_pos))) stop("brdu_pos/edu_pos must be finite.")
  if (any(!(brdu_pos %in% c(0, 1))) || any(!(edu_pos %in% c(0, 1)))) stop("brdu_pos/edu_pos must be 0/1.")
  if (length(unique(brdu_pos)) < 2 || length(unique(edu_pos)) < 2) {
    stop("brdu_pos/edu_pos must each have >=2 levels after filtering; otherwise terms are not identifiable.")
  }
  cat("Using positivity covariates: brdu_pos, edu_pos (and brdu_pos:edu_pos)\n")
} else {
  cat("Positivity covariates disabled via --no-pos\n")
}

done <- character(0)
	if (file.exists(out_tsv) && file.info(out_tsv)$size > 0) {
  existing <- read.delim(out_tsv, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("gene" %in% names(existing))) {
    stop("Cannot resume: existing OUT_TSV is missing a 'gene' column.")
  }
		  expected_cols <- c(
		    "gene", "inm_r2",
		    "p_spatial", "p_cycle", "p_interaction", "p_apml", "p_apml_r_um",
		    "p_brdu_pos", "p_edu_pos", "p_brdu_edu",
	      "log_p_spatial", "log_p_cycle", "log_p_interaction", "log_p_apml", "log_p_apml_r_um",
      "log_p_brdu_pos", "log_p_edu_pos", "log_p_brdu_edu",
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
remaining_rows <- sum(!(genes %in% done))
if (threads > 1 && .Platform$OS.type != "unix") {
  stop("--threads > 1 requires a Unix-like OS (forking).")
}
cat(sprintf("Total genes=%d; remaining_rows=%d; threads=%d; basis=%s\n", length(genes), remaining_rows, threads, basis))
cat(sprintf("AP/ML smooth basis dimension: k_uv=%d\n", as.integer(k_uv)))
cat(sprintf("Theta terms enabled: %s\n", if (isTRUE(use_theta)) "yes" else "no"))
cat(sprintf("Heartbeat interval: %ds (set --heartbeat-sec 0 to disable)\n", as.integer(heartbeat_sec)))

write_fit_error <- function(gene, msg) {
  err_row <- data.frame(gene = gene, error = msg, stringsAsFactors = FALSE)
  write_err_header <- !(file.exists(fit_error_path) && file.info(fit_error_path)$size > 0)
  if (write_err_header) {
    write.table(err_row, file = fit_error_path, sep = "\t", row.names = FALSE, quote = FALSE)
  } else {
    write.table(err_row, file = fit_error_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  }
}

write_fit_row <- function(row) {
  if (write_header) {
    write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
    write_header <<- FALSE
  } else {
    write.table(row, file = out_tsv, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  }
}

na_row <- function(gene) {
  data.frame(
    gene = gene,
    inm_r2 = coupling_r2,
    p_spatial = NA_real_,
    p_cycle = NA_real_,
    p_interaction = NA_real_,
    p_apml = NA_real_,
    p_apml_r_um = NA_real_,
    p_brdu_pos = NA_real_,
    p_edu_pos = NA_real_,
    p_brdu_edu = NA_real_,
    log_p_spatial = NA_real_,
    log_p_cycle = NA_real_,
    log_p_interaction = NA_real_,
    log_p_apml = NA_real_,
    log_p_apml_r_um = NA_real_,
    log_p_brdu_pos = NA_real_,
    log_p_edu_pos = NA_real_,
    log_p_brdu_edu = NA_real_,
    cycle_amp_link = NA_real_,
    spatial_grad_link = NA_real_,
    gating_index_link = NA_real_,
    stringsAsFactors = FALSE
  )
}

write_concurvity <- function(out_tsv, fit) {
  cv <- tryCatch(
    mgcv::concurvity(fit, full = TRUE),
    error = function(e) e
  )
  if (inherits(cv, "error")) {
    write.table(
      data.frame(note = paste0("concurvity() failed: ", conditionMessage(cv)), stringsAsFactors = FALSE),
      file = out_tsv,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
    return(invisible(NULL))
  }
  mats <- list()
  if (is.matrix(cv)) {
    mats[["matrix"]] <- cv
  } else if (is.list(cv)) {
    for (nm in names(cv)) {
      if (is.matrix(cv[[nm]])) mats[[nm]] <- cv[[nm]]
    }
  }

  to_long <- function(mat, kind) {
    rn <- rownames(mat)
    cn <- colnames(mat)
    if (is.null(rn)) rn <- paste0("row_", seq_len(nrow(mat)))
    if (is.null(cn)) cn <- paste0("col_", seq_len(ncol(mat)))
    row <- rep(rn, times = length(cn))
    col <- rep(cn, each = length(rn))
    data.frame(
      kind = kind,
      row = row,
      col = col,
      value = as.numeric(c(mat)),
      stringsAsFactors = FALSE
    )
  }

  if (length(mats) == 0) {
    write.table(
      data.frame(note = "concurvity() did not return a matrix/list-of-matrices", stringsAsFactors = FALSE),
      file = out_tsv,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
    return(invisible(NULL))
  }

  out <- do.call(rbind, lapply(names(mats), function(nm) to_long(mats[[nm]], nm)))
  write.table(out, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  invisible(out)
}

write_edf_table <- function(out_tsv, fit) {
  s <- summary(fit)
  st <- s$s.table
  if (is.null(st)) {
    write.table(
      data.frame(note = "summary(fit)$s.table is NULL", stringsAsFactors = FALSE),
      file = out_tsv,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
    return(invisible(NULL))
  }
  if (!all(c("edf", "Ref.df") %in% colnames(st))) {
    write.table(
      data.frame(note = "s.table missing edf/Ref.df columns", stringsAsFactors = FALSE),
      file = out_tsv,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
    return(invisible(NULL))
  }
  df <- data.frame(
    term = rownames(st),
    edf = as.numeric(st[, "edf"]),
    ref_df = as.numeric(st[, "Ref.df"]),
    stringsAsFactors = FALSE
  )
  df$edf_over_refdf <- df$edf / pmax(df$ref_df, 1e-9)
  df$warn_edf_near_refdf <- is.finite(df$edf_over_refdf) & (df$edf_over_refdf > 0.9)
  write.table(df, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  invisible(df)
}

write_k_check <- function(out_tsv, fit) {
  kc <- mgcv::k.check(fit)
  if (!(is.matrix(kc) || is.data.frame(kc))) {
    stop("k.check() did not return a matrix/data.frame.")
  }
  kc_df <- as.data.frame(kc, stringsAsFactors = FALSE)
  kc_df$term <- rownames(kc)
  rownames(kc_df) <- NULL
  write.table(kc_df, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  invisible(kc_df)
}

run_diagnostics <- function(fit, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  sum_path <- file.path(out_dir, "summary.txt")
  capture.output(summary(fit), file = sum_path)

  png(file.path(out_dir, "gam_check.png"), width = 1600, height = 1000, res = 180)
  op <- par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.3, 1.2))
  on.exit({ par(op); dev.off() }, add = TRUE)
  invisible(mgcv::gam.check(fit))

  write_k_check(file.path(out_dir, "k_check.tsv"), fit)
  write_edf_table(file.path(out_dir, "edf.tsv"), fit)
  write_concurvity(file.path(out_dir, "concurvity.tsv"), fit)
  invisible(sum_path)
}

tasks <- list()
for (j in seq_along(genes)) {
  gene <- genes[[j]]
  safe_gene <- gsub("[^A-Za-z0-9._-]+", "_", gene)
  fit_path <- file.path(fits_dir, sprintf("%04d_%s.gam.rds", j, safe_gene))
  diag_path <- if (is.null(diagnostics_dir)) NULL else file.path(diagnostics_dir, sprintf("%04d_%s", j, safe_gene))
  already_done <- gene %in% done
  if (already_done && file.exists(fit_path)) next
  tasks[[length(tasks) + 1]] <- list(j = j, gene = gene, fit_path = fit_path, diag_path = diag_path, already_done = already_done)
}

if (length(tasks) > 1 && length(priority_genes) > 0) {
  pr_rank <- vapply(tasks, function(t) {
    m <- match(as.character(t$gene), priority_genes)
    if (is.na(m)) Inf else as.numeric(m)
  }, numeric(1))
  js <- vapply(tasks, function(t) as.integer(t$j), integer(1))
  tasks <- tasks[order(pr_rank, js)]
  pri_remaining <- intersect(priority_genes, vapply(tasks, function(t) as.character(t$gene), character(1)))
  if (length(pri_remaining) > 0) {
    cat(sprintf("Prioritizing %d gene(s): %s\n", length(pri_remaining), paste(pri_remaining, collapse = ",")))
  }
}

if (length(tasks) == 0) {
  cat(sprintf("Nothing to do; all per-gene .gam.rds files exist and %s is up to date.\n", out_tsv))
  if (file.exists(out_tsv) && file.info(out_tsv)$size > 0) write_qbh(out_tsv)
  quit(status = 0)
}

fit_one <- function(task) {
  j <- task$j
  gene <- task$gene
  fit_path <- task$fit_path
  diag_path <- task$diag_path
  already_done <- task$already_done
  write_row <- !already_done
  cat(sprintf("[%d/%d] start %s\n", j, length(genes), gene))
  flush.console()

  y <- counts[, j]
  if (!is.numeric(y) || any(!is.finite(y)) || any(y < 0)) {
    return(list(j = j, gene = gene, write_row = write_row, row = na_row(gene), fit_path = fit_path, dt = NA_real_, err = NULL, p = c(NA_real_, NA_real_, NA_real_), diag = NULL, diag_err = NULL))
  }
  nonzero <- sum(y > 0)
  if (nonzero < min_nonzero) {
    return(list(j = j, gene = gene, write_row = write_row, row = na_row(gene), fit_path = fit_path, dt = NA_real_, err = NULL, p = c(NA_real_, NA_real_, NA_real_), diag = NULL, diag_err = NULL))
  }

  t0 <- proc.time()[[3]]
  fit <- tryCatch(
    {
      fit_gene_gam(
        y = y,
        sf = cells$s,
        r_um = r_um,
        theta = theta_model,
        AP_um = AP_um,
        ML_um = ML_um,
        brdu_pos = brdu_pos,
        edu_pos = edu_pos,
        batch = batch_model,
        animal = animal,
        use_theta = use_theta,
        shrinkage_basis = basis,
        k_uv = as.integer(k_uv),
        bam_nthreads = 1,
        gamma = GAM_GAMMA
      )
    },
    error = function(e) {
      conditionMessage(e)
    }
  )
  if (is.character(fit)) {
    return(list(j = j, gene = gene, write_row = write_row, row = na_row(gene), fit_path = fit_path, dt = proc.time()[[3]] - t0, err = fit, p = c(NA_real_, NA_real_, NA_real_), diag = NULL, diag_err = NULL))
  }

  pv <- extract_component_pvals(fit)
  es <- effect_sizes_from_fit(
    fit,
    r_um = r_um,
    AP_um = AP_um,
    ML_um = ML_um,
    has_pos = !is.null(brdu_pos) && !is.null(edu_pos),
    has_batch = !is.null(batch_model),
    batch_ref = batch_ref_model,
    has_animal = !is.null(animal),
    animal_ref = animal_ref,
    use_theta = use_theta
  )

  diag_written <- NULL
  diag_err <- NULL
  if (!is.null(diag_path) && isTRUE(diagnostics)) {
    p_diag <- c(pv$p_spatial, pv$p_cycle, pv$p_interaction)
    p_diag <- as.numeric(p_diag)
    p_diag <- p_diag[is.finite(p_diag)]
    is_hit <- (length(p_diag) > 0) && (min(p_diag) <= diagnostics_p)
    if (isTRUE(diagnostics_all) || isTRUE(is_hit)) {
      diag_try <- tryCatch(
        {
          run_diagnostics(fit, diag_path)
          NULL
        },
        error = function(e) {
          conditionMessage(e)
        }
      )
      if (is.null(diag_try)) {
        diag_written <- diag_path
      } else {
        diag_err <- diag_try
      }
    }
  }

  fit_save <- fit
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
    inm_r2 = coupling_r2,
    p_spatial = pv$p_spatial,
    p_cycle = pv$p_cycle,
    p_interaction = pv$p_interaction,
    p_apml = pv$p_apml,
    p_apml_r_um = pv$p_apml_r_um,
    p_brdu_pos = pv$p_brdu_pos,
    p_edu_pos = pv$p_edu_pos,
    p_brdu_edu = pv$p_brdu_edu,
    log_p_spatial = pv$log_p_spatial,
    log_p_cycle = pv$log_p_cycle,
    log_p_interaction = pv$log_p_interaction,
    log_p_apml = pv$log_p_apml,
    log_p_apml_r_um = pv$log_p_apml_r_um,
    log_p_brdu_pos = pv$log_p_brdu_pos,
    log_p_edu_pos = pv$log_p_edu_pos,
    log_p_brdu_edu = pv$log_p_brdu_edu,
    cycle_amp_link = es$cycle_amp_link,
    spatial_grad_link = es$spatial_grad_link,
    gating_index_link = es$gating_index_link,
    stringsAsFactors = FALSE
  )
  dt <- proc.time()[[3]] - t0
  list(
    j = j,
    gene = gene,
    write_row = write_row,
    row = row,
    fit_path = fit_path,
    dt = dt,
    err = NULL,
    p = c(row$p_spatial, row$p_cycle, row$p_interaction),
    diag = diag_written,
    diag_err = diag_err
  )
}

start_heartbeat <- function(label, heartbeat_sec) {
  if (.Platform$OS.type != "unix") return(NULL)
  sec <- as.integer(heartbeat_sec)
  if (!is.finite(sec) || is.na(sec) || sec <= 0) return(NULL)
  parallel::mcparallel({
    repeat {
      Sys.sleep(sec)
      cat(sprintf(
        "[%s] heartbeat: still running %s\n",
        format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
        label
      ))
      flush.console()
    }
  }, silent = FALSE)
}

stop_heartbeat <- function(job) {
  if (is.null(job)) return(invisible(NULL))
  pid <- job$pid
  if (!is.null(pid) && is.numeric(pid) && is.finite(pid) && pid > 0) {
    try(tools::pskill(pid), silent = TRUE)
  }
  try(parallel::mccollect(job, wait = FALSE), silent = TRUE)
  invisible(NULL)
}

chunk_size <- max(1L, as.integer(threads))
for (i0 in seq(1L, length(tasks), by = chunk_size)) {
  i1 <- min(length(tasks), i0 + chunk_size - 1L)
  chunk <- tasks[i0:i1]
  cat(sprintf("Chunk %d-%d / %d\n", i0, i1, length(tasks)))
  chunk_genes <- vapply(chunk, function(t) as.character(t$gene), character(1))
  cat(sprintf("  genes: %s\n", paste(chunk_genes, collapse = ",")))
  flush.console()

  hb <- start_heartbeat(sprintf("chunk %d-%d / %d", i0, i1, length(tasks)), heartbeat_sec)
  results <- tryCatch(
    {
      if (threads <= 1) {
        lapply(chunk, fit_one)
      } else {
        parallel::mclapply(chunk, fit_one, mc.cores = min(threads, length(chunk)))
      }
    },
    finally = {
      stop_heartbeat(hb)
    }
  )

  for (res in results) {
    if (!is.null(res$err)) {
      cat(sprintf("[%d/%d] ERROR %s: %s\n", res$j, length(genes), res$gene, res$err))
      write_fit_error(res$gene, res$err)
    }
    if (!is.null(res$diag_err)) {
      cat(sprintf("[%d/%d] DIAG_ERROR %s: %s\n", res$j, length(genes), res$gene, res$diag_err))
      write_fit_error(res$gene, sprintf("diagnostics: %s", res$diag_err))
    }
    if (isTRUE(res$write_row)) {
      write_fit_row(res$row)
      done <- c(done, res$gene)
    }
    if (is.finite(res$dt)) {
      cat(sprintf(
        "[%d/%d] done %s (%.2fs): p_spatial=%.3g p_cycle=%.3g p_interaction=%.3g\n",
        res$j, length(genes), res$gene, res$dt, res$p[[1]], res$p[[2]], res$p[[3]]
      ))
    } else {
      cat(sprintf("[%d/%d] done %s\n", res$j, length(genes), res$gene))
    }
    cat(sprintf("  saved %s\n", res$fit_path))
    if (!is.null(res$diag)) cat(sprintf("  diagnostics %s\n", res$diag))
    flush.console()
  }
}

cat(sprintf("Wrote fit results to %s\n", out_tsv))
write_qbh(out_tsv)

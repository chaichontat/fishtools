#!/usr/bin/env Rscript

# Fit a simplex topic model using ILR (isometric log-ratio) coordinates.
# This complements fit_inm_simplex_panel.R (ALR). We fit K-1 Gaussian GAMs on
# ILR coords and reconstruct simplex loadings via inverse-ILR (softmax of CLR).

read_cells <- function(in_dir) {
  cells <- read.delim(file.path(in_dir, "cells.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
  if (!("cell_id" %in% names(cells))) stop("cells.tsv must contain cell_id.")
  required <- c("r_um", "AP_um", "ML_um", "batch")
  if (!all(required %in% names(cells))) {
    stop(sprintf("cells.tsv must contain: %s", paste(required, collapse = ", ")))
  }
  cells$cell_id <- as.character(cells$cell_id)
  cells
}

read_usage <- function(path) {
  df <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (ncol(df) < 2) stop("usage.tsv must have first column cell_id and >=1 Usage_* columns.")
  id_col <- names(df)[[1]]
  df[[id_col]] <- as.character(df[[id_col]])
  rownames(df) <- df[[id_col]]
  df[[id_col]] <- NULL
  usage_cols <- grep("^Usage_", names(df), value = TRUE)
  if (length(usage_cols) <= 1) stop("usage.tsv must contain at least 2 Usage_* columns.")
  u <- as.matrix(df[, usage_cols, drop = FALSE])
  storage.mode(u) <- "numeric"
  if (any(!is.finite(u))) stop("usage.tsv contains non-finite values.")
  if (any(u < 0) || any(u > 1)) stop("usage.tsv values must be in [0,1].")
  list(u = u, cols = usage_cols, cell_ids = rownames(df))
}

clamp01 <- function(x) {
  x[x < 0] <- 0
  x[x > 1] <- 1
  x
}

usage_eps_renorm <- function(u, eps) {
  if (!is.matrix(u)) stop("u must be a matrix.")
  if (!is.numeric(eps) || length(eps) != 1 || !is.finite(eps) || eps <= 0) stop("eps must be finite and > 0.")
  x <- pmax(u, eps)
  rs <- rowSums(x)
  if (any(!is.finite(rs)) || any(rs <= 0)) stop("Invalid row sums after eps clamp.")
  x / rs
}

# Pivot ILR basis (K x (K-1)), orthonormal in clr space.
ilr_basis_pivot <- function(k) {
  if (!is.numeric(k) || length(k) != 1 || !is.finite(k) || k < 2) stop("k must be >= 2")
  k <- as.integer(k)
  V <- matrix(0.0, nrow = k, ncol = k - 1)
  for (j in seq_len(k - 1)) {
    V[seq_len(j), j] <- sqrt(1.0 / (j * (j + 1)))
    V[j + 1, j] <- -sqrt(j / (j + 1))
  }
  V
}

fit_ilr_gam <- function(
  z,
  r_um,
  theta,
  AP_um,
  ML_um,
  batch,
  animal,
  use_theta = TRUE,
  shrinkage_basis = c("shrink", "standard"),
  k_r = 5,
  k_theta = 8,
  k_uv = 15,
  k_uvr = 10,
  k_r_uvr = 5,
  k_rtheta = c(5, 8),
  gamma = 2.5,
  method = "REML",
  bam_discrete = TRUE,
  bam_nthreads = 1
) {
  suppressPackageStartupMessages({
    library(mgcv)
  })
  if (any(!is.finite(z))) stop("z must be finite.")
  if (!is.numeric(r_um) || any(!is.finite(r_um))) stop("r_um must be numeric and finite.")
  if (!is.numeric(AP_um) || !is.numeric(ML_um) || any(!is.finite(AP_um)) || any(!is.finite(ML_um))) {
    stop("AP_um/ML_um must be numeric and finite.")
  }
  if (!is.logical(use_theta) || length(use_theta) != 1) stop("use_theta must be TRUE/FALSE.")
  shrinkage_basis <- match.arg(shrinkage_basis)
  bs_uv <- if (shrinkage_basis == "shrink") "ts" else "tp"
  bs_r <- if (shrinkage_basis == "shrink") "cs" else "cr"

  df <- data.frame(z = as.numeric(z), r_um = as.numeric(r_um), AP_um = as.numeric(AP_um), ML_um = as.numeric(ML_um))

  has_theta <- isTRUE(use_theta)
  if (has_theta) {
    if (!is.numeric(theta) || any(!is.finite(theta))) stop("theta must be numeric and finite when use_theta=TRUE.")
    theta <- theta %% (2 * pi)
    theta[theta == 2 * pi] <- 0
    df$theta <- theta
  } else {
    df$theta <- 0.0
  }

  if (length(batch) != nrow(df)) stop("batch length mismatch.")
  df$batch <- as.factor(batch)

  has_animal <- !is.null(animal)
  if (has_animal) {
    if (length(animal) != nrow(df)) stop("animal length mismatch.")
    df$animal <- as.factor(animal)
  }

  knots <- if (has_theta) list(theta = c(0, 2 * pi)) else list()

  base_terms <- "1"
  if (has_animal) base_terms <- paste(base_terms, "+ s(animal, bs = 're')")
  if (!is.null(batch)) {
    if (has_animal) {
      df$ab <- interaction(df$animal, df$batch, drop = TRUE)
      base_terms <- paste(base_terms, "+ s(ab, bs = 're')")
    } else {
      base_terms <- paste(base_terms, "+ batch")
    }
  }
  smooth_terms <- paste0(
    if (!has_theta) "" else " + s(theta, bs = 'cc', k = k_theta)",
    sprintf(" + s(AP_um, ML_um, bs = '%s', k = k_uv)", bs_uv),
    sprintf(" + s(r_um, bs = '%s', k = k_r)", bs_r),
    sprintf(" + ti(AP_um, ML_um, r_um, d = c(2, 1), bs = c('%s', '%s'), k = c(k_uvr, k_r_uvr))", bs_uv, bs_r),
    if (!has_theta) "" else sprintf(" + ti(r_um, theta, bs = c('%s', 'cc'), k = k_rtheta)", bs_r)
  )

  ap_unique <- length(unique(df$AP_um))
  ml_unique <- length(unique(df$ML_um))
  if (ap_unique <= 1 || ml_unique <= 1) {
    smooth_terms <- paste0(
      if (!has_theta) "" else " + s(theta, bs = 'cc', k = k_theta)",
      sprintf(" + s(r_um, bs = '%s', k = k_r)", bs_r),
      if (!has_theta) "" else sprintf(" + ti(r_um, theta, bs = c('%s', 'cc'), k = k_rtheta)", bs_r)
    )
  }

  rhs <- paste0(base_terms, smooth_terms)
  formula <- as.formula(paste("z ~", rhs))
  if (!is.numeric(bam_nthreads) || length(bam_nthreads) != 1 || !is.finite(bam_nthreads) || bam_nthreads < 1) {
    stop("bam_nthreads must be a single finite integer >= 1.")
  }
  method_bam <- if (identical(method, "REML")) "fREML" else method
  bam(
    formula,
    data = df,
    family = gaussian(),
    method = method_bam,
    knots = knots,
    select = TRUE,
    gamma = gamma,
    discrete = isTRUE(bam_discrete),
    nthreads = as.integer(bam_nthreads)
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript scripts/gam/fit_inm_simplex_panel_ilr.R IN_DIR --usage-tsv PATH [--out-tsv PATH] [--eps X] [--r-max X] [--no-theta] [--threads N] [--bam-threads N] [--basis shrink|standard] [--k-uv N] [--k-r N] [--k-r-uvr N]\n")
  quit(status = 2)
}

in_dir <- args[[1]]
opts <- if (length(args) >= 2) args[2:length(args)] else character(0)

parse_opts <- function(opts) {
  usage_tsv <- NULL
  out_tsv <- file.path(in_dir, "fit_results.simplex_ilr.tsv")
  eps <- 1e-4
  r_max <- Inf
  use_theta <- TRUE
  threads <- 1L
  bam_threads <- 1L
  basis <- "standard"
  k_uv <- 15L
  k_r <- 5L
  k_r_uvr <- 5L
  i <- 1L
  while (i <= length(opts)) {
    opt <- opts[[i]]
    if (opt == "--usage-tsv") {
      if (i == length(opts)) stop("--usage-tsv requires a path.")
      usage_tsv <- as.character(opts[[i + 1L]])
      i <- i + 2L
      next
    }
    if (opt == "--out-tsv") {
      if (i == length(opts)) stop("--out-tsv requires a path.")
      out_tsv <- as.character(opts[[i + 1L]])
      i <- i + 2L
      next
    }
    if (opt == "--eps") {
      if (i == length(opts)) stop("--eps requires a numeric value.")
      val <- suppressWarnings(as.numeric(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val <= 0) stop("--eps must be finite and > 0.")
      eps <- val
      i <- i + 2L
      next
    }
    if (opt == "--r-max") {
      if (i == length(opts)) stop("--r-max requires a numeric value.")
      val <- suppressWarnings(as.numeric(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val <= 0) stop("--r-max must be finite and > 0.")
      r_max <- val
      i <- i + 2L
      next
    }
    if (opt == "--no-theta") {
      use_theta <- FALSE
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
    if (opt == "--bam-threads") {
      if (i == length(opts)) stop("--bam-threads requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 1) stop("--bam-threads must be an integer >= 1.")
      bam_threads <- val
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
    if (opt == "--k-r") {
      if (i == length(opts)) stop("--k-r requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 4) stop("--k-r must be an integer >= 4.")
      k_r <- val
      i <- i + 2L
      next
    }
    if (opt == "--k-r-uvr") {
      if (i == length(opts)) stop("--k-r-uvr requires an integer value.")
      val <- suppressWarnings(as.integer(opts[[i + 1L]]))
      if (!is.finite(val) || is.na(val) || val < 3) stop("--k-r-uvr must be an integer >= 3.")
      k_r_uvr <- val
      i <- i + 2L
      next
    }
    stop(sprintf("Unknown option: %s", opt))
  }
  list(
    usage_tsv = usage_tsv,
    out_tsv = out_tsv,
    eps = eps,
    r_max = r_max,
    use_theta = use_theta,
    threads = threads,
    bam_threads = bam_threads,
    basis = basis,
    k_uv = k_uv,
    k_r = k_r,
    k_r_uvr = k_r_uvr
  )
}

parsed <- parse_opts(opts)
usage_tsv <- parsed$usage_tsv
out_tsv <- parsed$out_tsv
eps <- parsed$eps
r_max <- parsed$r_max
use_theta <- parsed$use_theta
threads <- parsed$threads
bam_threads <- parsed$bam_threads
basis <- parsed$basis
k_uv <- parsed$k_uv
k_r <- parsed$k_r
k_r_uvr <- parsed$k_r_uvr

if (is.null(usage_tsv) || !nzchar(usage_tsv)) stop("--usage-tsv is required.")
if (!file.exists(usage_tsv)) stop(sprintf("usage.tsv not found: %s", usage_tsv))

source("scripts/gam/inm_gam.R")

cells <- read_cells(in_dir)
u0 <- read_usage(usage_tsv)
u <- u0$u
usage_cols <- u0$cols

idx <- match(cells$cell_id, u0$cell_ids)
if (any(is.na(idx))) {
  missing <- cells$cell_id[is.na(idx)]
  stop(sprintf("Missing usage rows for %d cells; e.g. %s", length(missing), paste(head(missing, 5), collapse = ",")))
}
u <- u[idx, , drop = FALSE]

r_um0 <- as.numeric(cells$r_um)
AP_um0 <- as.numeric(cells$AP_um)
ML_um0 <- as.numeric(cells$ML_um)
if (any(!is.finite(r_um0)) || any(!is.finite(AP_um0)) || any(!is.finite(ML_um0))) stop("cells r_um/AP_um/ML_um must be finite.")

keep <- rep(TRUE, nrow(cells))
if (is.finite(r_max)) {
  keep <- r_um0 <= r_max
  cat(sprintf("Filtering: r_um <= %.6g kept %d/%d cells\n", r_max, sum(keep), length(keep)))
}
if (!any(keep)) stop("r-max filter removed all cells; nothing to fit.")

cells <- cells[keep, , drop = FALSE]
u <- u[keep, , drop = FALSE]

u <- clamp01(u)
u_eps <- usage_eps_renorm(u, eps = eps)

topic_ids <- vapply(usage_cols, function(nm) suppressWarnings(as.integer(sub("^Usage_", "", nm))), integer(1))
if (any(!is.finite(topic_ids)) || any(is.na(topic_ids))) {
  stop("Usage_* columns must be named like Usage_1..Usage_K.")
}

r_um <- as.numeric(cells$r_um)
AP_um <- as.numeric(cells$AP_um)
ML_um <- as.numeric(cells$ML_um)
if (any(!is.finite(r_um)) || any(!is.finite(AP_um)) || any(!is.finite(ML_um))) stop("cells r_um/AP_um/ML_um must be finite.")

theta <- rep(0.0, nrow(cells))
if (isTRUE(use_theta)) {
  if (!("theta" %in% names(cells))) stop("cells.tsv missing theta column (required unless --no-theta).")
  theta <- as.numeric(cells$theta)
  if (any(!is.finite(theta))) stop("cells$theta must be finite.")
}

batch <- as.factor(cells$batch)
batch_ref <- levels(batch)[[1]]

extract_animal_from_dataset <- function(x) {
  s <- as.character(x)
  m <- regexpr("JaxA[0-9]+", s, perl = TRUE)
  out <- ifelse(m > 0, regmatches(s, m), NA_character_)
  out
}

animal_ref <- NULL
animal <- NULL
if ("animal" %in% names(cells)) {
  animal <- as.factor(cells$animal)
  if (any(is.na(animal))) stop("cells$animal must not contain NA.")
  animal_ref <- levels(animal)[[1]]
} else {
  a0 <- extract_animal_from_dataset(cells$batch)
  if (any(is.na(a0) | !nzchar(a0))) {
    bad <- unique(cells$batch[is.na(a0) | !nzchar(a0)])
    stop(sprintf("Could not parse animal (JaxA*) from cells$batch for dataset(s): %s.", paste(bad, collapse = ",")))
  }
  animal <- as.factor(a0)
  animal_ref <- levels(animal)[[1]]
}

# ILR coords: z = log(u) %*% V where V is orthonormal (pivot ILR basis).
k <- ncol(u_eps)
V <- ilr_basis_pivot(k)
z_mat <- log(u_eps) %*% V
if (any(!is.finite(z_mat))) stop("Non-finite ILR coordinates encountered; check eps and usage.")

coord_names <- vapply(seq_len(ncol(z_mat)), function(j) sprintf("ILR_C%d", j), character(1))

out_dir <- dirname(out_tsv)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
fits_dir <- file.path(out_dir, "fits_rds__fit_results_simplex_ilr")
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
fit_error_path <- if (grepl("\\.tsv$", out_tsv)) sub("\\.tsv$", ".errors.tsv", out_tsv) else paste0(out_tsv, ".errors.tsv")

write_fit_error <- function(name, msg) {
  err_row <- data.frame(component = name, error = msg, stringsAsFactors = FALSE)
  write_err_header <- !(file.exists(fit_error_path) && file.info(fit_error_path)$size > 0)
  if (write_err_header) {
    write.table(err_row, file = fit_error_path, sep = "\t", row.names = FALSE, quote = FALSE)
  } else {
    write.table(err_row, file = fit_error_path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  }
}

fit_one <- function(j) {
  comp_name <- coord_names[[j]]
  fit <- tryCatch(
    {
      fit_ilr_gam(
        z = z_mat[, j],
        r_um = r_um,
        theta = theta,
        AP_um = AP_um,
        ML_um = ML_um,
        batch = batch,
        animal = animal,
        use_theta = isTRUE(use_theta),
        shrinkage_basis = basis,
        k_r = k_r,
        k_uv = k_uv,
        k_r_uvr = k_r_uvr,
        bam_nthreads = bam_threads
      )
    },
    error = function(e) {
      write_fit_error(comp_name, conditionMessage(e))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)
  saveRDS(fit, file.path(fits_dir, paste0(comp_name, ".gam.rds")))
  pv <- extract_component_pvals(fit)
  data.frame(
    gene = comp_name,
    component = j,
    is_ref = FALSE,
    p_spatial = pv$p_spatial,
    p_cycle = pv$p_cycle,
    p_interaction = pv$p_interaction,
    p_apml = pv$p_apml,
    p_apml_r_um = pv$p_apml_r_um,
    log_p_spatial = pv$log_p_spatial,
    log_p_cycle = pv$log_p_cycle,
    log_p_interaction = pv$log_p_interaction,
    log_p_apml = pv$log_p_apml,
    log_p_apml_r_um = pv$log_p_apml_r_um,
    stringsAsFactors = FALSE
  )
}

results <- vector("list", length = ncol(z_mat))
if (threads > 1 && .Platform$OS.type != "unix") stop("--threads > 1 requires a Unix-like OS (forking).")
if (threads > 1) {
  chunk <- ceiling(ncol(z_mat) / threads)
  idxs <- split(seq_len(ncol(z_mat)), rep(seq_len(threads), each = chunk, length.out = ncol(z_mat)))
  out_list <- parallel::mclapply(
    idxs,
    function(js) lapply(js, fit_one),
    mc.cores = threads
  )
  results <- unlist(out_list, recursive = FALSE)
} else {
  for (j in seq_len(ncol(z_mat))) results[[j]] <- fit_one(j)
}

tab <- do.call(rbind, results[!vapply(results, is.null, logical(1))])
if (is.null(tab) || nrow(tab) <= 0) stop("No ILR components fit successfully.")

write.table(tab, file = out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote %s (n=%d rows)\n", out_tsv, nrow(tab)))

meta <- list(
  transform = "ilr",
  ilr_basis = "pivot",
  usage_tsv = usage_tsv,
  usage_cols = usage_cols,
  topic_ids = topic_ids,
  coord_names = coord_names,
  eps = eps,
  r_max = r_max,
  n_cells_fit = nrow(cells),
  use_theta = use_theta,
  basis = basis,
  k_uv = k_uv,
  k_r = k_r,
  k_r_uvr = k_r_uvr,
  batch_ref = batch_ref,
  animal_ref = animal_ref,
  fits_dir = fits_dir
)
meta_path <- file.path(out_dir, "simplex_meta_ilr.json")
writeLines(jsonlite::toJSON(meta, pretty = TRUE, auto_unbox = TRUE), con = meta_path)
cat(sprintf("Wrote %s\n", meta_path))

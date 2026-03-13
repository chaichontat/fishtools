#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(mgcv)
})

wrap_theta <- function(theta) {
  if (!is.numeric(theta)) stop("theta must be numeric (radians).")
  if (any(!is.finite(theta))) stop("theta must be finite (no NA/Inf).")
  theta <- theta %% (2 * pi)
  theta[theta == 2 * pi] <- 0
  theta
}

fit_inm_coupling_gam <- function(
  x,
  theta,
  k_theta = 8,
  method = "REML",
  group = NULL
) {
  if (!is.numeric(x)) stop("x must be numeric.")
  if (any(!is.finite(x))) stop("x must be finite (no NA/Inf).")
  if (length(x) != length(theta)) stop("x and theta must have same length.")
  if (!is.numeric(k_theta) || length(k_theta) != 1 || !is.finite(k_theta) || k_theta < 3) {
    stop("k_theta must be a single finite number >= 3.")
  }

  theta <- wrap_theta(theta)
  knots <- list(theta = c(0, 2 * pi))

  r2_from_fit <- function(xv, m_hat) {
    sst <- sum((xv - mean(xv))^2)
    if (!is.finite(sst) || sst <= 0) return(NA_real_)
    sse <- sum((xv - m_hat)^2)
    1 - (sse / sst)
  }

  if (is.null(group)) {
    df <- data.frame(x = x, theta = theta)
    fit <- gam(x ~ s(theta, bs = "cc", k = k_theta), data = df, method = method, knots = knots)
    m_hat <- as.numeric(predict(fit, newdata = df, type = "response"))
    r_hat <- x - m_hat
    list(
      fit = fit,
      m_hat = m_hat,
      r_hat = r_hat,
      r2 = r2_from_fit(x, m_hat)
    )
  } else {
    if (length(group) != length(x)) stop("group must have same length as x.")
    group <- as.factor(group)
    fits <- vector("list", length = nlevels(group))
    names(fits) <- levels(group)

    min_n <- max(10, as.integer(k_theta) + 1)
    ok_group <- function(idx) {
      idx <- as.integer(idx)
      if (length(idx) < min_n) return(FALSE)
      th <- theta[idx]
      th <- th[is.finite(th)]
      if (length(th) < min_n) return(FALSE)
      length(unique(th)) >= 3
    }
    ok <- vapply(levels(group), function(g) ok_group(which(group == g)), logical(1))
    if (!all(ok)) {
      bad <- paste(names(ok)[!ok], collapse = ",")
      warning(sprintf(
        "Not enough theta support to fit coupling per group (k_theta=%d, min_n=%d). Falling back to pooled coupling. Bad groups: %s",
        k_theta, min_n, bad
      ))
      return(fit_inm_coupling_gam(x = x, theta = theta, k_theta = k_theta, method = method, group = NULL))
    }

    m_hat <- rep(NA_real_, length(x))
    r2_by_group <- rep(NA_real_, length(fits))
    names(r2_by_group) <- names(fits)

    for (g in levels(group)) {
      idx <- which(group == g)
      df_g <- data.frame(x = x[idx], theta = theta[idx])
      fit_g <- gam(x ~ s(theta, bs = "cc", k = k_theta), data = df_g, method = method, knots = knots)
      m_hat_g <- as.numeric(predict(fit_g, newdata = df_g, type = "response"))
      m_hat[idx] <- m_hat_g
      fits[[g]] <- fit_g
      r2_by_group[[g]] <- r2_from_fit(df_g$x, m_hat_g)
    }

    r_hat <- x - m_hat
    list(
      fits = fits,
      m_hat = m_hat,
      r_hat = r_hat,
      r2 = r2_from_fit(x, m_hat),
      r2_by_group = r2_by_group
    )
  }
}

fit_gene_gam <- function(
  y,
  sf,
  r_um,
  theta,
  AP_um,
  ML_um,
  usage = NULL,
  brdu_pos = NULL,
  edu_pos = NULL,
  batch = NULL,
  animal = NULL,
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
  engine = c("bam", "auto", "gam"),
  bam_discrete = TRUE,
  bam_nthreads = 1
) {
  if (length(y) != length(sf) || length(y) != length(r_um) || length(y) != length(theta) || length(y) != length(AP_um) || length(y) != length(ML_um)) {
    stop("y, sf, r_um, theta, AP_um, ML_um must have the same length.")
  }
  if (!is.logical(use_theta) || length(use_theta) != 1) stop("use_theta must be TRUE/FALSE.")
  if (!is.numeric(y)) stop("y must be numeric.")
  if (any(!is.finite(y))) stop("y must be finite (no NA/Inf).")
  if (any(y < 0)) stop("y must be non-negative (counts).")
  if (!is.numeric(sf)) stop("sf must be numeric.")
  if (any(!is.finite(sf)) || any(sf <= 0)) stop("sf must be finite and > 0.")
  # Size factor is treated as a fixed offset; we do not estimate an additional log(sf) slope.
  if (!is.numeric(r_um)) stop("r_um must be numeric.")
  if (any(!is.finite(r_um))) stop("r_um must be finite (no NA/Inf).")
  theta <- wrap_theta(theta)
  if (!is.numeric(AP_um) || !is.numeric(ML_um)) stop("AP_um/ML_um must be numeric.")
  if (any(!is.finite(AP_um)) || any(!is.finite(ML_um))) stop("AP_um/ML_um must be finite (no NA/Inf).")

  shrinkage_basis <- match.arg(shrinkage_basis)
  bs_uv <- if (shrinkage_basis == "shrink") "ts" else "tp"
  bs_r <- if (shrinkage_basis == "shrink") "cs" else "cr"

  df <- data.frame(y = y, sf = sf, r_um = r_um, theta = theta, AP_um = AP_um, ML_um = ML_um)
  usage_terms <- ""
  if (!is.null(usage)) {
    if (nrow(usage) != length(y)) stop("usage must have nrow == length(y).")
    if (is.matrix(usage)) {
      usage <- as.data.frame(usage, stringsAsFactors = FALSE)
    }
    if (!is.data.frame(usage)) stop("usage must be a data.frame or matrix.")
    if (ncol(usage) < 1) stop("usage must have at least one column.")
    if (is.null(colnames(usage)) || any(!nzchar(colnames(usage)))) {
      stop("usage must have non-empty column names.")
    }
    for (nm in colnames(usage)) {
      v <- suppressWarnings(as.numeric(usage[[nm]]))
      if (any(!is.finite(v))) stop(sprintf("usage column %s must be finite.", nm))
      df[[nm]] <- v
    }
    usage_terms <- paste(colnames(usage), collapse = " + ")
  }
  has_pos <- FALSE
  if (!is.null(brdu_pos) || !is.null(edu_pos)) {
    if (is.null(brdu_pos) || is.null(edu_pos)) stop("Provide both brdu_pos and edu_pos, or neither.")
    if (length(brdu_pos) != length(y) || length(edu_pos) != length(y)) stop("brdu_pos/edu_pos length mismatch.")
    df$brdu_pos <- as.numeric(brdu_pos)
    df$edu_pos <- as.numeric(edu_pos)
    has_pos <- TRUE
  }

  has_batch <- FALSE
  if (!is.null(batch)) {
    if (length(batch) != length(y)) stop("batch length mismatch.")
    df$batch <- as.factor(batch)
    has_batch <- TRUE
  }

  has_animal <- FALSE
  if (!is.null(animal)) {
    if (length(animal) != length(y)) stop("animal length mismatch.")
    df$animal <- as.factor(animal)
    has_animal <- TRUE
  }
  knots <- if (isTRUE(use_theta)) list(theta = c(0, 2 * pi)) else list()

  base_terms <- "offset(log(sf))"
  if (isTRUE(has_animal)) base_terms <- paste(base_terms, "+ s(animal, bs = 're')")
  if (isTRUE(has_batch)) {
    if (isTRUE(has_animal)) {
      df$ab <- interaction(df$animal, df$batch, drop = TRUE)
      base_terms <- paste(base_terms, "+ s(ab, bs = 're')")
    } else {
      base_terms <- paste(base_terms, "+ batch")
    }
  }
  smooth_terms <- paste0(
    if (!isTRUE(use_theta)) "" else " + s(theta, bs = 'cc', k = k_theta)",
    sprintf(" + s(AP_um, ML_um, bs = '%s', k = k_uv)", bs_uv),
    sprintf(" + s(r_um, bs = '%s', k = k_r)", bs_r),
    sprintf(" + ti(AP_um, ML_um, r_um, d = c(2, 1), bs = c('%s', '%s'), k = c(k_uvr, k_r_uvr))", bs_uv, bs_r),
    if (!isTRUE(use_theta)) "" else sprintf(" + ti(r_um, theta, bs = c('%s', 'cc'), k = k_rtheta)", bs_r)
  )
  pos_terms <- if (!has_pos) "" else " + brdu_pos + edu_pos + brdu_pos:edu_pos"
  rhs <- paste0(base_terms, pos_terms, smooth_terms, if (usage_terms == "") "" else paste0(" + ", usage_terms))
  formula <- as.formula(paste("y ~", rhs))
  engine <- match.arg(engine)
  if (engine == "auto") {
    engine <- "bam"
  }
  if (engine == "bam") {
    method_bam <- if (identical(method, "REML")) "fREML" else method
    if (!is.numeric(bam_nthreads) || length(bam_nthreads) != 1 || !is.finite(bam_nthreads) || bam_nthreads < 1) {
      stop("bam_nthreads must be a single finite integer >= 1.")
    }
    fit <- bam(
      formula,
      data = df,
      family = nb(),
      method = method_bam,
      knots = knots,
      select = TRUE,
      gamma = gamma,
      discrete = isTRUE(bam_discrete),
      nthreads = as.integer(bam_nthreads)
    )
    fit
  } else {
    fit <- gam(formula, data = df, family = nb(), method = method, knots = knots, select = TRUE, gamma = gamma)
    fit
  }
}

effect_sizes_from_fit <- function(
  fit,
  r_um,
  AP_um,
  ML_um,
  has_pos = FALSE,
  has_batch = FALSE,
  batch_ref = NULL,
  has_animal = FALSE,
  animal_ref = NULL,
  usage_ref = NULL,
  use_theta = TRUE,
  sf_ref = 1.0,
  n_theta = 64
) {
  r_q <- as.numeric(quantile(r_um, probs = c(0.1, 0.9), names = FALSE))
  r_lo <- r_q[[1]]
  r_hi <- r_q[[2]]
  r_mid <- as.numeric(median(r_um))

  theta_grid <- seq(0, 2 * pi, length.out = n_theta + 1)
  theta_grid <- theta_grid[-length(theta_grid)]

  ap_ref <- as.numeric(median(AP_um))
  ml_ref <- as.numeric(median(ML_um))

  get_factor_levels <- function(fit, name) {
    if ("xlevels" %in% names(fit) && (name %in% names(fit$xlevels))) {
      return(as.character(fit$xlevels[[name]]))
    }
    if ("var.summary" %in% names(fit) && (name %in% names(fit$var.summary))) {
      v <- fit$var.summary[[name]]
      if (is.factor(v)) return(levels(v))
    }
    NULL
  }

  add_pos <- function(nd) {
    if (isTRUE(has_pos)) {
      nd$brdu_pos <- 0
      nd$edu_pos <- 0
    }
    if (isTRUE(has_batch) && !is.null(batch_ref)) {
      lev <- get_factor_levels(fit, "batch")
      nd$batch <- if (is.null(lev)) batch_ref else factor(rep(batch_ref, nrow(nd)), levels = lev)
    }
    if (isTRUE(has_animal) && !is.null(animal_ref)) {
      lev <- get_factor_levels(fit, "animal")
      nd$animal <- if (is.null(lev)) animal_ref else factor(rep(animal_ref, nrow(nd)), levels = lev)
    }
    if (isTRUE(has_batch) && isTRUE(has_animal) && !is.null(batch_ref) && !is.null(animal_ref)) {
      lev <- get_factor_levels(fit, "ab")
      ab0 <- interaction(nd$animal, nd$batch, drop = TRUE)
      nd$ab <- if (is.null(lev)) ab0 else factor(as.character(ab0), levels = lev)
    }
    if (!is.null(usage_ref)) {
      if (is.null(names(usage_ref)) || any(!nzchar(names(usage_ref)))) {
        stop("usage_ref must be a named numeric vector.")
      }
      for (nm in names(usage_ref)) {
        nd[[nm]] <- rep(as.numeric(usage_ref[[nm]]), nrow(nd))
      }
    }
    nd
  }

  if (!is.logical(use_theta) || length(use_theta) != 1) stop("use_theta must be TRUE/FALSE.")

  shrink_soft <- function(fit, se) {
    fit <- as.matrix(fit)
    se <- as.matrix(se)
    fit / (1.0 + se)
  }

  term_cols_compact <- function(cols) {
    gsub("\\s+", "", as.character(cols))
  }

  select_theta_dependent_cols <- function(cols) {
    cc <- term_cols_compact(cols)
    is_s_theta <- cc == "s(theta)"
    is_ti_r_theta <- startsWith(cc, "ti(") & grepl("r_um", cc, fixed = TRUE) & grepl("theta", cc, fixed = TRUE)
    which(is_s_theta | is_ti_r_theta)
  }

  select_r_dependent_cols <- function(cols) {
    cc <- term_cols_compact(cols)
    is_s_r <- cc == "s(r_um)"
    is_ti_r_theta <- startsWith(cc, "ti(") & grepl("r_um", cc, fixed = TRUE) & grepl("theta", cc, fixed = TRUE)
    is_ti_apml_r <- startsWith(cc, "ti(") &
      grepl("r_um", cc, fixed = TRUE) &
      grepl("AP_um", cc, fixed = TRUE) &
      grepl("ML_um", cc, fixed = TRUE) &
      !grepl("theta", cc, fixed = TRUE)
    which(is_s_r | is_ti_r_theta | is_ti_apml_r)
  }

  shrunk_effect_from_cols <- function(pred_terms, cols_idx) {
    if (length(cols_idx) == 0) return(rep(0.0, nrow(pred_terms$fit)))
    tfit <- as.matrix(pred_terms$fit)[, cols_idx, drop = FALSE]
    tse <- as.matrix(pred_terms$se.fit)[, cols_idx, drop = FALSE]
    rowSums(shrink_soft(tfit, tse))
  }

  amp_at_r <- function(r0) {
    if (!isTRUE(use_theta)) return(NA_real_)
    nd <- data.frame(
      r_um = rep(r0, length(theta_grid)),
      theta = theta_grid,
      AP_um = ap_ref,
      ML_um = ml_ref,
      sf = sf_ref
    )
    nd <- add_pos(nd)
    pred_terms <- predict(fit, newdata = nd, type = "terms", se.fit = TRUE)
    cols_idx <- select_theta_dependent_cols(colnames(pred_terms$fit))
    eff <- shrunk_effect_from_cols(pred_terms, cols_idx)
    if (any(!is.finite(eff))) return(NA_real_)
    max(eff) - min(eff)
  }

  cycle_amp_mid <- amp_at_r(r_mid)
  cycle_amp_lo <- amp_at_r(r_lo)
  cycle_amp_hi <- amp_at_r(r_hi)
  gating_index <- if (!isTRUE(use_theta)) NA_real_ else (cycle_amp_hi - cycle_amp_lo)

  nd_grad <- data.frame(
    r_um = c(r_lo, r_hi),
    theta = c(0, 0),
    AP_um = ap_ref,
    ML_um = ml_ref,
    sf = sf_ref
  )
  nd_grad <- add_pos(nd_grad)
  pred_terms_grad <- predict(fit, newdata = nd_grad, type = "terms", se.fit = TRUE)
  cols_idx_grad <- select_r_dependent_cols(colnames(pred_terms_grad$fit))
  eff_grad <- shrunk_effect_from_cols(pred_terms_grad, cols_idx_grad)
  spatial_grad <- if (length(eff_grad) == 2 && all(is.finite(eff_grad))) eff_grad[[2]] - eff_grad[[1]] else NA_real_

  list(
    cycle_amp_link = cycle_amp_mid,
    spatial_grad_link = spatial_grad,
    gating_index_link = gating_index
  )
}

extract_component_pvals <- function(fit, extra_params = character(0)) {
  log_p_min <- log(.Machine$double.xmin)
  empty_pvals <- list(
    p_spatial = NA_real_, p_cycle = NA_real_, p_interaction = NA_real_,
    p_apml = NA_real_, p_apml_r_um = NA_real_,
    p_brdu_pos = NA_real_, p_edu_pos = NA_real_, p_brdu_edu = NA_real_,
    log_p_spatial = NA_real_, log_p_cycle = NA_real_, log_p_interaction = NA_real_,
    log_p_apml = NA_real_, log_p_apml_r_um = NA_real_,
    log_p_brdu_pos = NA_real_, log_p_edu_pos = NA_real_, log_p_brdu_edu = NA_real_
  )
  sum_fit <- summary(fit)
  st <- sum_fit$s.table
  pt <- sum_fit$p.table
  if (is.null(st)) {
    out <- empty_pvals
    extra_params <- unique(as.character(extra_params))
    extra_params <- extra_params[nzchar(extra_params)]
    if (length(extra_params) > 0) {
      for (nm in extra_params) {
        out[[paste0("beta_", nm)]] <- NA_real_
        out[[paste0("p_", nm)]] <- NA_real_
        out[[paste0("log_p_", nm)]] <- NA_real_
      }
    }
    return(out)
  }

  rn <- rownames(st)
  rn_compact <- gsub("\\s+", "", rn)

  p_col <- if ("p-value" %in% colnames(st)) "p-value" else NA_character_
  if (is.na(p_col)) return(empty_pvals)

  clamp_logp <- function(lp) {
    if (!is.finite(lp)) {
      if (is.infinite(lp) && lp < 0) return(log_p_min)
      return(NA_real_)
    }
    if (lp > 0) return(0.0)
    lp
  }

  clamp_p <- function(p) {
    if (!is.finite(p)) return(NA_real_)
    if (p <= 0) return(.Machine$double.xmin)
    if (p > 1) return(1.0)
    p
  }

  p_from_logp <- function(lp) {
    lp <- clamp_logp(lp)
    if (!is.finite(lp)) return(NA_real_)
    if (lp < log_p_min) return(.Machine$double.xmin)
    exp(lp)
  }

  smooth_logp <- function(idx) {
    idx <- as.integer(idx)
    if (!is.finite(idx) || idx < 1 || idx > nrow(st)) return(NA_real_)
    if ("F" %in% colnames(st)) {
      fval <- as.numeric(st[idx, "F"])
      df1 <- as.numeric(st[idx, "Ref.df"])
      df2 <- as.numeric(fit$df.residual)
      if (!is.finite(fval) || !is.finite(df1) || df1 <= 0 || !is.finite(df2) || df2 <= 0) return(NA_real_)
      clamp_logp(pf(fval, df1 = df1, df2 = df2, lower.tail = FALSE, log.p = TRUE))
    } else if ("Chi.sq" %in% colnames(st)) {
      chi <- as.numeric(st[idx, "Chi.sq"])
      df_col <- if ("Ref.df" %in% colnames(st)) "Ref.df" else if ("edf" %in% colnames(st)) "edf" else NA_character_
      if (is.na(df_col)) return(NA_real_)
      dfv <- as.numeric(st[idx, df_col])
      if (!is.finite(chi) || !is.finite(dfv) || dfv <= 0) return(NA_real_)
      clamp_logp(pchisq(chi, df = dfv, lower.tail = FALSE, log.p = TRUE))
    } else {
      NA_real_
    }
  }

  pick_smooth <- function(regex, label) {
    idx <- which(grepl(regex, rn_compact, perl = TRUE))
    if (length(idx) == 0) return(list(p = NA_real_, log_p = NA_real_))
    if (length(idx) > 1) warning(sprintf("Multiple smooth terms matched %s; using first.", label))
    idx <- idx[[1]]
    p0 <- as.numeric(st[idx, p_col])
    if (is.finite(p0) && p0 > 0) {
      lp <- clamp_logp(log(p0))
      return(list(p = clamp_p(p0), log_p = lp))
    }
    lp <- smooth_logp(idx)
    list(p = p_from_logp(lp), log_p = lp)
  }

  pick_param <- function(name) {
    # Prefer summary table, but fall back to a Wald p-value from coef+Vp
    # (some stripped/edge fits can produce NaN p-values in summary()).
    p0 <- NA_real_
    if (!is.null(pt) && (name %in% rownames(pt))) {
      # `summary.gam()` / `summary.bam()` may report either z- or t-based p-values.
      col <- if ("Pr(>|z|)" %in% colnames(pt)) {
        "Pr(>|z|)"
      } else if ("Pr(>|t|)" %in% colnames(pt)) {
        "Pr(>|t|)"
      } else {
        NA_character_
      }
      if (!is.na(col)) {
        p0 <- as.numeric(pt[name, col])
      }
    }
    if (is.finite(p0) && p0 > 0) return(list(p = clamp_p(p0), log_p = clamp_logp(log(p0))))

    b <- fit$coefficients
    if (is.null(b) || !(name %in% names(b))) return(list(p = NA_real_, log_p = NA_real_))
    V <- fit$Vp
    if (is.null(V) || !is.matrix(V) || nrow(V) != length(b) || ncol(V) != length(b)) return(list(p = NA_real_, log_p = NA_real_))
    se <- sqrt(diag(V))
    if (any(!is.finite(se))) return(list(p = NA_real_, log_p = NA_real_))
    names(se) <- names(b)
    if (!(name %in% names(se))) return(list(p = NA_real_, log_p = NA_real_))
    tval <- as.numeric(b[[name]] / se[[name]])
    if (!is.finite(tval)) return(list(p = NA_real_, log_p = NA_real_))
    df <- fit$df.residual
    log_tail <- if (is.null(df) || !is.finite(df) || df <= 0) {
      pnorm(abs(tval), lower.tail = FALSE, log.p = TRUE)
    } else {
      pt(abs(tval), df = df, lower.tail = FALSE, log.p = TRUE)
    }
    log_p <- clamp_logp(log(2) + log_tail)
    list(p = p_from_logp(log_p), log_p = log_p)
  }

  sp <- pick_smooth("^s\\(r_um\\)$", "s(r_um)")
  cy <- pick_smooth("^s\\(theta\\)$", "s(theta)")
  it <- pick_smooth("^ti\\(r_um,theta\\)$", "ti(r_um,theta)")
  ap <- pick_smooth("^s\\(AP_um,ML_um\\)$", "s(AP_um,ML_um)")
  ap3 <- pick_smooth("^ti\\((?=.*AP_um)(?=.*ML_um)(?=.*r_um).+\\)$", "ti(AP_um,ML_um,r_um)")
  br <- pick_param("brdu_pos")
  ed <- pick_param("edu_pos")
  be <- pick_param("brdu_pos:edu_pos")

  out <- list(
    p_spatial = sp$p,
    p_cycle = cy$p,
    p_interaction = it$p,
    p_apml = ap$p,
    # mgcv may reorder tensor interaction term labels (e.g. "ti(r_um,AP_um,ML_um)").
    # Use lookaheads so we match any ti() term containing all three variables, regardless of order.
    p_apml_r_um = ap3$p,
    p_brdu_pos = br$p,
    p_edu_pos = ed$p,
    p_brdu_edu = be$p,
    log_p_spatial = sp$log_p,
    log_p_cycle = cy$log_p,
    log_p_interaction = it$log_p,
    log_p_apml = ap$log_p,
    log_p_apml_r_um = ap3$log_p,
    log_p_brdu_pos = br$log_p,
    log_p_edu_pos = ed$log_p,
    log_p_brdu_edu = be$log_p
  )

  extra_params <- unique(as.character(extra_params))
  extra_params <- extra_params[nzchar(extra_params)]
  if (length(extra_params) > 0) {
    for (nm in extra_params) {
      pr <- pick_param(nm)
      beta <- NA_real_
      if (!is.null(fit$coefficients) && (nm %in% names(fit$coefficients))) {
        beta <- as.numeric(fit$coefficients[[nm]])
      }
      out[[paste0("beta_", nm)]] <- beta
      out[[paste0("p_", nm)]] <- pr$p
      out[[paste0("log_p_", nm)]] <- pr$log_p
    }
  }

  out
}

fit_panel <- function(
  counts,
  x,
  r_um,
  AP_um,
  ML_um,
  theta,
  sf,
  brdu_pos = NULL,
  edu_pos = NULL,
  k_r = 5,
  k_theta = 8,
  k_uv = 30,
  k_uvr = 10,
  k_r_uvr = 5,
  k_rtheta = c(5, 8),
  gamma = 2.5,
  method = "REML",
  min_nonzero = 10,
  k_coupling_theta = 8,
  coupling_group = NULL,
  engine = c("bam", "auto", "gam"),
  return = c("table", "list")
) {
  if (!is.matrix(counts)) stop("counts must be a numeric matrix (n_cells x n_genes).")
  if (nrow(counts) != length(x)) stop("counts and x length mismatch.")
  if (nrow(counts) != length(r_um)) stop("counts and r_um length mismatch.")
  if (nrow(counts) != length(AP_um) || nrow(counts) != length(ML_um)) stop("counts and AP_um/ML_um length mismatch.")
  if (nrow(counts) != length(theta)) stop("counts and theta length mismatch.")
  if (nrow(counts) != length(sf)) stop("counts and sf length mismatch.")
  if (!is.numeric(min_nonzero) || length(min_nonzero) != 1 || !is.finite(min_nonzero) || min_nonzero < 0) {
    stop("min_nonzero must be a single finite number >= 0.")
  }
  if (!is.null(brdu_pos) || !is.null(edu_pos)) {
    if (is.null(brdu_pos) || is.null(edu_pos)) stop("Provide both brdu_pos and edu_pos, or neither.")
    if (length(brdu_pos) != nrow(counts) || length(edu_pos) != nrow(counts)) stop("brdu_pos/edu_pos length mismatch.")
  }

  theta <- wrap_theta(theta)
  coupling <- fit_inm_coupling_gam(x = x, theta = theta, k_theta = k_coupling_theta, method = method, group = coupling_group)
  r_um <- as.numeric(r_um)

  genes <- colnames(counts)
  if (is.null(genes)) genes <- paste0("gene_", seq_len(ncol(counts)))

  engine <- match.arg(engine)
  return <- match.arg(return)
  has_pos <- !is.null(brdu_pos) && !is.null(edu_pos)
  na_result_row <- function(gene) {
    data.frame(
      gene = gene,
      inm_r2 = coupling$r2,
      p_spatial = NA_real_,
      p_cycle = NA_real_,
      p_interaction = NA_real_,
      p_apml = NA_real_,
      p_apml_r_um = NA_real_,
      p_brdu_pos = NA_real_,
      p_edu_pos = NA_real_,
      p_brdu_edu = NA_real_,
      cycle_amp_link = NA_real_,
      spatial_grad_link = NA_real_,
      gating_index_link = NA_real_,
      stringsAsFactors = FALSE
    )
  }

  results <- vector("list", length = ncol(counts))
  for (j in seq_len(ncol(counts))) {
    y <- counts[, j]
    nonzero <- sum(is.finite(y) & (y > 0))
    if (nonzero < min_nonzero) {
      results[[j]] <- na_result_row(genes[j])
      next
    }

    fit <- tryCatch(
      {
      fit_gene_gam(
        y = y,
        sf = sf,
        r_um = r_um,
        theta = theta,
        AP_um = AP_um,
        ML_um = ML_um,
        brdu_pos = brdu_pos,
        edu_pos = edu_pos,
        k_r = k_r,
        k_theta = k_theta,
        k_uv = k_uv,
        k_uvr = k_uvr,
        k_r_uvr = k_r_uvr,
        k_rtheta = k_rtheta,
        gamma = gamma,
        method = method,
        engine = engine
      )
    },
    error = function(e) {
        warning(sprintf("fit_panel: gene=%s failed: %s", genes[j], conditionMessage(e)))
        NULL
      }
    )
    if (is.null(fit)) {
      results[[j]] <- na_result_row(genes[j])
      next
    }

    pv <- extract_component_pvals(fit)
    es <- effect_sizes_from_fit(fit, r_um = r_um, AP_um = AP_um, ML_um = ML_um, has_pos = has_pos)
    results[[j]] <- data.frame(
      gene = genes[j],
      inm_r2 = coupling$r2,
      p_spatial = pv$p_spatial,
      p_cycle = pv$p_cycle,
      p_interaction = pv$p_interaction,
      p_apml = pv$p_apml,
      p_apml_r_um = pv$p_apml_r_um,
      p_brdu_pos = pv$p_brdu_pos,
      p_edu_pos = pv$p_edu_pos,
      p_brdu_edu = pv$p_brdu_edu,
      cycle_amp_link = es$cycle_amp_link,
      spatial_grad_link = es$spatial_grad_link,
      gating_index_link = es$gating_index_link,
      stringsAsFactors = FALSE
    )
  }

  table <- do.call(rbind, results)
  if (return == "list") {
    return(list(table = table, coupling = coupling, r_um = r_um, theta = theta))
  }
  table
}

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
  r,
  theta,
  brdu_pos = NULL,
  edu_pos = NULL,
  batch = NULL,
  k_r = 5,
  k_theta = 8,
  k_int = c(5, 8),
  method = "REML",
  engine = c("auto", "gam", "bam"),
  bam_discrete = TRUE
) {
  if (length(y) != length(sf) || length(y) != length(r) || length(y) != length(theta)) {
    stop("y, sf, r, theta must have the same length.")
  }
  theta <- wrap_theta(theta)

  df <- data.frame(y = y, sf = sf, r = r, theta = theta)
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
  knots <- list(theta = c(0, 2 * pi))

  base_terms <- "offset(log(sf))"
  if (isTRUE(has_batch)) base_terms <- paste(base_terms, "+ batch")
  smooth_terms <- " + s(r, k = k_r) + s(theta, bs = 'cc', k = k_theta) + ti(r, theta, bs = c('tp','cc'), k = k_int)"
  pos_terms <- if (!has_pos) "" else " + brdu_pos + edu_pos + brdu_pos:edu_pos"
  rhs <- paste0(base_terms, pos_terms, smooth_terms)
  formula <- as.formula(paste("y ~", rhs))
  engine <- match.arg(engine)
  if (engine == "auto") {
    engine <- if (nrow(df) >= 20000) "bam" else "gam"
  }
  if (engine == "bam") {
    method_bam <- if (identical(method, "REML")) "fREML" else method
    bam(
      formula,
      data = df,
      family = nb(),
      method = method_bam,
      knots = knots,
      select = TRUE,
      discrete = isTRUE(bam_discrete)
    )
  } else {
    gam(formula, data = df, family = nb(), method = method, knots = knots, select = TRUE)
  }
}

effect_sizes_from_fit <- function(
  fit,
  r,
  has_pos = FALSE,
  has_batch = FALSE,
  batch_ref = NULL,
  sf_ref = 1.0,
  n_theta = 64
) {
  r_q <- as.numeric(quantile(r, probs = c(0.1, 0.9), names = FALSE))
  r_lo <- r_q[[1]]
  r_hi <- r_q[[2]]
  r_mid <- as.numeric(median(r))

  theta_grid <- seq(0, 2 * pi, length.out = n_theta + 1)
  theta_grid <- theta_grid[-length(theta_grid)]

  add_pos <- function(nd) {
    if (isTRUE(has_pos)) {
      nd$brdu_pos <- 0
      nd$edu_pos <- 0
    }
    if (isTRUE(has_batch) && !is.null(batch_ref)) {
      nd$batch <- batch_ref
    }
    nd
  }

  amp_at_r <- function(r0) {
    nd <- data.frame(r = rep(r0, length(theta_grid)), theta = theta_grid, sf = sf_ref)
    nd <- add_pos(nd)
    eta <- as.numeric(predict(fit, newdata = nd, type = "link"))
    if (any(!is.finite(eta))) return(NA_real_)
    max(eta) - min(eta)
  }

  cycle_amp_mid <- amp_at_r(r_mid)
  cycle_amp_lo <- amp_at_r(r_lo)
  cycle_amp_hi <- amp_at_r(r_hi)
  gating_index <- cycle_amp_hi - cycle_amp_lo

  nd_grad <- data.frame(r = c(r_lo, r_hi), theta = c(0, 0), sf = sf_ref)
  nd_grad <- add_pos(nd_grad)
  eta_grad <- as.numeric(predict(fit, newdata = nd_grad, type = "link"))
  spatial_grad <- if (length(eta_grad) == 2 && all(is.finite(eta_grad))) eta_grad[[2]] - eta_grad[[1]] else NA_real_

  list(
    cycle_amp_link = cycle_amp_mid,
    spatial_grad_link = spatial_grad,
    gating_index_link = gating_index
  )
}

extract_component_pvals <- function(fit) {
  empty_pvals <- list(
    p_spatial = NA_real_, p_cycle = NA_real_, p_interaction = NA_real_,
    p_brdu_pos = NA_real_, p_edu_pos = NA_real_, p_brdu_edu = NA_real_
  )
  sum_fit <- summary(fit)
  st <- sum_fit$s.table
  pt <- sum_fit$p.table
  if (is.null(st)) return(empty_pvals)

  rn <- rownames(st)
  rn_compact <- gsub("\\s+", "", rn)

  p_col <- if ("p-value" %in% colnames(st)) "p-value" else NA_character_
  if (is.na(p_col)) return(empty_pvals)

  pick_p <- function(regex, label) {
    idx <- which(grepl(regex, rn_compact, perl = TRUE))
    if (length(idx) == 0) return(NA_real_)
    if (length(idx) > 1) warning(sprintf("Multiple smooth terms matched %s; using first.", label))
    as.numeric(st[idx[[1]], p_col])
  }

  pick_p_param <- function(name) {
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
    if (is.finite(p0)) return(p0)

    b <- fit$coefficients
    if (is.null(b) || !(name %in% names(b))) return(NA_real_)
    V <- fit$Vp
    if (is.null(V) || !is.matrix(V) || nrow(V) != length(b) || ncol(V) != length(b)) return(NA_real_)
    se <- sqrt(diag(V))
    if (any(!is.finite(se))) return(NA_real_)
    names(se) <- names(b)
    if (!(name %in% names(se))) return(NA_real_)
    tval <- as.numeric(b[[name]] / se[[name]])
    if (!is.finite(tval)) return(NA_real_)
    df <- fit$df.residual
    if (is.null(df) || !is.finite(df) || df <= 0) {
      2 * pnorm(abs(tval), lower.tail = FALSE)
    } else {
      2 * pt(abs(tval), df = df, lower.tail = FALSE)
    }
  }

  list(
    p_spatial = pick_p("^s\\(r\\)$", "s(r)"),
    p_cycle = pick_p("^s\\(theta\\)$", "s(theta)"),
    p_interaction = pick_p("^ti\\(r,theta\\)$", "ti(r,theta)"),
    p_brdu_pos = pick_p_param("brdu_pos"),
    p_edu_pos = pick_p_param("edu_pos"),
    p_brdu_edu = pick_p_param("brdu_pos:edu_pos")
  )
}

fit_panel <- function(
  counts,
  x,
  theta,
  sf,
  brdu_pos = NULL,
  edu_pos = NULL,
  k_r = 5,
  k_theta = 8,
  k_int = c(5, 8),
  method = "REML",
  min_nonzero = 10,
  k_coupling_theta = 8,
  coupling_group = NULL,
  engine = c("auto", "gam", "bam"),
  return = c("table", "list")
) {
  if (!is.matrix(counts)) stop("counts must be a numeric matrix (n_cells x n_genes).")
  if (nrow(counts) != length(x)) stop("counts and x length mismatch.")
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
  r <- coupling$r_hat

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

    fit <- fit_gene_gam(
      y = y,
      sf = sf,
      r = r,
      theta = theta,
      brdu_pos = brdu_pos,
      edu_pos = edu_pos,
      k_r = k_r,
      k_theta = k_theta,
      k_int = k_int,
      method = method,
      engine = engine
    )
    pv <- extract_component_pvals(fit)
    es <- effect_sizes_from_fit(fit, r = r, has_pos = has_pos)
    results[[j]] <- data.frame(
      gene = genes[j],
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
  }

  table <- do.call(rbind, results)
  if (return == "list") {
    return(list(table = table, coupling = coupling, r = r, theta = theta))
  }
  table
}

import anndata as ad
import numpy as np
import scipy.linalg
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


def select_orthogonal_genes(
    adata: ad.AnnData,
    pre_selected_gene_names: list[str],
    k: int,
    layer: str | None = None,
    n_components_rs_svd: int | None = None,
    add_intercept_to_selected_regression: bool = True,
    random_state_rs_svd: int | None = 42,
    verbose: bool = False,
) -> list[str]:
    """
    Selects k additional genes that are most orthogonal to n pre-selected genes
    and capture the most remaining variance from an AnnData object.

    The method proceeds by:
    1. Partitioning genes into pre-selected and candidate sets.
    2. Calculating residuals: For each candidate gene, its expression is regressed
       against the pre-selected genes (optionally with an intercept). The
       residuals from this regression form the 'X_residual' matrix. This step
       makes candidate gene profiles orthogonal to the pre-selected ones.
    3. Dimensionality reduction of residuals: Randomized SVD is performed on
       X_residual to find its principal components (axes of most variance).
    4. Gene selection from loadings: QR decomposition with column pivoting is
       applied to the SVD component loadings. The first k pivot genes are
       chosen as they best represent the space of residual variance.

    Args:
        adata: AnnData object with gene expression (genes as vars, samples as obs).
        pre_selected_gene_names: List of names of the n pre-selected genes.
                                 Must be present in adata.var_names.
        k: Number of additional genes to select.
        layer: Layer in adata to use for expression. If None, uses adata.X.
        n_components_rs_svd: Number of principal components to compute for the
                             residual matrix using Randomized SVD.
                             If None, a default value is chosen: max(k + 20, 50),
                             capped by the dimensions of the residual matrix.
        add_intercept_to_selected_regression: If True (default), an intercept term is
                                   effectively added when regressing candidate
                                   genes against pre-selected genes. This ensures
                                   residuals are orthogonal to a constant offset
                                   plus the space spanned by pre-selected genes.
                                   If no pre-selected genes are given, this results
                                   in centering the candidate genes.
        random_state_rs_svd: Random state for Randomized SVD for reproducibility.
        verbose: If True, logger.info progress messages.

    Returns:
        A list of up to k selected additional gene names. Fewer than k genes may
        be returned if there are not enough candidate genes or if the
        dimensionality of the residual space is less than k.
    """

    if verbose:
        logger.info(f"Starting orthogonal gene selection for k={k} additional genes.")

    if k == 0:
        if verbose:
            logger.info("k=0, returning empty list.")
        return []

    if layer is None:
        expression_matrix_source = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object.")
        expression_matrix_source = adata.layers[layer]

    # Ensure expression_matrix is a NumPy array (densify if sparse)
    # This is important as X_residual will likely become dense anyway.
    if not isinstance(expression_matrix_source, np.ndarray):
        if verbose:
            logger.info(
                "Expression matrix source is sparse, converting to dense numpy array. You may run out of memory for big arrays."
            )
        expression_matrix = expression_matrix_source.toarray()
    else:
        expression_matrix = expression_matrix_source.copy()  # Work on a copy

    # Ensure float64 type for numerical stability in lstsq and svd
    if expression_matrix.dtype != np.float64:
        expression_matrix = expression_matrix.astype(np.float64)

    all_gene_names = adata.var_names.tolist()

    # Validate pre_selected_gene_names and get their indices
    valid_pre_selected_gene_names = []
    invalid_pre_selected_gene_names = []
    for gene_name in pre_selected_gene_names:
        if gene_name not in all_gene_names:
            invalid_pre_selected_gene_names.append(gene_name)
        else:
            valid_pre_selected_gene_names.append(gene_name)

    if invalid_pre_selected_gene_names:
        logger.warning(
            f"The following pre-selected genes were not found in adata.var_names and will be ignored: "
            f"{', '.join(invalid_pre_selected_gene_names)}"
        )

    pre_selected_indices = [all_gene_names.index(name) for name in valid_pre_selected_gene_names]

    candidate_gene_indices = [
        i for i, name in enumerate(all_gene_names) if name not in valid_pre_selected_gene_names
    ]
    candidate_gene_names_actual = [all_gene_names[i] for i in candidate_gene_indices]

    if not candidate_gene_names_actual:
        if verbose:
            logger.info(
                "No candidate genes available after excluding pre-selected ones. Returning empty list."
            )
        return []

    num_samples = expression_matrix.shape[0]

    # --- Step 1: Data Partitioning ---
    if len(pre_selected_indices) > 0:
        X_S_initial = np.ascontiguousarray(expression_matrix[:, pre_selected_indices])
    else:
        X_S_initial = np.empty((num_samples, 0), dtype=expression_matrix.dtype)

    X_candidate = np.ascontiguousarray(expression_matrix[:, candidate_gene_indices])

    if verbose:
        logger.info(f"Number of samples (m_obs): {num_samples}")
        logger.info(f"Number of valid pre-selected genes (n): {X_S_initial.shape[1]}")
        logger.info(f"Number of candidate genes (p-n): {X_candidate.shape[1]}")

    if X_candidate.shape[1] == 0:
        if verbose:
            logger.info("X_candidate matrix is empty (0 columns). Returning empty list.")
        return []

    if k >= X_candidate.shape[1]:
        if verbose:
            logger.info(
                f"k ({k}) is >= number of candidate genes ({X_candidate.shape[1]}). "
                "Returning all candidate genes."
            )
        return sorted(candidate_gene_names_actual)  # Return sorted for consistency

    # --- Step 2: Orthogonalization (Calculate Residuals) ---
    if X_S_initial.shape[1] > 0:
        X_S_for_lstsq = X_S_initial
        if add_intercept_to_selected_regression:
            if verbose:
                logger.info("Adding intercept column to pre-selected gene matrix for regression.")
            X_S_for_lstsq = np.hstack([X_S_initial, np.ones((num_samples, 1), dtype=X_S_initial.dtype)])

        try:
            Beta_matrix, _residuals_sum_sq, rank, _singular_values = np.linalg.lstsq(
                X_S_for_lstsq, X_candidate, rcond=None
            )
            if rank < X_S_for_lstsq.shape[1] and verbose:
                logger.warning(
                    f"Rank deficiency detected in lstsq for regressing out pre-selected genes. "
                    f"Effective rank {rank} < {X_S_for_lstsq.shape[1]} columns. "
                    "This may indicate collinearity in pre-selected genes (plus intercept if used)."
                )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Linear algebra error during lstsq: {e}. This might happen if X_S_initial (plus intercept) is ill-conditioned or contains all zeros."
            ) from e

        X_residual = X_candidate - (X_S_for_lstsq @ Beta_matrix)
    else:
        if add_intercept_to_selected_regression:
            if verbose:
                logger.info("No pre-selected genes. Centering candidate genes to form residuals.")
            X_residual = X_candidate - np.mean(X_candidate, axis=0, keepdims=True)
        else:
            if verbose:
                logger.info(
                    "No pre-selected genes and no intercept term. Using candidate genes directly as residuals."
                )
            X_residual = X_candidate

    # --- Step 3: Dimensionality Reduction of Residuals (Randomized SVD) ---
    m_obs_res, p_minus_n_res = X_residual.shape

    if m_obs_res == 0 or p_minus_n_res == 0:  # Should not happen if X_candidate was not empty
        if verbose:
            logger.info("Residual matrix is empty. Cannot proceed with SVD. Returning empty list.")
        return []

    actual_max_rank_res = min(m_obs_res, p_minus_n_res)

    if actual_max_rank_res == 0:
        return []  # Should be caught earlier

    if n_components_rs_svd is None:
        d_target = max(k + 20, 50)
    else:
        d_target = n_components_rs_svd

    d = min(d_target, actual_max_rank_res)
    d = max(1, d)  # Ensure d is at least 1 if possible

    if verbose:
        logger.info(
            f"Performing Randomized SVD on residual matrix ({m_obs_res} x {p_minus_n_res}) "
            f"with d={d} components."
        )

    if d > min(X_residual.shape):  # Should be caught by d = min(d_target, actual_max_rank_res)
        d = min(X_residual.shape)
    if d == 0:  # Should be caught by d = max(1,d) if actual_max_rank_res >=1
        if verbose:
            logger.info("Number of SVD components d is 0. Cannot select genes. Returning empty list.")
        return []

    try:
        # Note: randomized_svd requires n_components <= min(X.shape)
        # It may internally adjust n_components further if n_oversamples is large.
        _U_res, _S_res, V_T_res = randomized_svd(
            X_residual, n_components=d, n_iter="auto", random_state=random_state_rs_svd
        )
    except ValueError as e:
        # Example: "n_components must be <= min(X.shape)" which should be handled by 'd' calculation
        # or "Setting n_oversamples=10 too high"
        raise RuntimeError(
            f"Error during randomized_svd with d={d} on matrix of shape {X_residual.shape}: {e}. "
            "Try a smaller n_components_rs_svd or check data."
        ) from e

    # --- Step 4: Selection of k Genes from Residual Loadings using QR with column pivoting ---
    L_matrix = V_T_res  # shape: d x p_minus_n_res

    if verbose:
        logger.info(
            f"Performing QR decomposition with pivoting on loadings matrix ({L_matrix.shape[0]} x {L_matrix.shape[1]}) "
            f"to select up to {k} genes."
        )

    try:
        # P_L_indices contains indices *into L_matrix columns* (i.e., into candidate_gene_names_actual)
        _Q_L, _R_L, P_L_indices = scipy.linalg.qr(L_matrix, pivoting=True, mode="economic")
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Linear algebra error during QR decomposition of SVD loadings: {e}.") from e

    num_selectable_via_qr = len(P_L_indices)  # This is min(d, p_minus_n_res)

    num_to_select = min(k, num_selectable_via_qr)
    selected_indices_in_candidates = P_L_indices[:num_to_select]

    selected_additional_gene_names = [candidate_gene_names_actual[i] for i in selected_indices_in_candidates]

    if verbose:
        if len(selected_additional_gene_names) < k:
            logger.info(
                f"Selected {len(selected_additional_gene_names)} (fewer than k={k}) additional genes due to "
                f"limited candidate genes or residual dimensionality."
            )
        else:
            logger.info(f"Selected {len(selected_additional_gene_names)} additional genes.")
        if (
            len(selected_additional_gene_names) <= 20 and len(selected_additional_gene_names) > 0
        ):  # logger.info if not too many
            logger.info(f"Selected genes: {selected_additional_gene_names}")

    return selected_additional_gene_names


def calculate_variance_capture_in_global_pcs(
    adata: ad.AnnData,
    selected_gene_names: list[str],
    n_global_pcs_target: int = 30,
    layer: str | None = None,
    center_data: bool = True,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """
    Calculates the percentage of variance from the top N global Principal Components (PCs)
    that is captured by a selected set of genes.

    This method evaluates how well the selected genes can reconstruct the information
    contained within the dominant structural dimensions of the entire dataset.

    Method:
    1. Perform PCA on the full dataset (all genes, optionally centered) to get global PCs
       and their explained variances (eigenvalues).
    2. The sum of variances of the top `n_global_pcs_target` global PCs is the
       "total target structured variance".
    3. For each of these top N global PCs, regress its scores against the expression
       values of the `selected_gene_names`.
    4. The variance of the predicted PC scores from these regressions is summed up. This
       is the "variance captured by selected genes within the target PC space".
    5. The final metric is (variance captured / total target structured variance) * 100.

    Args:
        adata: AnnData object with gene expression (genes as vars, samples as obs).
        selected_gene_names: List of names of the selected genes.
        n_global_pcs_target: The number of top global PCs to define the target
                             structured variance (e.g., 20).
        layer: Layer in adata to use for expression. If None, uses adata.X.
        center_data: If True (default), data is centered (mean subtracted per gene)
                     before any PCA or regression. Essential for meaningful PCA.
        verbose: If True, logger.info progress messages.

    Returns:
        A tuple containing:
        - percentage_captured_in_global_pcs (float): The percentage of variance from
          the top N global PCs captured by the selected genes.
        - var_captured_by_selected_in_target_space (float): Absolute variance
          captured by selected genes within the target global PC space.
        - total_variance_in_target_global_pcs (float): Total variance summed across
          the top N global PCs.
    """

    if verbose:
        logger.info(f"Starting calculation of variance capture within top {n_global_pcs_target} global PCs.")

    if layer is None:
        expression_matrix_source = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object.")
        expression_matrix_source = adata.layers[layer]

    if not isinstance(expression_matrix_source, np.ndarray):
        if verbose:
            logger.info("Expression matrix source is sparse, converting to dense numpy array.")
        expression_matrix_full = expression_matrix_source.toarray()
    else:
        expression_matrix_full = expression_matrix_source.copy()

    if expression_matrix_full.dtype != np.float64:
        expression_matrix_full = expression_matrix_full.astype(np.float64)

    num_samples, num_total_genes = expression_matrix_full.shape

    if num_samples < 2:
        raise ValueError("Cannot perform PCA or calculate variance with fewer than 2 samples.")
    if num_total_genes == 0:
        raise ValueError("Expression matrix has no genes (0 columns).")

    if n_global_pcs_target <= 0:
        raise ValueError("n_global_pcs_target must be positive.")

    if center_data:
        if verbose:
            logger.info("Centering full data matrix.")
        # Mean per gene (column)
        mean_full_data = np.mean(expression_matrix_full, axis=0, keepdims=True)
        X_full_processed = expression_matrix_full - mean_full_data
    else:
        X_full_processed = expression_matrix_full

    # --- Step 1: Global PCA to define target subspace ---
    # Determine max possible components for global PCA
    max_components_global_pca = min(num_samples, num_total_genes)

    if n_global_pcs_target > max_components_global_pca:
        logger.warning(
            f"n_global_pcs_target ({n_global_pcs_target}) is greater than the maximum possible "
            f"components ({max_components_global_pca}). Using {max_components_global_pca} instead."
        )
        n_global_pcs_target = max_components_global_pca

    if n_global_pcs_target == 0:  # Should be caught by earlier check but as a safeguard
        if verbose:
            logger.info("Effective n_global_pcs_target is 0. Cannot proceed.")
        return (0.0, 0.0, 0.0) if selected_gene_names else (np.nan, 0.0, 0.0)

    if verbose:
        logger.info(
            f"Performing global PCA on full dataset ({num_samples}x{num_total_genes}) "
            f"to get top {n_global_pcs_target} PCs."
        )

    global_pca = PCA(n_components=n_global_pcs_target, svd_solver="auto", random_state=42)
    # Shape: (num_samples, n_global_pcs_target)
    global_pc_scores_target = global_pca.fit_transform(X_full_processed)
    global_pc_eigenvalues_target = global_pca.explained_variance_

    total_variance_in_target_global_pcs = np.sum(global_pc_eigenvalues_target)

    if verbose:
        logger.info(
            f"Total variance in top {n_global_pcs_target} global PCs: "
            f"{total_variance_in_target_global_pcs:.4f}"
        )

    if np.isclose(total_variance_in_target_global_pcs, 0.0):
        logger.warning(
            "Total variance in target global PCs is zero. This implies no structure "
            "in the top PCs of the data or all data is constant."
        )
        return (np.nan, 0.0, 0.0)  # Or (0.0, 0.0, 0.0) if selected genes also capture 0

    # --- Step 2 & 3: Evaluate selected genes against target global PCs ---
    all_gene_names_list = adata.var_names.tolist()
    valid_selected_gene_names = [name for name in selected_gene_names if name in all_gene_names_list]

    if not valid_selected_gene_names:
        if verbose:
            logger.info("No valid selected genes provided or found. Captured variance is 0.")
        return (0.0, 0.0, total_variance_in_target_global_pcs)

    selected_indices = [all_gene_names_list.index(name) for name in valid_selected_gene_names]
    # Use the same processed (e.g., centered) data for the selected subset
    X_selected_processed = X_full_processed[:, selected_indices]

    if X_selected_processed.shape[1] == 0:  # Should be caught by `if not valid_selected_gene_names`
        if verbose:
            logger.info("X_selected_processed matrix is empty. Captured variance is 0.")
        return (0.0, 0.0, total_variance_in_target_global_pcs)

    var_captured_by_selected_in_target_space = 0.0

    if verbose:
        logger.info(
            f"Regressing each of the top {n_global_pcs_target} global PCs "
            f"against the {X_selected_processed.shape[1]} selected genes."
        )

    for i in range(n_global_pcs_target):
        current_global_pc_scores = global_pc_scores_target[:, i]  # Target for this iteration

        # Regress: current_global_pc_scores = X_selected_processed @ beta + epsilon
        try:
            beta, _residuals_sum_sq, _rank, _singular_values = np.linalg.lstsq(
                X_selected_processed, current_global_pc_scores, rcond=None
            )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Linear algebra error during lstsq for PC {i} against selected genes: {e}. "
                "This might happen if X_selected_processed is ill-conditioned or all zeros."
            ) from e

        predicted_pc_scores = X_selected_processed @ beta

        # Variance of the part of the global PC explained by selected genes
        # ddof=0 because global_pc_eigenvalues_target (from sklearn PCA) are based on N
        explained_variance_for_this_pc = np.var(predicted_pc_scores, ddof=0)
        var_captured_by_selected_in_target_space += explained_variance_for_this_pc

    if verbose:
        logger.info(
            f"Total variance captured by selected genes within the target PC space: "
            f"{var_captured_by_selected_in_target_space:.4f}"
        )

    percentage_captured_in_global_pcs = (
        var_captured_by_selected_in_target_space / total_variance_in_target_global_pcs
    ) * 100.0

    if verbose:
        logger.info(
            f"Percentage of target global PC variance captured: {percentage_captured_in_global_pcs:.2f}%"
        )

    return (
        percentage_captured_in_global_pcs,
        var_captured_by_selected_in_target_space,
        total_variance_in_target_global_pcs,
    )

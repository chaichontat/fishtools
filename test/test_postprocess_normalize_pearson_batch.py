import os
import sys

import anndata as ad
import numpy as np

from fishtools.postprocess.utils_h5ad import normalize_pearson


def test_normalize_pearson_batch_key_normalizes_separately() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

    # The test suite stubs `scanpy` in `test/conftest.py`. This test exercises
    # the real Scanpy implementation.
    scanpy_stub = sys.modules.get("scanpy")
    for key in list(sys.modules):
        if key == "scanpy" or key.startswith("scanpy."):
            del sys.modules[key]

    try:
        import scanpy as sc

        x = np.array(
            [
                [18, 1, 1],
                [18, 1, 1],
                [18, 1, 1],
                [18, 1, 1],
                [18, 1, 1],
                [18, 1, 1],
                [1, 18, 1],
                [1, 18, 1],
                [1, 18, 1],
                [1, 18, 1],
                [1, 18, 1],
                [1, 18, 1],
            ],
            dtype=np.int64,
        )
        obs = {"batch": ["a"] * 6 + ["b"] * 6}
        var = {"gene": ["g0", "g1", "g2"]}
        adata = ad.AnnData(X=x, obs=obs, var=var)

        adata_batch, _ = normalize_pearson(adata.copy(), n_top_genes=3, batch_key="batch", clip=10.0)
        assert adata_batch.n_vars == 3
        assert adata_batch.X.dtype == np.float32

        # Reference: do per-batch residuals manually and stack.
        sc.experimental.pp.highly_variable_genes(
            adata,
            flavor="pearson_residuals",
            n_top_genes=3,
            batch_key="batch",
            clip=10.0,
        )
        adata_ref = adata[:, adata.var["highly_variable"]].copy()
        residuals = np.empty((adata_ref.n_obs, adata_ref.n_vars), dtype=np.float32)
        for value in adata_ref.obs["batch"].unique():
            mask = (adata_ref.obs["batch"] == value).to_numpy(dtype=bool, copy=False)
            sub = adata_ref[mask].copy()
            sc.experimental.pp.normalize_pearson_residuals(sub, clip=10.0)
            residuals[mask] = np.asarray(sub.X, dtype=np.float32)
        assert np.allclose(adata_batch.X, residuals, atol=1e-6, rtol=1e-6)

        # Sanity: global residuals should differ from per-batch residuals for this contrived example.
        adata_global = adata_ref.copy()
        sc.experimental.pp.normalize_pearson_residuals(adata_global, clip=10.0)
        assert not np.allclose(adata_batch.X, np.asarray(adata_global.X), atol=1e-6, rtol=1e-6)
    finally:
        for key in list(sys.modules):
            if key == "scanpy" or key.startswith("scanpy."):
                del sys.modules[key]
        if scanpy_stub is not None:
            sys.modules["scanpy"] = scanpy_stub

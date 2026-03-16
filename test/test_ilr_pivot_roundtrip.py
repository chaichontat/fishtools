import numpy as np

from scripts.gam.plot_simplex_native_proj import ilr_basis_pivot
from scripts.gam.plot_simplex_native_proj import inv_ilr


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def test_ilr_pivot_roundtrip_recovers_simplex() -> None:
    rng = np.random.default_rng(0)
    n = 128
    k = 11
    # Draw random simplex points with no exact zeros.
    u = rng.dirichlet(alpha=np.ones(k), size=n).astype(np.float64, copy=False)

    V = ilr_basis_pivot(k)
    z = np.log(u) @ V  # (n, k-1)
    u_hat = inv_ilr(z, V=V)

    assert u_hat.shape == u.shape
    assert np.allclose(np.sum(u_hat, axis=1), 1.0, atol=1e-8)
    assert np.all(u_hat >= 0.0) and np.all(u_hat <= 1.0)
    assert np.allclose(u_hat, u, atol=1e-8, rtol=1e-8)


def test_ilr_basis_is_orthonormal_in_clr_space() -> None:
    k = 9
    V = ilr_basis_pivot(k)
    # Columns should be orthonormal in the ambient Euclidean space and sum to 0.
    gram = V.T @ V
    assert np.allclose(gram, np.eye(k - 1), atol=1e-12)
    assert np.allclose(np.sum(V, axis=0), 0.0, atol=1e-12)


def test_ilr_inverse_matches_softmax_of_clr() -> None:
    rng = np.random.default_rng(1)
    n = 64
    k = 7
    V = ilr_basis_pivot(k)
    z = rng.normal(size=(n, k - 1))
    clr = z @ V.T
    u1 = inv_ilr(z, V=V)
    u2 = _softmax(clr)
    assert np.allclose(u1, u2, atol=1e-12)


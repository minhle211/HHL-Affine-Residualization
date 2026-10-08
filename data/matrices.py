"""Structured families of real symmetric test matrices.

Every generator has the signature ``gen(N, kappa, rng) -> A`` and returns an
``N x N`` symmetric positive-definite matrix. Families with a prescribed
spectrum hit ``kappa`` exactly; ``wishart`` and the Laplacians have their own
natural condition number and get a diagonal shift when it exceeds ``kappa``
(otherwise the natural value is kept).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate import haar_orthogonal, sample_spectrum  # noqa: E402


def _from_spectrum(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Q = haar_orthogonal(len(s), rng)
    A = (Q * s) @ Q.T
    return 0.5 * (A + A.T)


def wishart(N: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """``M M^T + 1e-3 I`` with Gaussian ``M`` (the vector-benchmark matrices).

    Its natural condition number grows like ``N^2 / 1e-3``; when that exceeds
    ``kappa`` a diagonal shift brings it down to exactly ``kappa``.
    """
    M = rng.standard_normal((N, N))
    return _shift_to_kappa(M @ M.T + 1e-3 * np.eye(N), kappa)


def _shift_to_kappa(L: np.ndarray, kappa: float) -> np.ndarray:
    lam = np.linalg.eigvalsh(L)
    lo, hi = lam[0], lam[-1]
    if hi / lo <= kappa:
        return L
    # L + c I has condition number exactly kappa
    c = (hi - kappa * lo) / (kappa - 1)
    return L + c * np.eye(L.shape[0])


def laplacian_1d(N: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet 1D Laplacian ``tridiag(-1, 2, -1)``; eigenvalues ~ k^2."""
    L = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    return _shift_to_kappa(L, kappa)


def laplacian_2d(N: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """5-point 2D Dirichlet Laplacian on an ``n x n`` grid (``N = n^2``).

    For non-square ``N`` the grid is ``n x (N // n)`` padded to ``N`` with
    the 1D Laplacian on the leftover nodes.
    """
    n = int(np.floor(np.sqrt(N)))
    m = N // n
    T = lambda k: 2 * np.eye(k) - np.eye(k, k=1) - np.eye(k, k=-1)
    L = np.kron(T(n), np.eye(m)) + np.kron(np.eye(n), T(m))
    r = N - n * m
    if r:
        A = np.zeros((N, N))
        A[: n * m, : n * m] = L
        A[n * m :, n * m :] = 2 * T(r)
        L = A
    return _shift_to_kappa(L, kappa)


def clustered_small(N: int, kappa: float, rng: np.random.Generator,
                    n_small: int | None = None) -> np.ndarray:
    """A few tiny eigenvalues near ``1/kappa``; the rest in ``[0.1, 1]``.

    The best case for deflation: removing ``n_small`` modes drops the
    condition number from ``kappa`` to ~10.
    """
    n_small = max(1, N // 8) if n_small is None else n_small
    small = (1.0 / kappa) * 10 ** rng.uniform(0, 0.5, n_small)
    small[0] = 1.0 / kappa
    rest = 10 ** rng.uniform(-1, 0, N - n_small)
    rest[0] = 1.0
    return _from_spectrum(np.sort(np.concatenate([small, rest])), rng)


def log_uniform(N: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Log-uniform spectrum in ``[1/kappa, 1]`` (the existing dataset family)."""
    return _from_spectrum(sample_spectrum(N, kappa, rng), rng)


def sparse_banded(N: int, kappa: float, rng: np.random.Generator,
                  bandwidth: int = 2) -> np.ndarray:
    """Random symmetric banded matrix, shifted to SPD with condition ``kappa``."""
    B = np.zeros((N, N))
    for d in range(1, bandwidth + 1):
        v = rng.standard_normal(N - d)
        B += np.diag(v, d) + np.diag(v, -d)
    B += np.diag(rng.standard_normal(N))
    lam = np.linalg.eigvalsh(B)
    lo, hi = lam[0], lam[-1]
    # shift so that the spectrum becomes [s, s + (hi - lo)] with ratio kappa
    s = (hi - lo) / (kappa - 1)
    return B + (s - lo) * np.eye(N)


def _shift_spectrum(lam: np.ndarray, kappa: float) -> np.ndarray:
    lo, hi = lam.min(), lam.max()
    if hi / lo <= kappa:
        return lam
    return lam + (hi - kappa * lo) / (kappa - 1)


def spectrum(family: str, N: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Eigenvalues of ``MATRIX_FAMILIES[family](N, kappa, rng)`` (same distribution).

    Analytic for the Laplacians, sampled directly for prescribed spectra, and
    ``eigvalsh`` of a generated matrix for ``wishart`` / ``sparse_banded``.
    """
    if family == "laplacian_1d":
        j = np.arange(1, N + 1)
        return _shift_spectrum(2 - 2 * np.cos(np.pi * j / (N + 1)), kappa)
    if family == "laplacian_2d":
        n = int(np.floor(np.sqrt(N)))
        m = N // n
        t = lambda k: 2 - 2 * np.cos(np.pi * np.arange(1, k + 1) / (k + 1))
        lam = (t(n)[:, None] + t(m)[None, :]).ravel()
        r = N - n * m
        if r:
            lam = np.concatenate([lam, 2 * t(r)])
        return _shift_spectrum(np.sort(lam), kappa)
    if family == "clustered_small":
        n_small = max(1, N // 8)
        small = (1.0 / kappa) * 10 ** rng.uniform(0, 0.5, n_small)
        small[0] = 1.0 / kappa
        rest = 10 ** rng.uniform(-1, 0, N - n_small)
        rest[0] = 1.0
        return np.sort(np.concatenate([small, rest]))
    if family == "log_uniform":
        return sample_spectrum(N, kappa, rng)
    if family == "wishart":
        M = rng.standard_normal((N, N))
        return _shift_spectrum(np.linalg.eigvalsh(M @ M.T) + 1e-3, kappa)
    if family == "sparse_banded":
        B = np.zeros((N, N))
        for d in range(1, 3):
            v = rng.standard_normal(N - d)
            B += np.diag(v, d) + np.diag(v, -d)
        B += np.diag(rng.standard_normal(N))
        lam = np.linalg.eigvalsh(B)
        return lam + ((lam[-1] - lam[0]) / (kappa - 1) - lam[0])
    return np.linalg.eigvalsh(MATRIX_FAMILIES[family](N, kappa, rng))


MATRIX_FAMILIES = {
    "wishart": wishart,
    "laplacian_1d": laplacian_1d,
    "laplacian_2d": laplacian_2d,
    "clustered_small": clustered_small,
    "log_uniform": log_uniform,
    "sparse_banded": sparse_banded,
}

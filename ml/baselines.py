"""Cheap classical approximations of the deflation vector z.

All providers have the signature ``(A, b, budget) -> z`` and approximate
``z = sum_{i in R} (beta_i / lambda_i) u_i`` for the ``budget`` modes with the
largest energy ``|beta_i / lambda_i|``, using an approximate eigenbasis:

* ``lanczos_z``: ``m`` Lanczos steps on ``A`` from ``b`` (Krylov subspace
  ``K_m(A, b)``), Rayleigh-Ritz, select Ritz pairs by energy. Only needs
  matrix-vector products.
* ``inverse_iteration_z``: a few steps of block (subspace) inverse iteration.
  Each step needs linear solves with ``A``; for dense small ``A`` that already
  costs as much as solving the system, so this is a quality reference rather
  than a cheap method.
* ``randomized_svd_z``: randomized range finder (Halko et al.) on the shifted
  operator ``B = sigma I - A`` (SPD) or ``sigma^2 I - A^2`` (indefinite),
  whose dominant eigenvectors are the small-|lambda| modes of ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _z_from_ritz(theta: np.ndarray, V: np.ndarray, b: np.ndarray, budget: int) -> np.ndarray:
    beta = V.T @ b
    energy = np.abs(beta) / np.maximum(np.abs(theta), 1e-300)
    k = min(budget, len(theta))
    idx = np.argsort(-energy)[:k]
    return V[:, idx] @ (beta[idx] / theta[idx])


def lanczos_z(A: np.ndarray, b: np.ndarray, budget: int, n_iter: int | None = None) -> np.ndarray:
    N = A.shape[0]
    m = min(N, n_iter if n_iter is not None else budget + 2)
    Q = np.zeros((N, m))
    alpha = np.zeros(m)
    beta_l = np.zeros(m)
    q = b / np.linalg.norm(b)
    q_prev = np.zeros(N)
    steps = m
    for j in range(m):
        Q[:, j] = q
        w = A @ q
        alpha[j] = q @ w
        w = w - alpha[j] * q - (beta_l[j - 1] * q_prev if j > 0 else 0.0)
        w -= Q[:, : j + 1] @ (Q[:, : j + 1].T @ w)  # full reorthogonalization
        if j + 1 < m:
            beta_l[j] = np.linalg.norm(w)
            if beta_l[j] < 1e-12:
                steps = j + 1
                break
            q_prev, q = q, w / beta_l[j]
    T = np.diag(alpha[:steps]) + np.diag(beta_l[: steps - 1], 1) + np.diag(beta_l[: steps - 1], -1)
    theta, S = np.linalg.eigh(T)
    return _z_from_ritz(theta, Q[:, :steps] @ S, b, budget)


@dataclass
class LanczosDeflation:
    z: np.ndarray  # deflation vector over the k smallest-|theta| Ritz pairs
    removed_ritz: np.ndarray  # the k removed Ritz values
    next_ritz: float  # smallest |theta| not removed, shrunk by its error bound
    matvecs: int  # products with A (Lanczos steps)
    converged: bool  # all k+1 targeted Ritz values within relative error tol
    ritz_values: np.ndarray = field(repr=False)


class DiagonalOperator:
    """``diag(lam)`` as a matrix-free operator (``A @ v`` costs O(N)).

    Lanczos and CG on ``Q diag(lam) Q^T`` from ``b`` are, in exact
    arithmetic, the rotated iterates of the same methods on ``diag(lam)``
    from ``Q^T b``, so spectra suffice to emulate them at large N.
    """

    def __init__(self, lam: np.ndarray):
        self.lam = np.asarray(lam, dtype=float)
        self.shape = (len(self.lam), len(self.lam))

    def __matmul__(self, v: np.ndarray) -> np.ndarray:
        return self.lam * v if v.ndim == 1 else self.lam[:, None] * v


class LanczosRun:
    """Incremental Lanczos (full reorthogonalization) started from ``b``.

    ``step()`` performs one matrix-vector product with ``A`` (any object with
    ``@`` and ``shape``); ``ritz()`` returns the Ritz values, the
    eigenvectors ``S`` of the tridiagonal matrix and the residual norms
    ``|beta_m s_{m,i}|``. Ritz vectors are formed only on demand.
    """

    def __init__(self, A, b: np.ndarray):
        self.A = A
        self.N = A.shape[0]
        self._Q = np.empty((self.N, min(self.N, 32)))
        self._Q[:, 0] = b / np.linalg.norm(b)
        self.alpha: list[float] = []
        self.beta: list[float] = []
        self.matvecs = 0
        self.exhausted = False
        self._ritz_cache: tuple | None = None

    def step(self) -> None:
        if self.exhausted:
            return
        m = len(self.alpha)
        q = self._Q[:, m]
        w = self.A @ q
        self.matvecs += 1
        a = float(q @ w)
        self.alpha.append(a)
        Qm = self._Q[:, : m + 1]
        w = w - Qm @ (Qm.T @ w)
        w = w - Qm @ (Qm.T @ w)
        nb = float(np.linalg.norm(w))
        self.beta.append(nb)
        if nb < 1e-12 * max(1.0, abs(a)) or m + 1 == self.N:
            self.exhausted = True
            return
        if m + 1 == self._Q.shape[1]:
            grown = np.empty((self.N, min(self.N, 2 * self._Q.shape[1])))
            grown[:, : m + 1] = self._Q
            self._Q = grown
        self._Q[:, m + 1] = w / nb

    def ritz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = len(self.alpha)
        key = (m, self.exhausted)
        if self._ritz_cache is not None and self._ritz_cache[0] == key:
            return self._ritz_cache[1]
        theta, S = eigh_tridiagonal(np.array(self.alpha), np.array(self.beta[: m - 1]))
        res = np.zeros(m) if self.exhausted else np.abs(self.beta[m - 1] * S[-1, :])
        self._ritz_cache = (key, (theta, S, res))
        return theta, S, res

    def ritz_vectors(self, idx: np.ndarray) -> np.ndarray:
        _, S, _ = self.ritz()
        return self._Q[:, : S.shape[0]] @ S[:, idx]

    def value_error(self) -> np.ndarray:
        """Relative Ritz-value error bound ``r_i / |theta_i|``.

        For symmetric ``A`` some eigenvalue lies within ``r_i`` of
        ``theta_i`` (Bauer-Fike), so this bound is rigorous. Gap-based
        ``r^2 / gap`` bounds are tighter but unreliable before the Ritz
        values have separated, which is exactly the regime of interest.
        """
        theta, _, res = self.ritz()
        if self.exhausted:
            return np.zeros_like(theta)
        return res / np.maximum(np.abs(theta), 1e-300)

    def converged(self, k: int, tol: float) -> bool:
        """True once the ``k`` smallest-|theta| Ritz values have relative error <= ``tol``."""
        if len(self.alpha) < k:
            return False
        if self.exhausted:
            return True
        theta, _, _ = self.ritz()
        idx = np.argsort(np.abs(theta))[:k]
        return bool(np.all(self.value_error()[idx] <= tol))

    def deflation(self, b: np.ndarray, k: int, tol: float) -> LanczosDeflation:
        theta, _, _ = self.ritz()
        order = np.argsort(np.abs(theta))
        sel = order[:k]
        if k:
            V = self.ritz_vectors(sel)
            z = V @ ((V.T @ b) / theta[sel])
        else:
            z = np.zeros_like(b, dtype=float)
        if k < len(theta):
            # shrink the edge by its error bound so lambda_{k+1} stays in the domain
            err = float(min(self.value_error()[order[k]], 0.5))
            nxt = float(np.abs(theta[order[k]])) * (1 - err)
        else:
            nxt = float("nan")
        return LanczosDeflation(
            z=z,
            removed_ritz=theta[sel],
            next_ritz=nxt,
            matvecs=self.matvecs,
            converged=self.converged(k + 1, tol),
            ritz_values=theta,
        )


def lanczos_deflation(A, b: np.ndarray, k: int, tol: float = 1e-4,
                      max_iter: int | None = None) -> LanczosDeflation:
    """Deflate the ``k`` smallest-|lambda| modes with Lanczos from ``b``.

    Runs until the ``k + 1`` smallest-|theta| Ritz values have relative
    error bound ``r / |theta| <= tol`` (so ``next_ritz``, shrunk by its
    bound, is a safe lower estimate of ``lambda_{k+1}``, the edge of the
    deflated spectrum), the Krylov space is exhausted, or ``max_iter`` steps.
    Only matrix-vector products with ``A`` are used (``matvecs``).
    """
    b = np.asarray(b, dtype=float)
    max_iter = A.shape[0] if max_iter is None else max_iter
    run = LanczosRun(A, b)
    while run.matvecs < max_iter and not run.exhausted:
        run.step()
        if run.matvecs >= k + 1 and run.converged(k + 1, tol):
            break
    return run.deflation(b, k, tol)


def inverse_iteration_z(A: np.ndarray, b: np.ndarray, budget: int, n_iter: int = 2,
                        block: int | None = None, seed: int = 0) -> np.ndarray:
    N = A.shape[0]
    p = min(N, block if block is not None else budget + 1)
    rng = np.random.default_rng(seed)
    X = np.column_stack([b, rng.normal(size=(N, p - 1))]) if p > 1 else b[:, None]
    X, _ = np.linalg.qr(X)
    for _ in range(n_iter):
        X, _ = np.linalg.qr(np.linalg.solve(A, X))
    theta, S = np.linalg.eigh(X.T @ A @ X)
    return _z_from_ritz(theta, X @ S, b, budget)


def randomized_svd_z(A: np.ndarray, b: np.ndarray, budget: int, oversample: int = 2,
                     n_power: int = 2, seed: int = 0, spd: bool = False) -> np.ndarray:
    """``spd=True`` uses the better-separated shift ``sigma I - A``; the default
    ``sigma^2 I - A^2`` is valid for any Hermitian ``A``."""
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    # power-iteration estimate of ||A|| (matvecs only)
    v = rng.normal(size=N)
    for _ in range(10):
        v = A @ v
        v /= np.linalg.norm(v)
    sigma = 1.05 * np.linalg.norm(A @ v)
    B = (sigma * np.eye(N) - A) if spd else (sigma**2 * np.eye(N) - A @ A)
    p = min(N, budget + oversample)
    Y = B @ rng.normal(size=(N, p))
    for _ in range(n_power):
        Y, _ = np.linalg.qr(Y)
        Y = B @ Y
    Qb, _ = np.linalg.qr(Y)
    theta, S = np.linalg.eigh(Qb.T @ A @ Qb)
    return _z_from_ritz(theta, Qb @ S, b, budget)


def exact_z(A: np.ndarray, b: np.ndarray, budget: int, criterion: str = "energy") -> np.ndarray:
    from hhl.residualization import budgeted_affine_residualization

    return budgeted_affine_residualization(A, b, budget, criterion).z


def lanczos_smallest_z(A: np.ndarray, b: np.ndarray, budget: int) -> np.ndarray:
    """Provider form of ``lanczos_deflation`` (smallest-|lambda| modes)."""
    return lanczos_deflation(A, b, budget).z


BASELINES = {
    "lanczos": lanczos_z,
    "lanczos_smallest": lanczos_smallest_z,
    "inverse_iter": inverse_iteration_z,
    "rand_svd": randomized_svd_z,
}

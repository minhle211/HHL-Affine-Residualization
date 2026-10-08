"""Emulated QSVT matrix inversion and analytic optimal-QLSA references.

QSVT with a block-encoding of ``A / alpha`` applies a bounded polynomial
``P(A / alpha)`` to ``|b>``; the postselected state is exactly
``P(A / alpha) b / ||P(A / alpha) b||`` and the success probability is
``||P(A / alpha) b||^2``. We therefore emulate QSVT exactly by evaluating
the polynomial (via the eigendecomposition, or with a Clenshaw recurrence
where every matrix-vector product stands for one block-encoding query) and
charge ``degree`` queries per run.

Polynomial: odd Chebyshev approximation of the smooth surrogate
``g(x) = (1 - exp(-(beta x)^2)) / (M x)`` of ``1 / (M x)``.
``beta = sqrt(ln(2 / eps)) / a`` makes ``g`` match ``1/(Mx)`` to relative
error ``eps/2`` for ``|x| >= a``; ``M = G beta / (1 - margin)`` with
``G = max_u (1 - exp(-u^2)) / u ~ 0.638`` keeps ``|g| <= 1 - margin`` on
``[-1, 1]``; the Chebyshev series is truncated where its coefficient tail
drops below ``eps / (2M)``. Degree is ``O(kappa log(kappa / eps))``, matching
the standard QSVT inversion bound (Gilyen et al. 2019; Martyn et al. 2021).
For ``|x| < a`` the polynomial stays below 1, so leakage onto modes outside
the targeted domain is bounded rather than amplified like ``1 / x``.

Analytic references (expected block-encoding queries, Hermitian ``A``,
``kappa = alpha / lambda_min``):

* Dalzell 2024, arXiv:2406.12086, Eq. (48):
  ``56.0 kappa + 1.05 kappa ln(sqrt(1 - eps^2) / eps) + 2.78 ln(kappa)^3 + 3.17``
* Jennings et al. 2024, arXiv:2305.11352v3, Eq. (175) with ``alpha = 1``:
  ``835.4 kappa + ceil(kappa ln(2 / (sqrt(1 + eps/4) - 1)) + 2)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.fft import dct

from .cost import aa_cost, aa_rounds, repeat_cost

_MARGIN = 0.05
_u = np.linspace(1e-6, 5.0, 500_001)
_G = float(np.max(-np.expm1(-(_u**2)) / _u))  # ~0.6382
# Quantization of the domain edge: a is rounded *down* to this log-grid so
# polynomials can be cached; the domain only grows, so accuracy is kept at
# the price of <= 2^(1/16) - 1 ~ 4.4% extra degree.
_GRID = 16


@dataclass(frozen=True)
class InversePoly:
    coeffs: np.ndarray  # Chebyshev coefficients (odd)
    degree: int
    a: float  # domain edge: accurate for a <= |x| <= 1
    eps: float
    beta: float
    M: float  # P(x) ~ 1 / (M x) on the domain

    def surrogate(self, x: np.ndarray) -> np.ndarray:
        """The smooth target ``g``; ``|P - g| <= eps / (2M)`` on ``[-1, 1]``."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        nz = x != 0
        out[nz] = -np.expm1(-((self.beta * x[nz]) ** 2)) / (self.M * x[nz])
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate on ``[-1, 1]`` via ``T_k(cos t) = cos(k t)`` (odd terms only)."""
        x = np.asarray(x, dtype=float)
        if np.any(np.abs(x) > 1 + 1e-12):
            return C.chebval(x, self.coeffs)
        t = np.arccos(np.clip(x, -1.0, 1.0)).reshape(-1, 1)
        k = np.arange(1, self.degree + 1, 2)
        ck = self.coeffs[1::2]
        out = np.zeros(t.shape[0])
        step = max(1, (1 << 22) // max(1, t.shape[0]))
        for s in range(0, len(k), step):
            out += np.cos(t * k[s : s + step]) @ ck[s : s + step]
        return out.reshape(x.shape)


def _cheb_coeffs(f, n: int) -> np.ndarray:
    k = np.arange(n)
    x = np.cos(np.pi * (k + 0.5) / n)
    c = dct(f(x), type=2) / n
    c[0] /= 2
    return c


_MAX_NODES = 1 << 22  # Chebyshev nodes; caps emulation at kappa ~ 1e5-1e6


@lru_cache(maxsize=1024)
def _inverse_poly_cached(a: float, eps: float) -> InversePoly:
    if not 0 < a <= 1:
        raise ValueError(f"domain edge a must be in (0, 1], got {a}")
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0, 1)")
    beta = math.sqrt(math.log(2 / eps)) / a
    if 2 * beta * math.sqrt(math.log(4 / eps)) > _MAX_NODES:
        raise ValueError(f"kappa = {1 / a:.3g} is too large to emulate the QSVT polynomial")
    M = _G * beta / (1 - _MARGIN)
    tol = eps / (2 * M)

    def g(x):
        out = np.zeros_like(x)
        nz = x != 0
        out[nz] = -np.expm1(-((beta * x[nz]) ** 2)) / (M * x[nz])
        return out

    n = 1 << max(6, math.ceil(math.log2(2 * beta * math.sqrt(math.log(4 / eps)) + 64)))
    while True:
        c = _cheb_coeffs(g, n)
        if np.sum(np.abs(c[-n // 8:])) < 1e-3 * tol or n >= _MAX_NODES:
            break
        n *= 2
    c[0::2] = 0.0
    tail = np.cumsum(np.abs(c[::-1]))[::-1]  # tail[j] = sum_{i >= j} |c_i|
    above = np.nonzero(tail > tol)[0]
    d = int(above[-1]) if above.size else 1
    if d % 2 == 0:
        d += 1
    return InversePoly(coeffs=c[: d + 1].copy(), degree=d, a=a, eps=eps, beta=beta, M=M)


def quantize_edge(a: float) -> float:
    """Round ``a`` down to the cache grid (never shrinks the accurate domain)."""
    return 2.0 ** (math.floor(math.log2(a) * _GRID) / _GRID)


def inverse_poly(kappa: float, eps: float, quantize: bool = True) -> InversePoly:
    """Bounded odd polynomial approximating ``1 / (M x)`` on ``1/kappa <= |x| <= 1``."""
    a = 1.0 / float(kappa)
    if quantize:
        a = quantize_edge(a)
    return _inverse_poly_cached(min(a, 1.0), float(eps))


def qsvt_degree(kappa: float, eps: float) -> int:
    return inverse_poly(kappa, eps).degree


def clenshaw_apply(A: np.ndarray, v: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, int]:
    """``sum_k c_k T_k(A) v`` with one matvec per degree; returns (result, matvecs)."""
    dt = np.result_type(A, v, float)
    if len(coeffs) == 1:
        return coeffs[0] * v.astype(dt), 0
    b1 = coeffs[-1] * v.astype(dt)
    b2 = np.zeros_like(b1)
    matvecs = 0
    for ck in coeffs[-2:0:-1]:
        b1, b2 = ck * v + 2 * (A @ b1) - b2, b1
        matvecs += 1
    return coeffs[0] * v + A @ b1 - b2, matvecs + 1


@dataclass
class QSVTResult:
    y: np.ndarray  # estimate of A^{-1} b_hat (b_hat = b / ||b||), with magnitude
    p_success: float
    degree: int
    queries_per_run: int
    queries_aa: float
    queries_repeat: float
    aa_rounds: int
    alpha: float
    kappa_target: float
    poly: InversePoly


def solve_qsvt(
    A: np.ndarray,
    b: np.ndarray,
    lam_min: float,
    eps: float = 1e-3,
    alpha: float | None = None,
    method: str = "eig",
    quantize: bool = True,
) -> QSVTResult:
    """Emulate QSVT inversion of Hermitian ``A`` on ``b``.

    The polynomial is accurate for ``lam_min <= |lambda| <= alpha``;
    ``alpha`` defaults to ``||A||`` (the block-encoding normalization).
    """
    A = np.asarray(A)
    b = np.asarray(b)
    nb = np.linalg.norm(b)
    if nb == 0:
        raise ValueError("b must be non-zero")
    bh = b / nb
    if alpha is None:
        alpha = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    kappa_t = alpha / abs(lam_min)
    poly = inverse_poly(kappa_t, eps, quantize)
    if method == "eig":
        lam, U = np.linalg.eigh(A)
        state = U @ (poly(lam / alpha) * (U.conj().T @ bh))
    elif method == "clenshaw":
        state, _ = clenshaw_apply(A / alpha, bh, poly.coeffs)
    else:
        raise ValueError("method must be 'eig' or 'clenshaw'")
    p = float(np.real(np.vdot(state, state)))
    if np.isrealobj(A) and np.isrealobj(b):
        state = np.real(state)
    return QSVTResult(
        y=state * poly.M / alpha,
        p_success=p,
        degree=poly.degree,
        queries_per_run=poly.degree,
        queries_aa=aa_cost(poly.degree, p),
        queries_repeat=repeat_cost(poly.degree, p),
        aa_rounds=aa_rounds(p),
        alpha=alpha,
        kappa_target=kappa_t,
        poly=poly,
    )


def dalzell_queries(kappa: float, eps: float) -> float:
    """Dalzell 2024 (arXiv:2406.12086) Eq. (48) expected query upper bound."""
    return (56.0 * kappa + 1.05 * kappa * math.log(math.sqrt(1 - eps**2) / eps)
            + 2.78 * math.log(kappa) ** 3 + 3.17)


def jennings_queries(kappa: float, eps: float) -> float:
    """Jennings et al. 2024 (arXiv:2305.11352v3) Eq. (175), alpha = 1."""
    return 835.4 * kappa + math.ceil(kappa * math.log(2 / (math.sqrt(1 + eps / 4) - 1)) + 2)


OPTIMAL_REFERENCES = {"dalzell": dalzell_queries, "jennings": jennings_queries}


def optimal_reference_queries(kappa: float, eps: float, method: str = "dalzell") -> float:
    return OPTIMAL_REFERENCES[method](kappa, eps)

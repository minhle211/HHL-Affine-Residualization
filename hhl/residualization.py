"""Budgeted affine residualization (classical eigenmode deflation).

For Hermitian ``A = U diag(lam) U^H`` and ``beta = U^H b``, the exact solution
is ``x = sum_i (beta_i / lam_i) u_i``. Given a budget ``k`` we remove the
``k`` "hardest" modes classically:

    z      = sum_{i in R} (beta_i / lam_i) u_i          (explanation vector)
    b_res  = P_keep (b - A z)                           (residual RHS)

and solve ``A y = b_res`` with HHL tuned to the remaining spectrum. The
projection ``P_keep`` is applied explicitly: any numerical leakage onto
removed modes would otherwise be amplified by ``1 / lam_min`` in HHL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Residualization:
    z: np.ndarray
    b_res: np.ndarray
    removed: np.ndarray  # indices (into eigh ordering) of removed modes
    kept: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    kappa: float
    kappa_eff: float
    energies: np.ndarray = field(repr=False)

    @property
    def kept_eigvals(self) -> np.ndarray:
        return self.eigvals[self.kept]


def check_hermitian(A: np.ndarray, atol: float = 1e-10) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, A.conj().T, atol=atol):
        raise ValueError("A must be Hermitian; use hermitian_embedding(A, b)")


def condition_number(eigvals: np.ndarray) -> float:
    a = np.abs(eigvals)
    if a.size == 0:
        return 1.0
    return float(a.max() / a.min())


def hermitian_embedding(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embed a general square ``A`` as ``H = [[0, A], [A^H, 0]]``, ``b' = [b, 0]``.

    If ``H [u; v] = [b; 0]`` with ``A`` invertible, then ``v = A^{-1} b`` and
    ``u = 0``; recover the solution with ``embedded_solution``. ``H`` has
    eigenvalues ``+-sigma_i`` (the singular values of ``A``), so HHL must use a
    signed clock register.
    """
    A = np.asarray(A)
    n = A.shape[0]
    Z = np.zeros_like(A)
    H = np.block([[Z, A], [A.conj().T, Z]])
    be = np.concatenate([np.asarray(b), np.zeros(n, dtype=np.asarray(b).dtype)])
    return H, be


def embedded_solution(x_emb: np.ndarray, n: int) -> np.ndarray:
    return x_emb[n:]


def pad_to_power_of_two(
    A: np.ndarray, b: np.ndarray, pad_value: float | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad Hermitian ``A`` with ``pad_value * I`` and ``b`` with zeros.

    The padding eigenvalue defaults to the median |eigenvalue| so it does not
    change the condition number; padded components of ``b`` are zero so the
    padded modes are never excited.
    """
    n = A.shape[0]
    N = 1 << max(1, (n - 1).bit_length())
    if N == n:
        return A, b, n
    if pad_value is None:
        pad_value = float(np.median(np.abs(np.linalg.eigvalsh(A))))
    Ap = np.eye(N, dtype=A.dtype) * pad_value
    Ap[:n, :n] = A
    bp = np.zeros(N, dtype=np.asarray(b).dtype)
    bp[:n] = b
    return Ap, bp, n


def mode_energies(eigvals: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Per-mode contribution ``|beta_i / lam_i|`` to the solution norm."""
    return np.abs(beta) / np.abs(eigvals)


def select_modes(
    eigvals: np.ndarray, beta: np.ndarray, budget: int, criterion: str = "energy"
) -> np.ndarray:
    """Indices of the ``budget`` modes to remove.

    ``criterion="energy"`` removes the largest ``|beta_i/lam_i|`` (the modes
    that dominate the solution); ``"smallest"`` removes the smallest
    ``|lam_i|`` (the modes that set kappa), independent of ``b``.
    """
    budget = int(max(0, min(budget, len(eigvals) - 1)))
    if budget == 0:
        return np.array([], dtype=int)
    if criterion == "energy":
        order = np.argsort(-mode_energies(eigvals, beta), kind="stable")
    elif criterion == "smallest":
        order = np.argsort(np.abs(eigvals), kind="stable")
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    return np.sort(order[:budget])


def budgeted_affine_residualization(
    A: np.ndarray,
    b: np.ndarray,
    budget: int,
    criterion: str = "energy",
    z: np.ndarray | None = None,
    project: bool = True,
) -> Residualization:
    """Remove ``budget`` hard modes classically and form the residual RHS.

    If ``z`` is given (e.g. predicted by an ML model or a cheap baseline), it
    is used instead of the exact deflation vector, while the removed-mode set
    is still chosen by ``criterion``. With ``project=True`` the residual is
    projected onto the kept modes, so any error of ``z`` along removed modes
    is *not* corrected downstream (reduced-kappa circuit). With
    ``project=False`` the raw residual ``b - A z`` is returned and the caller
    must use a full-kappa circuit to resolve it.
    """
    A = np.asarray(A)
    b = np.asarray(b)
    check_hermitian(A)
    lam, U = np.linalg.eigh(A)
    if np.any(lam == 0):
        raise ValueError("A is singular")
    beta = U.conj().T @ b
    removed = select_modes(lam, beta, budget, criterion)
    kept = np.setdiff1d(np.arange(len(lam)), removed)

    if z is None:
        coeff = np.zeros_like(beta, dtype=np.result_type(beta, lam))
        coeff[removed] = beta[removed] / lam[removed]
        z = U @ coeff
    else:
        z = np.asarray(z)

    r = b - A @ z
    if project:
        Uk = U[:, kept]
        r = Uk @ (Uk.conj().T @ r)
    if np.isrealobj(A) and np.isrealobj(b):
        z = np.real(z)
        r = np.real(r)

    return Residualization(
        z=z,
        b_res=r,
        removed=removed,
        kept=kept,
        eigvals=lam,
        eigvecs=U,
        kappa=condition_number(lam),
        kappa_eff=condition_number(lam[kept]),
        energies=mode_energies(lam, beta),
    )


def hard_projector(A: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the removed (hard) eigenmodes of ``A``."""
    _, U = np.linalg.eigh(A)
    Ur = U[:, removed]
    return Ur @ Ur.conj().T

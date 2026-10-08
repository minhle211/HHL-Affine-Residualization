"""End-to-end hybrid solver: residualize -> tune QPE -> HHL -> recombine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .circuit import choose_hhl_params, fidelity, run_hhl
from .cost import aa_cost, hhl_cost_metrics
from .qsvt import inverse_poly, solve_qsvt
from .residualization import (
    budgeted_affine_residualization,
    check_hermitian,
    condition_number,
    embedded_solution,
    hermitian_embedding,
    pad_to_power_of_two,
)

# A z-provider maps (A, b, budget) -> z (an approximation of the deflation
# vector). ``None`` means exact eigen-deflation.
ZProvider = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class HybridResult:
    x: np.ndarray
    z: np.ndarray
    y: np.ndarray
    b_res_norm: float
    metrics: dict = field(default_factory=dict)


def _prepare(A: np.ndarray, b: np.ndarray, embed: str):
    """Make the system Hermitian and power-of-two sized; return an unpacker."""
    A = np.asarray(A)
    b = np.asarray(b)
    n = A.shape[0]
    hermitian = np.allclose(A, A.conj().T, atol=1e-10)
    if embed == "always" or (embed == "auto" and not hermitian):
        H, be = hermitian_embedding(A, b)
        Hp, bp, _ = pad_to_power_of_two(H, be)
        return Hp, bp, (lambda v: embedded_solution(v[: 2 * n], n)), True
    if not hermitian:
        raise ValueError("A is not Hermitian and embed='never'")
    Ap, bp, _ = pad_to_power_of_two(A, b)
    return Ap, bp, (lambda v: v[:n]), False


def solve_hybrid(
    A: np.ndarray,
    b: np.ndarray,
    z_provider: ZProvider | None = None,
    budget: int = 1,
    criterion: str = "smallest",
    mode: str = "reduced",
    solver: str = "hhl",
    embed: str = "auto",
    extra_bits: int = 2,
    max_clock: int = 12,
    n_clock: int | None = None,
    compute_resources: bool = False,
    tol: float = 1e-12,
    eps: float = 1e-2,
) -> HybridResult:
    """Solve ``A x = b`` as ``x = z + ||b_res|| * y`` with ``A y = b_res / ||b_res||``.

    Parameters
    ----------
    z_provider : callable or None
        ``None`` uses exact eigen-deflation of the ``budget`` hardest modes.
        Otherwise ``z = z_provider(A, b, budget)`` (for embedded/padded
        systems the provider receives the original ``A, b`` and the result is
        lifted into the embedded space).
    budget : int
        Number of eigenmodes removed classically. ``budget=0`` is standard HHL.
    mode : {"reduced", "full"}
        ``"reduced"`` projects the residual onto the kept modes and tunes QPE
        to the kept spectrum (kappa_eff). ``"full"`` keeps the raw residual
        ``b - A z`` and tunes QPE to the full spectrum, so it can correct
        errors of ``z`` along hard modes at full-kappa cost.
    solver : {"hhl", "qsvt", "classical"}
        ``"qsvt"`` uses emulated QSVT inversion with a polynomial accurate on
        the targeted spectrum. ``"classical"`` replaces HHL by an exact dense
        solve (used for tests and as a reference).
    eps : float
        Target precision for the query-cost model (HHL simulation error
        budget, QSVT polynomial accuracy).
    """
    if mode not in ("reduced", "full"):
        raise ValueError("mode must be 'reduced' or 'full'")
    if solver not in ("hhl", "qsvt", "classical"):
        raise ValueError("solver must be 'hhl', 'qsvt' or 'classical'")

    A0, b0 = np.asarray(A), np.asarray(b)
    Ah, bh, unpack, embedded = _prepare(A0, b0, embed)
    x_true_h = np.linalg.solve(Ah, bh)

    z_h = None
    if z_provider is not None:
        z0 = np.asarray(z_provider(A0, b0, budget))
        if embedded:
            z_h = np.zeros(Ah.shape[0], dtype=np.result_type(z0, Ah))
            z_h[A0.shape[0] : 2 * A0.shape[0]] = z0
        else:
            z_h = np.zeros(Ah.shape[0], dtype=np.result_type(z0, Ah))
            z_h[: A0.shape[0]] = z0

    res = budgeted_affine_residualization(
        Ah, bh, budget, criterion=criterion, z=z_h, project=(mode == "reduced")
    )
    b_res = res.b_res
    nr = float(np.linalg.norm(b_res))

    if mode == "reduced":
        target = res.kept_eigvals
    else:
        target = res.eigvals
    lam_abs = np.abs(target)

    metrics: dict = {
        "N": int(A0.shape[0]),
        "N_hermitian": int(Ah.shape[0]),
        "embedded": embedded,
        "budget": int(len(res.removed)),
        "criterion": criterion,
        "mode": mode,
        "solver": solver,
        "kappa": res.kappa,
        "kappa_eff": res.kappa_eff,
        "kappa_target": condition_number(target),
        "removed_modes": res.removed.tolist(),
        "removed_eigvals": res.eigvals[res.removed].tolist(),
        "b_res_norm": nr,
        "b_res_rel": nr / float(np.linalg.norm(bh)),
    }

    alpha = float(np.max(np.abs(res.eigvals)))
    y = np.zeros_like(bh, dtype=complex)
    if nr <= tol * np.linalg.norm(bh):
        metrics["hhl_skipped"] = True
        metrics.update(p_success=1.0, p_success_clock0=1.0, hhl_fidelity=1.0,
                       queries_per_run=0, queries_aa=0.0, queries_repeat=0.0)
    elif solver == "classical":
        y = np.linalg.solve(Ah, b_res / nr).astype(complex)
        metrics["hhl_skipped"] = False
    elif solver == "qsvt":
        out = solve_qsvt(Ah, b_res, lam_abs.min(), eps=eps, alpha=alpha)
        y = out.y.astype(complex)
        y_ref = np.linalg.solve(Ah, b_res / nr)
        metrics.update(
            hhl_skipped=False,
            degree=out.degree,
            p_success=out.p_success,
            queries_per_run=out.queries_per_run,
            queries_aa=out.queries_aa,
            queries_repeat=out.queries_repeat,
            aa_rounds=out.aa_rounds,
            hhl_fidelity=fidelity(y, y_ref),
            hhl_rel_error=float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)),
        )
    else:
        params = choose_hhl_params(
            lam_abs.min(),
            lam_abs.max(),
            signed=bool(np.any(target < 0)),
            extra_bits=extra_bits,
            n_clock=n_clock,
            max_clock=max_clock,
        )
        out = run_hhl(Ah, b_res, params, compute_resources=compute_resources)
        y = out.y
        y_ref = np.linalg.solve(Ah, b_res / nr)
        metrics.update(
            hhl_skipped=False,
            n_clock=params.n_clock,
            clock_capped=params.clock_capped,
            t=params.t,
            C=params.C,
            p_success=out.p_success,
            p_success_clock0=out.p_success_clock0,
            hhl_fidelity=fidelity(y, y_ref),
            hhl_rel_error=float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)),
        )
        metrics.update(hhl_cost_metrics(params.n_clock, params.t, alpha, eps, out.p_success))
        if out.resources is not None:
            metrics.update({f"{k}": v for k, v in out.resources.items()})

    x_h = res.z + nr * y
    if np.isrealobj(A0) and np.isrealobj(b0):
        metrics["max_imag"] = float(np.max(np.abs(np.imag(x_h)))) if x_h.size else 0.0
        x_h = np.real(x_h)
        y = np.real(y)

    metrics["fidelity"] = fidelity(x_h, x_true_h)
    metrics["rel_error"] = float(
        np.linalg.norm(x_h - x_true_h) / np.linalg.norm(x_true_h)
    )
    x = unpack(x_h)
    return HybridResult(x=x, z=unpack(res.z), y=unpack(y), b_res_norm=nr, metrics=metrics)


def solve_standard(A: np.ndarray, b: np.ndarray, **kwargs) -> HybridResult:
    """Standard HHL (no residualization), tuned to the full spectrum."""
    kwargs.pop("budget", None)
    return solve_hybrid(A, b, z_provider=None, budget=0, **kwargs)


# ------------------------------------------------- adaptive deflation + QSVT


def estimate_qsvt_queries(theta: np.ndarray, c: np.ndarray, removed: np.ndarray,
                          edge: float, alpha: float, eps: float) -> float:
    """AA-adjusted QSVT queries for the residual, estimated from Ritz data.

    ``c`` are the coefficients of ``b_hat`` in the Ritz basis (``S[0, :]``,
    since Lanczos starts from ``b_hat``). Deflating the removed Ritz pairs
    zeroes their coefficients and the success probability is estimated as
    ``sum_kept |g(theta_i / alpha) c_i|^2 / sum_kept |c_i|^2`` with the
    polynomial's smooth target ``g`` (within ``eps/(2M)`` of ``P``).
    """
    poly = inverse_poly(alpha / edge, eps)
    keep = np.ones(len(theta), dtype=bool)
    keep[removed] = False
    ck = c[keep]
    norm2 = float(ck @ ck)
    if norm2 <= 1e-30 * float(c @ c):
        return 0.0
    p = float(np.sum((poly.surrogate(theta[keep] / alpha) * ck) ** 2) / norm2)
    return aa_cost(poly.degree, max(min(p, 1.0), 1e-300))


@dataclass
class BudgetChoice:
    k: int
    deflation: object  # ml.baselines.LanczosDeflation (None for k = 0)
    matvecs: int  # all Lanczos matvecs spent during the search
    est_quantum: dict  # k -> estimated AA-adjusted QSVT queries
    est_total: dict  # k -> est_quantum[k] + w * matvecs needed to reach k


@dataclass
class LanczosTrajectory:
    """Everything the budget search can observe, recorded once per run.

    ``checkpoints`` holds ``(matvecs, est_q0, [(k, est_q, deflation), ...])``
    for every step at which convergence was checked; the search for any cost
    ratio ``w`` is a replay of this list (``pick_budget``).
    """

    checkpoints: list
    est_q0_initial: float


def _is_checkpoint(m: int) -> bool:
    # every step early on, then ~16 checks per doubling of the step count
    return m <= 32 or m % max(1, m // 16) == 0


def lanczos_trajectory(A, b: np.ndarray, eps: float, alpha: float, lam_min: float,
                       k_max: int | None = None, tol: float | None = None,
                       max_matvecs: int | None = None) -> LanczosTrajectory:
    """Run Lanczos from ``b`` and record deflation candidates as they converge.

    Candidate ``k`` appears at the first checkpoint where the ``k + 1``
    smallest Ritz values have relative error ``<= tol``. Stops at ``k_max``,
    ``max_matvecs`` or when the Krylov space is exhausted. ``A`` may be any
    operator with ``@`` and ``shape`` (e.g. ``ml.baselines.DiagonalOperator``).
    """
    from ml.baselines import LanczosRun

    b = np.asarray(b, dtype=float)
    N = A.shape[0]
    k_max = N - 1 if k_max is None else min(k_max, N - 1)
    max_matvecs = N if max_matvecs is None else max_matvecs
    tol = 0.1 * eps if tol is None else tol
    bh = b / np.linalg.norm(b)
    run = LanczosRun(A, bh)
    cps = []
    k_next = 1
    while k_next <= k_max and run.matvecs < max_matvecs and not run.exhausted:
        run.step()
        if not (_is_checkpoint(run.matvecs) or run.exhausted
                or run.matvecs >= max_matvecs):
            continue
        theta, S, _ = run.ritz()
        c = S[0, :]
        order = np.argsort(np.abs(theta))
        q0 = estimate_qsvt_queries(theta, c, np.array([], dtype=int), lam_min, alpha, eps)
        new = []
        while k_next <= k_max and k_next < len(theta) and run.converged(k_next + 1, tol):
            d = run.deflation(bh, k_next, tol)
            new.append((k_next, estimate_qsvt_queries(theta, c, order[:k_next], d.next_ritz,
                                                      alpha, eps), d))
            k_next += 1
        cps.append((run.matvecs, q0, new))
    return LanczosTrajectory(checkpoints=cps,
                             est_q0_initial=float(inverse_poly(alpha / lam_min, eps).degree))


def pick_budget(traj: LanczosTrajectory, w: float) -> BudgetChoice:
    """Replay the budget search for cost ratio ``w``.

    ``cost(k) = est_QSVT_queries(kappa_eff(k)) + w * matvecs(k)``, with
    ``matvecs(k)`` the step at which candidate ``k`` converged; ``k = 0``
    (plain QSVT, no matvecs) is always a candidate, so the choice never has
    a higher estimated cost than plain QSVT. The search stops once
    ``w * matvecs`` alone reaches the best candidate's total, since no later
    candidate can then be cheaper. (A "stop when cost rises" rule would give
    up inside clusters of small eigenvalues, where only removing the whole
    cluster pays off.) All matvecs up to the stop are charged.
    """
    est_q = {0: traj.est_q0_initial}
    est_t = {0: traj.est_q0_initial}
    defl = {}
    spent = 0
    for m, q0, new in traj.checkpoints:
        if w * spent >= min(est_t.values()):
            break
        spent = m
        if q0:
            est_q[0] = est_t[0] = q0
        for k, q, d in new:
            est_q[k], est_t[k], defl[k] = q, q + w * m, d
    k = min(est_t, key=est_t.get)
    return BudgetChoice(k=k, deflation=defl.get(k), matvecs=spent,
                        est_quantum=est_q, est_total=est_t)


def choose_budget(A, b: np.ndarray, eps: float, w: float,
                  k_max: int | None = None, tol: float | None = None,
                  alpha: float | None = None, lam_min: float | None = None,
                  max_matvecs: int | None = None) -> BudgetChoice:
    """Pick the deflation budget ``k`` minimizing estimated total cost.

    See ``pick_budget`` for the rule. Like every QLSA we assume
    ``alpha = ||A||`` and ``lambda_min`` are known (computed here if not
    given); deflated spectrum edges come from Lanczos.
    """
    if alpha is None or lam_min is None:
        lam = np.abs(np.linalg.eigvalsh(np.asarray(A)))
        alpha = float(lam.max()) if alpha is None else alpha
        lam_min = float(lam.min()) if lam_min is None else lam_min
    traj = lanczos_trajectory(A, b, eps, alpha, lam_min, k_max, tol, max_matvecs)
    return pick_budget(traj, w)


def lanczos_deflate(A, b: np.ndarray, eps: float, w: float, alpha: float,
                    lam_min: float, budget: int | None = None, k_max: int | None = None,
                    tol: float | None = None, max_matvecs: int | None = None):
    """Classical half of ``solve_deflated_qsvt``.

    Returns ``(k, z, b_res, edge, matvecs, est)`` where ``edge`` is the lower
    end of the spectrum the QSVT polynomial must cover and ``matvecs``
    includes the extra product for ``b_res = b - A z``.
    """
    from ml.baselines import lanczos_deflation

    tol = 0.1 * eps if tol is None else tol
    if budget is None:
        ch = choose_budget(A, b, eps, w, k_max, tol, alpha, lam_min, max_matvecs)
        return (*deflate_from_choice(A, b, ch, lam_min), {"est_total": ch.est_total})
    if budget == 0:
        return 0, np.zeros_like(b), b, lam_min, 0, {}
    d = lanczos_deflation(A, b / np.linalg.norm(b), budget, tol, max_matvecs)
    ch = BudgetChoice(k=budget, deflation=d, matvecs=d.matvecs, est_quantum={}, est_total={})
    return (*deflate_from_choice(A, b, ch, lam_min), {})


def deflate_from_choice(A, b: np.ndarray, ch: BudgetChoice, lam_min: float):
    """``(k, z, b_res, edge, matvecs)`` for a budget choice (Lanczos started from ``b_hat``)."""
    if ch.k == 0:
        return 0, np.zeros_like(b), b, lam_min, ch.matvecs
    z = ch.deflation.z * np.linalg.norm(b)
    return ch.k, z, b - A @ z, ch.deflation.next_ritz, ch.matvecs + 1


def solve_deflated_qsvt(
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-3,
    w: float = 0.1,
    budget: int | None = None,
    k_max: int | None = None,
    tol: float | None = None,
    max_matvecs: int | None = None,
) -> HybridResult:
    """Adaptive Lanczos deflation + emulated QSVT inversion of the residual.

    ``x = z + ||b_res|| * y`` with ``z`` from Lanczos Ritz pairs (no exact
    eigendecomposition), ``b_res = b - A z`` (one extra matvec, not
    projected), and ``y`` from QSVT with a polynomial accurate on
    ``[lambda_{k+1}, alpha]``. Residual leakage onto removed modes is damped
    by the bounded polynomial. ``budget=None`` chooses ``k`` adaptively.
    ``max_matvecs`` caps Lanczos; keeping it well below ``N`` avoids the
    small-``N`` artifact where the Krylov space is exhausted and deflation
    degenerates into a full classical solve.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    check_hermitian(A)
    lam = np.abs(np.linalg.eigvalsh(A))
    alpha, lam_min = float(lam.max()), float(lam.min())
    x_true = np.linalg.solve(A, b)

    k, z, b_res, edge, matvecs, est = lanczos_deflate(
        A, b, eps, w, alpha, lam_min, budget, k_max, tol, max_matvecs)
    nr = float(np.linalg.norm(b_res))
    metrics: dict = {"N": int(A.shape[0]), "budget": k, "matvecs": matvecs, "eps": eps, "w": w,
                     "kappa": alpha / lam_min, "kappa_eff": alpha / edge,
                     "b_res_rel": nr / float(np.linalg.norm(b)), **est}
    y = np.zeros_like(b)
    if nr <= 1e-12 * np.linalg.norm(b):
        metrics.update(hhl_skipped=True, degree=0, p_success=1.0, queries_aa=0.0,
                       queries_per_run=0)
    else:
        out = solve_qsvt(A, b_res, edge, eps=eps, alpha=alpha)
        y = out.y
        metrics.update(hhl_skipped=False, degree=out.degree, p_success=out.p_success,
                       queries_per_run=out.queries_per_run, queries_aa=out.queries_aa,
                       aa_rounds=out.aa_rounds)
    x = z + nr * y
    metrics["total_cost"] = metrics["queries_aa"] + w * matvecs
    metrics["fidelity"] = fidelity(x, x_true)
    metrics["rel_error"] = float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
    return HybridResult(x=x, z=z, y=y, b_res_norm=nr, metrics=metrics)

"""Backward-compatible wrapper for the original ``hhl_algorithm.py`` API.

The original functions are kept with the same names, signatures and return
tuples, but now delegate to the ``hhl`` package:

* eigen-based residualization (Hermitian ``A``; non-Hermitian ``A`` is
  embedded as ``[[0, A], [A^T, 0]]`` by the pipeline);
* a real Qiskit HHL circuit tuned to the remaining spectrum (reduced mode),
  instead of the previous ``np.linalg.inv(A)`` placeholder for ``y``.
"""

from __future__ import annotations

import numpy as np

from hhl.pipeline import solve_hybrid
from hhl.residualization import budgeted_affine_residualization as _residualize


def budgeted_affine_residualization(A, b, budget=1):
    """Remove the ``budget`` highest-energy eigenmodes of Hermitian ``A``.

    Returns ``(z, b_res, top_modes)``: the explanation vector, the residual
    right-hand side projected onto the kept modes, and the indices (in
    ascending-eigenvalue order of ``np.linalg.eigh``) of the removed modes.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    r = _residualize(A, b, budget, criterion="energy")
    print(f"[residualization] removed modes {r.removed.tolist()} "
          f"(eigenvalues {np.round(r.eigvals[r.removed], 6).tolist()}); "
          f"kappa {r.kappa:.4g} -> kappa_eff {r.kappa_eff:.4g}; "
          f"||b_res|| = {np.linalg.norm(r.b_res):.4g}")
    return r.z, r.b_res, r.removed


def run_hybrid_quantum_linear_solver(A, b, budget=1):
    """Solve ``A x = b`` as ``x = z + ||b_res|| * y`` with HHL for ``y``.

    Returns ``(x_hybrid, z, y)``. ``y`` is the HHL solution for the normalized
    residual ``b_res / ||b_res||`` (zero if the residual vanishes, in which
    case HHL is skipped).
    """
    out = solve_hybrid(np.asarray(A, dtype=float), np.asarray(b, dtype=float),
                       budget=budget, criterion="energy", mode="reduced")
    m = out.metrics
    if m.get("hhl_skipped"):
        print("[hybrid] b_res = 0: the removed modes explain b completely, HHL skipped")
    else:
        print(f"[hybrid] HHL ran with {m['n_clock']} clock qubits "
              f"(kappa_eff {m['kappa_eff']:.4g}), P(ancilla=1) = {m['p_success']:.4f}, "
              f"HHL fidelity = {m['hhl_fidelity']:.6f}")
    return out.x, out.z, out.y


def _report(A, b, budget):
    x, z, y = run_hybrid_quantum_linear_solver(A, b, budget)
    x_exact = np.linalg.solve(A, b)
    err = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    print(f"  z        = {z}")
    print(f"  y        = {y}")
    print(f"  x_hybrid = {x}")
    print(f"  x_exact  = {x_exact}")
    print(f"  relative error = {err:.3e}")


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    A = np.diag([1e-4, 5e-3])

    print("=== Original example: b = [0, 100], budget = 1 ===")
    print("b lies entirely in one eigenmode, so removing it leaves b_res = 0.")
    budgeted_affine_residualization(A, np.array([0.0, 100.0]), budget=1)
    _report(A, np.array([0.0, 100.0]), budget=1)

    print("\n=== Multi-mode example: b = 100 * [1, 1] / sqrt(2), budget = 1 ===")
    b = 100.0 * np.array([1.0, 1.0]) / np.sqrt(2.0)
    budgeted_affine_residualization(A, b, budget=1)
    _report(A, b, budget=1)

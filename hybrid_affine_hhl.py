"""Demo: standard HHL vs. budgeted affine-residualized (hybrid) HHL.

The right-hand side spans every eigenmode, so after removing the hardest
mode(s) the residual is non-zero and the HHL circuit actually runs.

Run:  python hybrid_affine_hhl.py
"""

from __future__ import annotations

import numpy as np

from hhl import solve_hybrid, solve_standard


def build_system(eigs, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(eigs) @ Q.T
    # equal weight on every eigenmode -> b_res != 0 after deflation
    b = Q @ (np.ones(n) / np.sqrt(n))
    return A, b


def fmt(m: dict) -> str:
    keys = ["kappa_target", "n_clock", "C", "p_success", "hhl_fidelity", "fidelity",
            "rel_error", "depth", "cx"]
    parts = []
    for k in keys:
        v = m.get(k)
        if v is None:
            continue
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)


def run_case(name, eigs, budgets, criterion="energy", seed=0):
    A, b = build_system(np.asarray(eigs, dtype=float), seed)
    print(f"\n=== {name}: N={len(eigs)}, eigenvalues={np.round(eigs, 5).tolist()}")
    std = solve_standard(A, b, compute_resources=True)
    print(f"standard HHL   kappa={std.metrics['kappa']:.4g}")
    print("   ", fmt(std.metrics))
    for k in budgets:
        hyb = solve_hybrid(A, b, budget=k, criterion=criterion, compute_resources=True)
        m = hyb.metrics
        print(f"hybrid k={k} ({criterion})  removed eigvals={np.round(m['removed_eigvals'], 5).tolist()}"
              f"  kappa_eff={m['kappa_eff']:.4g}  ||b_res||/||b||={m['b_res_rel']:.3g}")
        print("   ", fmt(m))
        ratio = m["p_success"] / std.metrics["p_success"]
        print(f"    P_success gain vs standard: x{ratio:.3g}")


def tradeoff_case(eigs, rel_errors=(0.0, 0.01, 0.05, 0.2), seed=0):
    """Perturb z along the removed (hard) mode and compare circuits.

    Reduced circuit: residual projected onto kept modes, QPE tuned to
    kappa_eff -> the hard-mode error delta survives in x.
    Full circuit: raw residual, QPE tuned to the full kappa -> HHL can
    correct delta, but at full-kappa cost (clock qubits, depth, P_success).
    """
    A, b = build_system(np.asarray(eigs, dtype=float), seed)
    lam, U = np.linalg.eigh(A)
    x = np.linalg.solve(A, b)
    i_hard = int(np.argmin(np.abs(lam)))
    z_exact = U[:, i_hard] * (U[:, i_hard] @ b) / lam[i_hard]
    print(f"\n=== Trade-off: error delta on hard mode, N={len(eigs)}, "
          f"kappa={np.abs(lam).max() / np.abs(lam).min():.3g}")
    for eps in rel_errors:
        z_hat = z_exact + eps * np.linalg.norm(z_exact) * U[:, i_hard]
        prov = lambda A_, b_, k, z_hat=z_hat: z_hat
        row = [f"||delta||/||z||={eps:<5}"]
        for mode in ("reduced", "full"):
            m = solve_hybrid(A, b, prov, budget=1, criterion="smallest", mode=mode,
                             compute_resources=True).metrics
            err = m["rel_error"]
            row.append(f"{mode}: rel_err={err:.3g} n_clock={m.get('n_clock', '-')} "
                       f"cx={m.get('cx', '-')} P={m['p_success']:.3g}")
        print("   " + " | ".join(row))
    print(f"    (||z||/||x|| = {np.linalg.norm(z_exact) / np.linalg.norm(x):.3g})")


def main():
    np.set_printoptions(precision=4, suppress=True)
    run_case("N=2, kappa=100", [0.01, 1.0], budgets=[1])
    run_case("N=4, kappa=1000", [1e-3, 0.05, 0.4, 1.0], budgets=[1, 2])
    run_case("N=8, kappa=200", np.logspace(np.log10(5e-3), 0, 8), budgets=[1, 3],
             criterion="smallest", seed=3)
    tradeoff_case([1e-2, 0.2, 0.5, 1.0])


if __name__ == "__main__":
    main()

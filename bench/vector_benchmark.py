"""Classical sanity benchmark for budgeted affine residualization.

Runs the residualize -> solve -> recombine pipeline (with ``np.linalg.solve``
standing in for HHL) against a family of right-hand-side generators, so the
method is exercised on vectors that are spread over all modes, concentrated on
the hard mode, concentrated on the easy mode, sparse, badly scaled, etc.

Usage:
    python bench/vector_benchmark.py --samples 10000 --N 2 4 8 16 32 --budget 1
"""

from __future__ import annotations

import argparse
import time

import numpy as np


# ==========================================
# Matrix generator
# ==========================================
def generate_symmetric_matrix(n=2, rng=None):
    """Generate a random symmetric positive-definite matrix for HHL."""
    rng = rng or np.random.default_rng()
    M = rng.standard_normal((n, n))
    # Make it symmetric and positive definite
    A = M @ M.T + np.eye(n) * 1e-3
    return A


# ==========================================
# Vector (right-hand side) generators
# Signature: gen(A, rng) -> b
# ==========================================
def _eig(A):
    """Eigenpairs of symmetric A, ascending by |eigenvalue|."""
    lam, U = np.linalg.eigh(A)
    order = np.argsort(np.abs(lam))
    return lam[order], U[:, order]


def gaussian_vector(A, rng):
    """i.i.d. standard normal entries (unnormalized)."""
    return rng.standard_normal(A.shape[0])


def unit_sphere_vector(A, rng):
    """Uniformly random direction on the unit sphere (HHL-style |b>)."""
    b = rng.standard_normal(A.shape[0])
    return b / np.linalg.norm(b)


def uniform_vector(A, rng):
    """Entries uniform in [-1, 1]."""
    return rng.uniform(-1.0, 1.0, A.shape[0])


def basis_vector(A, rng):
    """A random computational-basis state |i>."""
    b = np.zeros(A.shape[0])
    b[rng.integers(A.shape[0])] = 1.0
    return b


def sparse_vector(A, rng, density=0.25):
    """Gaussian vector with most entries zeroed (at least one non-zero)."""
    n = A.shape[0]
    b = rng.standard_normal(n)
    mask = rng.random(n) < density
    mask[rng.integers(n)] = True
    return b * mask


def hard_mode_vector(A, rng):
    """Exactly the smallest-eigenvalue eigenvector: residual should vanish."""
    _, U = _eig(A)
    return U[:, 0] * rng.choice([-1.0, 1.0])


def easy_mode_vector(A, rng):
    """Exactly the largest-eigenvalue eigenvector: nothing hard to remove."""
    _, U = _eig(A)
    return U[:, -1] * rng.choice([-1.0, 1.0])


def noisy_hard_mode_vector(A, rng, noise=0.1):
    """Mostly the hard mode plus a small isotropic perturbation."""
    _, U = _eig(A)
    b = U[:, 0] + noise * rng.standard_normal(A.shape[0])
    return b / np.linalg.norm(b)


def equal_energy_vector(A, rng):
    """Equal |weight| on every eigenmode with random signs."""
    _, U = _eig(A)
    n = A.shape[0]
    c = rng.choice([-1.0, 1.0], n) / np.sqrt(n)
    return U @ c


def ill_scaled_vector(A, rng, decades=6):
    """Entries with log-uniform magnitudes spanning ``decades`` orders."""
    n = A.shape[0]
    mags = 10.0 ** rng.uniform(-decades / 2, decades / 2, n)
    return mags * rng.choice([-1.0, 1.0], n)


def known_solution_vector(A, rng):
    """b = A x_true for a random x_true (well-scaled solution, smooth b)."""
    return A @ rng.standard_normal(A.shape[0])


VECTOR_GENERATORS = {
    "gaussian": gaussian_vector,
    "unit_sphere": unit_sphere_vector,
    "uniform": uniform_vector,
    "basis": basis_vector,
    "sparse": sparse_vector,
    "hard_mode": hard_mode_vector,
    "easy_mode": easy_mode_vector,
    "noisy_hard_mode": noisy_hard_mode_vector,
    "equal_energy": equal_energy_vector,
    "ill_scaled": ill_scaled_vector,
    "known_solution": known_solution_vector,
}


# ==========================================
# Residualization
# ==========================================
def budgeted_affine_residualization(A, b, budget=1):
    """The new classical pre-processing step."""
    U, S, Vh = np.linalg.svd(A)
    beta = U.T @ b
    energy = np.abs(beta / (S + 1e-12))

    top_modes = np.argsort(energy)[-budget:]

    z = np.zeros_like(b, dtype=np.float64)
    b_res = np.copy(b).astype(np.float64)

    for idx in top_modes:
        z += (beta[idx] / S[idx]) * Vh[idx]
        b_res -= beta[idx] * U[:, idx]

    # Calculate effective condition number (remaining modes)
    remaining_modes = [s for i, s in enumerate(S) if i not in top_modes]
    k_eff = max(remaining_modes) / (min(remaining_modes) + 1e-12) if remaining_modes else 1.0
    k_original = max(S) / min(S)

    return z, b_res, k_original, k_eff


# ==========================================
# Benchmark
# ==========================================
def run_benchmark(vector_gen, num_samples=1000, matrix_size=2, budget=1, rng=None):
    rng = rng or np.random.default_rng()

    time_old_total = 0.0
    time_new_total = 0.0
    k_original = []
    k_eff = []
    b_res_rel = []
    hhl_skipped = 0
    rel_errors = []

    for _ in range(num_samples):
        A = generate_symmetric_matrix(matrix_size, rng)
        b = vector_gen(A, rng)

        # OLD APPROACH (Standard Setup + Solve)
        start_time = time.perf_counter()
        # In a real scenario, this is where standard Qiskit HHL runs
        x_old = np.linalg.solve(A, b)
        time_old_total += time.perf_counter() - start_time

        # NEW APPROACH (Hybrid Affine Residualization)
        start_time = time.perf_counter()

        # 1. Classical Pre-processing
        z, b_res, k_orig, k_e = budgeted_affine_residualization(A, b, budget=budget)

        # 2. Quantum Execution (Simulated)
        norm_b_res = np.linalg.norm(b_res)
        if norm_b_res < 1e-8 * max(np.linalg.norm(b), 1e-300):
            y = np.zeros_like(b, dtype=np.float64)
            hhl_skipped += 1
        else:
            normalized_b_res = b_res / norm_b_res
            # In a real scenario, this is where Qiskit HHL runs on b_res
            normalized_y = np.linalg.solve(A, normalized_b_res)
            y = normalized_y * norm_b_res

        # 3. Recombination
        x_new = z + y

        time_new_total += time.perf_counter() - start_time

        k_original.append(k_orig)
        k_eff.append(k_e)
        b_res_rel.append(norm_b_res / np.linalg.norm(b))
        rel_errors.append(np.linalg.norm(x_new - x_old) / np.linalg.norm(x_old))

    max_err = max(rel_errors)
    assert max_err < 1e-6, f"Mismatch in solutions! max rel error = {max_err:.3e}"

    return dict(
        time_old=time_old_total,
        time_new=time_new_total,
        k=float(np.median(k_original)),
        k_eff=float(np.median(k_eff)),
        b_res_rel=float(np.mean(b_res_rel)),
        hhl_skipped=hhl_skipped / num_samples,
        max_rel_err=max_err,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=10_000)
    p.add_argument("--N", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--budget", type=int, default=1)
    p.add_argument("--vectors", nargs="+", default=list(VECTOR_GENERATORS),
                   choices=list(VECTOR_GENERATORS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    for N in args.N:
        budget = min(args.budget, N)
        print(f"\n{'=' * 96}")
        print(f"N={N}  budget={budget}  samples={args.samples}  "
              f"(kappa / kappa_eff are medians; ||b_res||/||b|| is a mean)")
        print("=" * 96)
        print(f"{'vector':>16} {'kappa':>10} {'kappa_eff':>10} {'reduction':>10} "
              f"{'|b_res|/|b|':>12} {'HHL skip':>9} {'max rel err':>12} {'overhead':>10}")
        for name in args.vectors:
            rng = np.random.default_rng(args.seed)
            r = run_benchmark(VECTOR_GENERATORS[name], args.samples, N, budget, rng)
            reduction = (1 - r["k_eff"] / r["k"]) * 100
            overhead = r["time_new"] - r["time_old"]
            print(f"{name:>16} {r['k']:>10.3g} {r['k_eff']:>10.3g} {reduction:>9.1f}% "
                  f"{r['b_res_rel']:>12.3g} {r['hhl_skipped']:>8.1%} "
                  f"{r['max_rel_err']:>12.2e} {overhead:>9.4f}s")


if __name__ == "__main__":
    main()

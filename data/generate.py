"""Synthetic dataset of Hermitian systems with controlled condition number.

Each sample: ``A = Q diag(s) Q^T`` with Haar-random orthogonal ``Q`` and a
log-uniform spectrum ``s`` in ``[1/kappa, 1]`` (endpoints included, so the
condition number is exactly ``kappa``), ``kappa`` log-uniform in
``[kappa_min, kappa_max]``. ``b = Q c`` with Gaussian ``c`` spreads energy over
all modes. Targets are the exact deflation vector ``z`` for a given budget and
criterion plus the projector ``P_hard`` onto the removed modes (used by the
physics loss).

Usage:
    python data/generate.py --n-samples 10000 --N 2 4 8 --budget 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhl.residualization import budgeted_affine_residualization  # noqa: E402


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    return Q * np.sign(np.diag(R))


def sample_spectrum(n: int, kappa: float, rng: np.random.Generator,
                    signed: bool = False) -> np.ndarray:
    if n == 1:
        s = np.array([1.0])
    else:
        inner = rng.uniform(-np.log10(kappa), 0.0, size=n - 2)
        s = 10.0 ** np.concatenate([[-np.log10(kappa), 0.0], inner])
    if signed:
        s = s * rng.choice([-1.0, 1.0], size=n)
    return np.sort(s)


def generate(
    n_samples: int,
    N: int,
    budget: int = 1,
    criterion: str = "energy",
    kappa_min: float = 10.0,
    kappa_max: float = 1e4,
    signed: bool = False,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = np.empty((n_samples, N, N))
    b = np.empty((n_samples, N))
    x = np.empty((n_samples, N))
    z = np.empty((n_samples, N))
    P_hard = np.empty((n_samples, N, N))
    eigvals = np.empty((n_samples, N))
    kappa = np.empty(n_samples)
    kappa_eff = np.empty(n_samples)
    margin = np.empty(n_samples)
    for i in range(n_samples):
        kap = 10 ** rng.uniform(np.log10(kappa_min), np.log10(kappa_max))
        s = sample_spectrum(N, kap, rng, signed)
        Q = haar_orthogonal(N, rng)
        Ai = (Q * s) @ Q.T
        Ai = 0.5 * (Ai + Ai.T)
        bi = Q @ rng.normal(size=N)
        bi /= np.linalg.norm(bi)
        r = budgeted_affine_residualization(Ai, bi, budget, criterion)
        Ur = r.eigvecs[:, r.removed]
        A[i], b[i] = Ai, bi
        x[i] = np.linalg.solve(Ai, bi)
        z[i] = r.z
        P_hard[i] = Ur @ Ur.T
        eigvals[i] = r.eigvals
        kappa[i], kappa_eff[i] = r.kappa, r.kappa_eff
        # ratio between the last selected and first unselected mode energy;
        # values near 1 mean the hard-mode selection is close to a tie, where
        # z jumps discontinuously.
        e = np.sort(r.energies)[::-1]
        k = len(r.removed)
        margin[i] = e[k - 1] / e[k] if 0 < k < N else np.inf
    return dict(A=A, b=b, x=x, z=z, P_hard=P_hard, eigvals=eigvals, kappa=kappa,
                kappa_eff=kappa_eff, selection_margin=margin,
                budget=np.array(budget), criterion=np.array(criterion))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--N", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--budget", type=int, default=1)
    p.add_argument("--criterion", choices=["energy", "smallest"], default="energy")
    p.add_argument("--kappa-min", type=float, default=10.0)
    p.add_argument("--kappa-max", type=float, default=1e4)
    p.add_argument("--signed", action="store_true", help="random eigenvalue signs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for N in args.N:
        d = generate(args.n_samples, N, args.budget, args.criterion, args.kappa_min,
                     args.kappa_max, args.signed, args.seed + N)
        path = args.out / f"dataset_N{N}_k{args.budget}_{args.criterion}.npz"
        np.savez_compressed(path, **d)
        near_tie = float(np.mean(d["selection_margin"] < 1.1))
        print(f"N={N}: {args.n_samples} samples -> {path.name}  "
              f"median kappa={np.median(d['kappa']):.3g}  "
              f"median kappa_eff={np.median(d['kappa_eff']):.3g}  "
              f"near-tie selections (<1.1x)={near_tie:.1%}")


if __name__ == "__main__":
    main()

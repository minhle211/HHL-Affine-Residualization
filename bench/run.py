"""Benchmark standard HHL vs. hybrid (affine-residualized) HHL.

Experiments (all exact statevector simulation, metrics per run):

* ``grid``     : N x kappa x budget, exact eigen-deflation (budget 0 = standard HHL).
* ``zsource``  : budget 1, z from {exact, lanczos, inverse_iter, rand_svd, mlp},
                 reduced (kappa_eff) vs full-kappa circuit.
* ``tradeoff`` : controlled error delta on the removed mode, reduced vs full.

Metrics: clock qubits, transpiled depth and CX count ({u, cx} basis),
P(ancilla=1), solution fidelity and relative error. Wall-clock time of the
simulator is *not* reported as a performance metric.

Usage:
    python bench/run.py --N 2 4 8 --kappa 10 100 1000 --trials 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate import haar_orthogonal, sample_spectrum  # noqa: E402
from hhl import solve_hybrid  # noqa: E402
from ml.baselines import BASELINES  # noqa: E402

FIELDS = ["experiment", "N", "kappa_nominal", "trial", "z_source", "mode", "budget",
          "criterion", "delta_rel", "kappa", "kappa_eff", "kappa_target", "n_clock",
          "clock_capped", "depth", "cx", "num_qubits", "p_success", "p_success_clock0",
          "hhl_fidelity", "fidelity", "rel_error", "z_rel_err", "b_res_rel", "hhl_skipped"]


def make_system(N: int, kappa: float, rng: np.random.Generator):
    s = sample_spectrum(N, kappa, rng)
    Q = haar_orthogonal(N, rng)
    A = (Q * s) @ Q.T
    A = 0.5 * (A + A.T)
    b = Q @ rng.normal(size=N)
    return A, b / np.linalg.norm(b)


def record(exp, N, kappa, trial, src, m, extra=None):
    row = {k: m.get(k) for k in FIELDS if k in m}
    row.update(experiment=exp, N=N, kappa_nominal=kappa, trial=trial, z_source=src)
    if extra:
        row.update(extra)
    return row


def load_mlp(N: int, ckpt_dir: Path):
    path = ckpt_dir / f"mlp_N{N}.pt"
    if not path.exists():
        return None
    from ml.model import MLPZProvider

    return MLPZProvider.from_path(path)


def run(Ns, kappas, budgets, trials, criterion, experiments, ckpt_dir, resources, seed,
        out_csv: Path, max_clock: int):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    kw = dict(criterion=criterion, compute_resources=resources, max_clock=max_clock)
    for N in Ns:
        mlp = load_mlp(N, ckpt_dir) if "zsource" in experiments else None
        for kappa in kappas:
            for trial in range(trials):
                A, b = make_system(N, kappa, rng)
                if "grid" in experiments:
                    for k in budgets:
                        if k > N - 1:
                            continue
                        m = solve_hybrid(A, b, budget=k, **kw).metrics
                        rows.append(record("grid", N, kappa, trial,
                                           "none" if k == 0 else "exact", m))
                if "zsource" in experiments:
                    from hhl.residualization import budgeted_affine_residualization

                    z_ex = budgeted_affine_residualization(A, b, 1, criterion).z
                    sources = {"exact": None, **BASELINES}
                    if mlp is not None:
                        sources["mlp"] = mlp
                    for name, prov in sources.items():
                        z_hat = z_ex if prov is None else prov(A, b, 1)
                        zerr = float(np.linalg.norm(z_hat - z_ex) / np.linalg.norm(z_ex))
                        for mode in ("reduced", "full"):
                            m = solve_hybrid(A, b, prov, budget=1, mode=mode, **kw).metrics
                            rows.append(record("zsource", N, kappa, trial, name, m,
                                               {"z_rel_err": zerr}))
                if "tradeoff" in experiments and trial == 0:
                    lam, U = np.linalg.eigh(A)
                    i = int(np.argmin(np.abs(lam)))
                    z_ex = U[:, i] * (U[:, i] @ b) / lam[i]
                    for eps in (0.0, 0.01, 0.03, 0.1, 0.3):
                        z_hat = z_ex + eps * np.linalg.norm(z_ex) * U[:, i]
                        prov = lambda A_, b_, k, z_hat=z_hat: z_hat
                        for mode in ("reduced", "full"):
                            m = solve_hybrid(A, b, prov, budget=1, mode=mode,
                                             **{**kw, "criterion": "smallest"}).metrics
                            rows.append(record("tradeoff", N, kappa, trial, "perturbed",
                                               m, {"delta_rel": eps, "z_rel_err": eps}))
                print(f"  N={N} kappa={kappa:g} trial={trial}  rows={len(rows)}  "
                      f"elapsed={time.time() - t0:.1f}s", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"wrote {len(rows)} rows to {out_csv}")
    return rows


def summarize(rows):
    print("\nGrid summary (mean over trials):")
    print(f"{'N':>2} {'kappa':>6} {'k':>2} {'kappa_eff':>9} {'clock':>5} {'depth':>7} "
          f"{'cx':>6} {'P_succ':>7} {'fid':>7} {'rel_err':>8}")
    grid = [r for r in rows if r["experiment"] == "grid"]
    keys = sorted({(r["N"], r["kappa_nominal"], r["budget"]) for r in grid})
    for N, kap, k in keys:
        g = [r for r in grid if (r["N"], r["kappa_nominal"], r["budget"]) == (N, kap, k)]

        def mean(f):
            v = [r[f] for r in g if r.get(f) is not None]
            return float(np.mean(v)) if v else float("nan")

        print(f"{N:>2} {kap:>6g} {k:>2} {mean('kappa_eff'):>9.3g} {mean('n_clock'):>5.1f} "
              f"{mean('depth'):>7.0f} {mean('cx'):>6.0f} {mean('p_success'):>7.3f} "
              f"{mean('fidelity'):>7.4f} {mean('rel_error'):>8.2e}")

    zs = [r for r in rows if r["experiment"] == "zsource"]
    if zs:
        print("\nz-source summary (median over systems): z_rel_err | rel_error reduced | full")
        for N in sorted({r["N"] for r in zs}):
            for src in dict.fromkeys(r["z_source"] for r in zs):
                sel = [r for r in zs if r["N"] == N and r["z_source"] == src]
                if not sel:
                    continue
                red = [r["rel_error"] for r in sel if r["mode"] == "reduced"]
                full = [r["rel_error"] for r in sel if r["mode"] == "full"]
                print(f"  N={N} {src:>12}: {np.median([r['z_rel_err'] for r in sel]):9.2e} | "
                      f"{np.median(red):9.2e} | {np.median(full):9.2e}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--kappa", type=float, nargs="+", default=[10, 100, 1000])
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--criterion", choices=["energy", "smallest"], default="energy")
    p.add_argument("--experiments", nargs="+", default=["grid", "zsource", "tradeoff"],
                   choices=["grid", "zsource", "tradeoff"])
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "ml" / "checkpoints")
    p.add_argument("--no-resources", action="store_true", help="skip transpilation")
    p.add_argument("--max-clock", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "bench.csv")
    args = p.parse_args(argv)
    rows = run(args.N, args.kappa, args.budgets, args.trials, args.criterion,
               args.experiments, args.ckpt_dir, not args.no_resources, args.seed,
               args.out, args.max_clock)
    summarize(rows)


if __name__ == "__main__":
    main()

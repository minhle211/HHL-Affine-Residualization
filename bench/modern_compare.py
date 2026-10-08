"""Deflated QSVT vs. modern quantum linear solvers (query-cost benchmark).

Every system ``A x = b`` (real SPD, ``b`` Gaussian) is solved or costed by:

* ``hhl_honest``   : textbook HHL, QPE clock sized for ``kappa / eps``; queries
                     include Hamiltonian simulation of every ``U^(2^j)`` and
                     amplitude amplification (analytic, ideal ``p_success``).
* ``qsvt``         : emulated QSVT inversion on the full spectrum (+ AA).
* ``dalzell``      : Dalzell 2024 optimal QLSA bound (arXiv:2406.12086).
* ``jennings``     : Jennings et al. 2024 bound (arXiv:2305.11352v3).
* ``hybrid_exact`` : exact eigen-deflation, best ``k <= k_max`` by quantum
                     queries (oracle; classical cost not counted -> quantum-only).
* ``hybrid_adaptive`` (per ``w``): Lanczos deflation with adaptive ``k`` +
                     emulated QSVT; all Lanczos matvecs are charged.
* ``hybrid_dalzell``  (per ``w``): Dalzell's bound at the adaptive
                     ``kappa_eff`` + the same matvecs (deflation composes with
                     the optimal solver).
* ``cg``           : classical conjugate gradient, matvecs until the true
                     relative error is ``<= eps`` (oracle stopping; at small N
                     this is capped by N because CG is exact after N steps).
* ``cg_bound``     : the large-N CG iteration bound ``sqrt(kappa)/2 ln(2/eps)``.

Deflation is restricted to the low-rank regime: ``k <= min(k_cap, k_frac * N)``
and at most ``mv_frac * N`` Lanczos matvecs, so the Krylov space is never
exhausted (which would turn deflation into a full classical solve).

Systems are emulated spectrally: Lanczos, CG and QSVT on ``Q diag(lam) Q^T``
from Gaussian ``b`` are equivalent to the same methods on ``diag(lam)`` from
Gaussian ``c = Q^T b``, which makes N up to ~1e3+ cheap. For
``degree * N > 2e7`` the QSVT output uses the polynomial's smooth target
``g`` (``|P - g| <= eps / (2M)``), so reported errors may differ from the
exact polynomial's by at most ~``eps/2``.

Cost: ``total = quantum_queries + w * classical_matvecs`` for each ``w``.
Raw rows go to ``results/modern_compare_raw.csv.gz``; medians per
configuration (and per-system speedups) to ``results/modern_compare.csv``.

Usage:
    python bench/modern_compare.py --trials 1000
    python bench/modern_compare.py --N 8 32 --kappa 1e3 --trials 50 --families wishart
"""

from __future__ import annotations

import os

# one BLAS thread per worker process; the sweep parallelizes over systems
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import gzip
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.matrices import MATRIX_FAMILIES, spectrum  # noqa: E402
from hhl.circuit import choose_hhl_params  # noqa: E402
from hhl.cost import aa_cost, hhl_cost_metrics  # noqa: E402
from hhl.pipeline import deflate_from_choice, lanczos_trajectory, pick_budget  # noqa: E402
from hhl.qsvt import dalzell_queries, inverse_poly, jennings_queries  # noqa: E402
from ml.baselines import DiagonalOperator  # noqa: E402

RAW_FIELDS = ["family", "N", "kappa_nominal", "eps", "trial", "method", "w", "kappa",
              "kappa_eff", "budget", "degree", "queries", "matvecs", "rel_error", "p_success"]
BASE_METHODS = ["hhl_honest", "qsvt", "dalzell", "jennings", "hybrid_exact", "cg", "cg_bound"]
W_METHODS = ["hybrid_adaptive", "hybrid_dalzell"]


def conjugate_gradient(A, b, x_true, eps, max_iter):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    nx = np.linalg.norm(x_true)
    for it in range(1, max_iter + 1):
        Ap = A @ p
        a = rr / (p @ Ap)
        x += a * p
        r -= a * Ap
        if np.linalg.norm(x - x_true) <= eps * nx:
            return it
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
    return max_iter


EXACT_EVAL_LIMIT = 2e7  # degree * N above which P is replaced by its smooth target g


def qsvt_eig(lam, c, edge, alpha, eps, keep=None, exact=True, memo=None):
    """Emulated QSVT on eigen-coefficients ``c`` (optionally restricted to ``keep``).

    ``exact=False`` uses the polynomial's smooth target ``g`` (``|P - g| <=
    eps / (2M)`` on ``[-1, 1]``), which is also used automatically when
    ``degree * N`` exceeds ``EXACT_EVAL_LIMIT``; ``memo`` caches values of
    ``P(lam / alpha)`` per polynomial within one system.
    """
    poly = inverse_poly(alpha / edge, eps)
    ck = c if keep is None else np.where(keep, c, 0.0)
    nr = np.linalg.norm(ck)
    if nr <= 1e-14:
        return np.zeros_like(c), 0.0, poly.degree, 1.0
    if not exact or poly.degree * len(lam) > EXACT_EVAL_LIMIT:
        pv = poly.surrogate(lam / alpha)
    elif memo is not None:
        key = (poly.a, poly.eps)
        if key not in memo:
            memo[key] = poly(lam / alpha)
        pv = memo[key]
    else:
        pv = poly(lam / alpha)
    state = pv * ck / nr
    p = float(state @ state)
    return state * poly.M / alpha * nr, aa_cost(poly.degree, p), poly.degree, p


def run_system(args):
    family, N, kappa_nom, eps, trial, ws, seed, k_frac, mv_frac, k_cap = args
    k_max = max(1, min(k_cap, int(N * k_frac)))
    max_mv = max(k_max + 2, int(N * mv_frac))
    rng = np.random.default_rng([seed, N, int(math.log10(kappa_nom) * 100),
                                 int(-math.log10(eps) * 100), trial,
                                 list(MATRIX_FAMILIES).index(family)])
    # Spectral emulation: A = Q diag(lam) Q^T with Gaussian b is equivalent
    # (for Lanczos, CG and QSVT) to diag(lam) with Gaussian c = Q^T b.
    lam = spectrum(family, N, kappa_nom, rng)
    A = DiagonalOperator(lam)
    b = rng.standard_normal(N)
    b /= np.linalg.norm(b)
    alpha, lam_min = float(np.max(np.abs(lam))), float(np.min(np.abs(lam)))
    kappa = alpha / lam_min
    c = b
    x_true = c / lam
    nx = np.linalg.norm(x_true)
    base = dict(family=family, N=N, kappa_nominal=kappa_nom, eps=eps, trial=trial, kappa=kappa)
    rows = []

    def add(method, **kw):
        rows.append({**base, "method": method, **kw})

    # textbook HHL, honest cost
    params = choose_hhl_params(lam_min, alpha, extra_bits=math.ceil(math.log2(1 / eps)),
                               max_clock=64)
    p_hhl = float(np.sum((c * params.C / lam) ** 2))
    hc = hhl_cost_metrics(params.n_clock, params.t, alpha, eps, p_hhl)
    add("hhl_honest", kappa_eff=kappa, budget=0, degree=params.n_clock,
        queries=hc["queries_aa"], matvecs=0, p_success=p_hhl)

    # plain QSVT
    memo = {}
    y, q, d, p = qsvt_eig(lam, c, lam_min, alpha, eps, memo=memo)
    add("qsvt", kappa_eff=kappa, budget=0, degree=d, queries=q, matvecs=0,
        rel_error=np.linalg.norm(y - x_true) / nx, p_success=p)

    add("dalzell", kappa_eff=kappa, budget=0, queries=dalzell_queries(kappa, eps), matvecs=0)
    add("jennings", kappa_eff=kappa, budget=0, queries=jennings_queries(kappa, eps), matvecs=0)

    # exact deflation, oracle-best k (quantum-only)
    order = np.argsort(np.abs(lam))

    def keep_mask(k):
        keep = np.ones(N, dtype=bool)
        keep[order[:k]] = False
        return keep

    edges = [float(np.abs(lam[order[k]])) for k in range(k_max + 1)]
    k = min(range(len(edges)),
            key=lambda k: qsvt_eig(lam, c, edges[k], alpha, eps, keep_mask(k), exact=False)[1])
    keep = keep_mask(k)
    y, q, d, p = qsvt_eig(lam, c, edges[k], alpha, eps, keep, memo=memo)
    err = np.linalg.norm(np.where(keep, 0.0, c / lam) + y - x_true) / nx
    ke = alpha / edges[k]
    add("hybrid_exact", kappa_eff=ke, budget=k, degree=d, queries=q, matvecs=0,
        rel_error=err, p_success=p)

    add("cg", kappa_eff=kappa, budget=0, queries=0,
        matvecs=conjugate_gradient(A, b, x_true, eps, 20 * N), rel_error=eps)
    # standard CG iteration bound: what CG needs once N >> sqrt(kappa)
    add("cg_bound", kappa_eff=kappa, budget=0, queries=0,
        matvecs=math.ceil(0.5 * math.sqrt(kappa) * math.log(2 / eps)))

    # same classical half as hhl.pipeline.solve_deflated_qsvt: one Lanczos
    # trajectory, replayed per w (the search only truncates it)
    traj = lanczos_trajectory(A, b, eps, alpha, lam_min, k_max=k_max, max_matvecs=max_mv)
    for w in ws:
        k, z, b_res, edge, mv = deflate_from_choice(A, b, pick_budget(traj, w), lam_min)
        nr = np.linalg.norm(b_res)
        y, q, d, p = qsvt_eig(lam, b_res, edge, alpha, eps, memo=memo)
        x = z + y
        ke = alpha / edge
        add("hybrid_adaptive", w=w, kappa_eff=ke, budget=k, degree=d, queries=q, matvecs=mv,
            rel_error=np.linalg.norm(x - x_true) / nx, p_success=p, b_res_rel=nr)
        add("hybrid_dalzell", w=w, kappa_eff=ke, budget=k,
            queries=dalzell_queries(ke, eps), matvecs=mv)
    return rows


def summarize(rows, ws):
    """Median metrics and per-system speedups per (family, N, kappa, eps, method, w)."""
    by_sys = defaultdict(dict)
    for r in rows:
        key = (r["family"], r["N"], r["kappa_nominal"], r["eps"], r["trial"])
        by_sys[key][(r["method"], r.get("w"))] = r
    groups = defaultdict(lambda: defaultdict(list))
    for key, ms in by_sys.items():
        cfg = key[:4]
        for w in ws:
            ref_q = ms[("qsvt", None)]["queries"]
            ref_d = ms[("dalzell", None)]["queries"]
            for (method, mw), r in ms.items():
                if mw is not None and mw != w:
                    continue
                total = r["queries"] + w * r["matvecs"]
                g = groups[(*cfg, method, w)]
                g["kappa"].append(r["kappa"])
                g["kappa_eff"].append(r.get("kappa_eff", np.nan))
                g["budget"].append(r.get("budget", 0))
                g["queries"].append(r["queries"])
                g["matvecs"].append(r["matvecs"])
                g["total_cost"].append(total)
                g["rel_error"].append(r.get("rel_error", np.nan))
                g["speedup_vs_qsvt"].append(ref_q / total if total else np.inf)
                g["speedup_vs_dalzell"].append(ref_d / total if total else np.inf)
    out = []
    for (fam, N, kap, eps, method, w), g in sorted(groups.items(), key=lambda t: str(t[0])):
        med = lambda f: float(np.nanmedian(g[f])) if np.any(np.isfinite(g[f])) else np.nan
        out.append(dict(
            family=fam, N=N, kappa_nominal=kap, eps=eps, method=method, w=w,
            n_systems=len(g["kappa"]), kappa=med("kappa"), kappa_eff=med("kappa_eff"),
            budget_mean=float(np.mean(g["budget"])), queries=med("queries"),
            matvecs=med("matvecs"), total_cost=med("total_cost"),
            rel_error_median=med("rel_error"),
            rel_error_max=float(np.nanmax(g["rel_error"])) if np.any(np.isfinite(g["rel_error"])) else np.nan,
            speedup_vs_qsvt=med("speedup_vs_qsvt"),
            speedup_vs_dalzell=med("speedup_vs_dalzell"),
        ))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+", default=[16, 64, 256, 1024])
    p.add_argument("--kappa", type=float, nargs="+", default=[1e2, 1e3, 1e4])
    p.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    p.add_argument("--w", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--families", nargs="+", default=list(MATRIX_FAMILIES),
                   choices=list(MATRIX_FAMILIES))
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--k-frac", type=float, default=0.125,
                   help="deflation budget cap k_max = max(1, N * k_frac)")
    p.add_argument("--mv-frac", type=float, default=0.5,
                   help="Lanczos cap = max(k_max + 2, N * mv_frac) matvecs")
    p.add_argument("--k-cap", type=int, default=16, help="absolute cap on k_max")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "modern_compare.csv")
    args = p.parse_args(argv)

    tasks = [(fam, N, kap, eps, t, tuple(args.w), args.seed, args.k_frac, args.mv_frac,
              args.k_cap)
             for fam in args.families for N in args.N for kap in args.kappa
             for eps in args.eps for t in range(args.trials)]
    print(f"{len(tasks)} systems on {args.workers} workers ...", flush=True)
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for i, rs in enumerate(ex.map(run_system, tasks, chunksize=64)):
                rows.extend(rs)
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        for task in tasks:
            rows.extend(run_system(task))
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_name(args.out.stem + "_raw.csv.gz")
    with gzip.open(raw_path, "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in RAW_FIELDS})
    summary = summarize(rows, args.w)
    with args.out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    print(f"wrote {len(rows)} raw rows -> {raw_path.name}, {len(summary)} summary rows -> "
          f"{args.out.name}")
    print_headline(summary, args)


def print_headline(summary, args):
    eps = min(args.eps)
    w = 0.1 if 0.1 in args.w else args.w[len(args.w) // 2]
    N = max(args.N)
    print(f"\nMedian total cost (queries + {w:g} * matvecs), N={N}, eps={eps:g}")
    methods = ["hhl_honest", "qsvt", "jennings", "dalzell", "hybrid_exact",
               "hybrid_adaptive", "hybrid_dalzell", "cg", "cg_bound"]
    print(f"{'family':>16} {'kappa':>7} " + " ".join(f"{m:>15}" for m in methods)
          + f" {'k':>5} {'x vs qsvt':>10} {'x vs dalzell':>12}")
    idx = {(r["family"], r["N"], r["kappa_nominal"], r["eps"], r["method"], r["w"]): r
           for r in summary}
    for fam in args.families:
        for kap in args.kappa:
            vals = [idx.get((fam, N, kap, eps, m, w)) for m in methods]
            if any(v is None for v in vals):
                continue
            ad = idx[(fam, N, kap, eps, "hybrid_adaptive", w)]
            print(f"{fam:>16} {kap:>7.0e} " + " ".join(f"{v['total_cost']:>15.4g}" for v in vals)
                  + f" {ad['budget_mean']:>5.1f} {ad['speedup_vs_qsvt']:>10.3g}"
                  f" {ad['speedup_vs_dalzell']:>12.3g}")


if __name__ == "__main__":
    main()

"""Large-sample checks of the exact-arithmetic residualization lemmas.

Each sample is a Hermitian system with a log-uniform condition number in
``[10, 1e4]``, the same family as the paper. The residual solver is
``numpy.linalg.solve``. Nothing here runs an HHL circuit.

One random stream is used per ``(N, budget, criterion)``. Checkpoints are
prefixes of that stream, so the 10,000-draw row is the first 10,000 draws of
the million-draw row.

Usage:
    python bench/theory_scale.py --samples 1000000 --N 2 4 8 --budget 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate import haar_orthogonal, sample_spectrum  # noqa: E402
from hhl.residualization import budgeted_affine_residualization  # noqa: E402

_SQRT2 = float(np.sqrt(2.0))
_FIELDS = [
    "samples", "N", "budget", "criterion", "signed",
    "max_recombine", "max_ortho", "max_growth", "n_growth_over",
    "n_kappa_over", "n_kappa_stuck", "max_formula",
    "max_reduced", "max_full", "n_wild_over", "max_wild", "seconds",
]


def _sample_system(n: int, rng: np.random.Generator, signed: bool) -> tuple[np.ndarray, np.ndarray]:
    kappa = float(10.0 ** rng.uniform(1.0, 4.0))
    spectrum = sample_spectrum(n, kappa, rng, signed=signed)
    Q = haar_orthogonal(n, rng)
    A = Q @ np.diag(spectrum) @ Q.T
    b = rng.normal(size=n)
    b /= np.linalg.norm(b)
    return A, b


def _empty():
    return {
        "n": 0,
        "max_recombine": 0.0,
        "max_ortho": 0.0,
        "max_growth": 0.0,
        "n_growth_over": 0,
        "n_kappa_over": 0,
        "n_kappa_stuck": 0,
        "max_formula": 0.0,
        "max_reduced": 0.0,
        "max_full": 0.0,
        "n_wild_over": 0,
        "max_wild": 0.0,
    }


def _update(stats: dict, value: float, key: str) -> None:
    stats[key] = max(stats[key], value)


def check_sample(A: np.ndarray, b: np.ndarray, criterion: str, stats: dict) -> None:
    r = budgeted_affine_residualization(A, b, stats.get("_budget"), criterion)
    x = np.linalg.solve(A, b)
    x_norm = float(np.linalg.norm(x))
    y_res = np.linalg.solve(A, r.b_res)
    recombine = float(np.linalg.norm(r.z + y_res - x) / x_norm)
    y = x - r.z
    ortho = float(abs(np.dot(r.z, y)) / x_norm**2)
    growth = float((np.linalg.norm(r.z) + np.linalg.norm(y)) / x_norm)

    order = np.argsort(np.abs(r.eigvals), kind="stable")
    k = int(len(r.removed))
    formula = float(np.abs(r.eigvals).max() / np.abs(r.eigvals[order[k]]))

    u_hard = np.real(r.eigvecs[:, r.removed])
    delta = 0.1 * float(np.linalg.norm(r.z)) * u_hard[:, 0]
    z_hat = r.z + delta
    Uk = np.real(r.eigvecs[:, r.kept])
    raw = b - A @ z_hat
    reduced_rhs = Uk @ (Uk.T @ raw)
    x_reduced = z_hat + np.linalg.solve(A, reduced_rhs)
    x_full = z_hat + np.linalg.solve(A, raw)
    hard = u_hard @ (u_hard.T @ delta)
    reduced_err = abs(float(np.linalg.norm(x_reduced - x)) - float(np.linalg.norm(hard))) / x_norm
    full_err = float(np.linalg.norm(x_full - x) / x_norm)

    direction = np.real(r.eigvecs[:, r.kept[0]])
    z_wild = x + 40.0 * x_norm * direction
    y_wild = x - z_wild
    wild = float((np.linalg.norm(z_wild) + np.linalg.norm(y_wild)) / x_norm)

    stats["n"] += 1
    _update(stats, recombine, "max_recombine")
    _update(stats, ortho, "max_ortho")
    _update(stats, growth, "max_growth")
    stats["n_growth_over"] += int(growth > _SQRT2 + 1e-6)
    stats["n_kappa_over"] += int(r.kappa_eff > r.kappa * (1.0 + 1e-8) + 1e-8)
    stats["n_kappa_stuck"] += int(r.kappa_eff > 0.99 * r.kappa)
    if criterion == "smallest":
        _update(stats, abs(r.kappa_eff - formula) / formula, "max_formula")
    _update(stats, reduced_err, "max_reduced")
    _update(stats, full_err, "max_full")
    stats["n_wild_over"] += int(wild > _SQRT2 + 1e-6)
    _update(stats, wild, "max_wild")


def _row(stats: dict, samples: int, n: int, budget: int, criterion: str,
         signed: bool, seconds: float) -> dict:
    formula = stats["max_formula"] if criterion == "smallest" else ""
    return {
        "samples": samples,
        "N": n,
        "budget": budget,
        "criterion": criterion,
        "signed": int(signed),
        "max_recombine": stats["max_recombine"],
        "max_ortho": stats["max_ortho"],
        "max_growth": stats["max_growth"],
        "n_growth_over": stats["n_growth_over"],
        "n_kappa_over": stats["n_kappa_over"],
        "n_kappa_stuck": stats["n_kappa_stuck"],
        "max_formula": formula,
        "max_reduced": stats["max_reduced"],
        "max_full": stats["max_full"],
        "n_wild_over": stats["n_wild_over"],
        "max_wild": stats["max_wild"],
        "seconds": seconds,
    }


def _print_row(row: dict) -> None:
    formula = (
        f"formula={row['max_formula']:.2e}"
        if row["max_formula"] != "" else "formula=n/a"
    )
    print(
        f"N={row['N']:<3} k={row['budget']} {row['criterion']:<9} "
        f"n={row['samples']:<8} "
        f"recombine={row['max_recombine']:.2e}  "
        f"ortho={row['max_ortho']:.2e}  "
        f"growth={row['max_growth']:.6f}  "
        f"over_sqrt2={row['n_growth_over']}  "
        f"kappa_over={row['n_kappa_over']}  "
        f"kappa_stuck={row['n_kappa_stuck']}  "
        f"{formula}  "
        f"reduced={row['max_reduced']:.2e}  "
        f"full={row['max_full']:.2e}  "
        f"wild_over={row['n_wild_over']}  "
        f"{row['seconds']:.1f}s",
        flush=True,
    )


def run_cell(task: tuple) -> list[dict]:
    n, budget, criterion, signed, seed, samples, checkpoints = task
    rng = np.random.default_rng(seed + 1000 * n + 10 * budget + int(criterion == "energy") + 17 * int(signed))
    stats = _empty()
    stats["_budget"] = budget
    marks = set(c for c in checkpoints if c <= samples)
    rows = []
    t0 = time.perf_counter()
    for i in range(1, samples + 1):
        A, b = _sample_system(n, rng, signed)
        check_sample(A, b, criterion, stats)
        if i in marks or i == samples:
            row = _row(stats, i, n, budget, criterion, signed, time.perf_counter() - t0)
            rows.append(row)
            _print_row(row)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--record", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    p.add_argument("--N", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--budget", type=int, nargs="+", default=[1])
    p.add_argument("--signed", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "theory_scale_log.csv")
    args = p.parse_args(argv)

    tasks = [
        (n, budget, criterion, args.signed, args.seed, args.samples, tuple(args.record))
        for n in args.N
        for budget in args.budget
        if budget < n
        for criterion in ("smallest", "energy")
    ]
    print(
        f"{len(tasks)} streams up to {args.samples} samples, "
        f"record={args.record}, workers={args.workers}",
        flush=True,
    )
    rows: list[dict] = []
    workers = min(args.workers, len(tasks))
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(run_cell, task) for task in tasks]
        for future in as_completed(futures):
            cell_rows = future.result()
            rows.extend(cell_rows)
            rows.sort(key=lambda r: (r["samples"], r["N"], r["budget"], r["criterion"]))
            _write(args.out, rows)
            print(f"wrote {len(rows)} rows -> {args.out}", flush=True)
    print(f"done, {len(rows)} rows in {args.out}", flush=True)


if __name__ == "__main__":
    main()

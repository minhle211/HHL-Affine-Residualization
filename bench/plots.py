"""Figures from results/bench.csv or results/modern_compare.csv (PNG + PDF in figures/).

Usage:
    python bench/plots.py [--csv results/bench.csv] [--out figures]
    python bench/plots.py --modern [results/modern_compare.csv]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NUM = {"N", "kappa_nominal", "trial", "budget", "delta_rel", "kappa", "kappa_eff",
       "kappa_target", "n_clock", "depth", "cx", "num_qubits", "p_success",
       "p_success_clock0", "hhl_fidelity", "fidelity", "rel_error", "z_rel_err", "b_res_rel"}


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in NUM:
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else np.nan
            rows.append(r)
    return rows


def agg(rows, key, field):
    """Mean of ``field`` grouped by ``key`` -> (sorted keys, means)."""
    g = defaultdict(list)
    for r in rows:
        if not np.isnan(r[field]):
            g[r[key]].append(r[field])
    ks = sorted(g)
    return np.array(ks), np.array([np.mean(g[k]) for k in ks])


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print("saved", out / f"{name}.png")


def plot_vs_kappa(rows, out, field, ylabel, name, logy=True):
    grid = [r for r in rows if r["experiment"] == "grid"]
    Ns = sorted({int(r["N"]) for r in grid})
    if not Ns:
        return
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.2 * len(Ns), 3.4), squeeze=False)
    for ax, N in zip(axes[0], Ns):
        sub = [r for r in grid if r["N"] == N]
        for k in sorted({int(r["budget"]) for r in sub}):
            x, y = agg([r for r in sub if r["budget"] == k], "kappa_nominal", field)
            ax.plot(x, y, "o-", label="standard HHL" if k == 0 else f"hybrid k={k}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"N={N}")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, out, name)


def plot_kappa_eff(rows, out):
    grid = [r for r in rows if r["experiment"] == "grid"]
    Ns = sorted({int(r["N"]) for r in grid})
    if not Ns:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for N in Ns:
        for kap in sorted({r["kappa_nominal"] for r in grid}):
            sub = [r for r in grid if r["N"] == N and r["kappa_nominal"] == kap]
            x, y = agg(sub, "budget", "kappa_eff")
            if len(x) > 1:
                ax.plot(x, y, "o-", label=fr"N={N}, $\kappa$={kap:g}")
    ax.set_yscale("log")
    ax.set_xlabel("budget k (modes removed)")
    ax.set_ylabel(r"$\kappa_{\mathrm{eff}}$ (mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save(fig, out, "kappa_eff_vs_budget")


def plot_zsource(rows, out):
    zs = [r for r in rows if r["experiment"] == "zsource"]
    if not zs:
        return
    sources = list(dict.fromkeys(r["z_source"] for r in zs))
    fig, ax = plt.subplots(figsize=(5.4, 4))
    markers = dict(zip(sources, "osD^vP*"))
    for src in sources:
        for mode, fill in (("reduced", True), ("full", False)):
            sel = [r for r in zs if r["z_source"] == src and r["mode"] == mode]
            x = np.maximum([r["z_rel_err"] for r in sel], 1e-16)
            y = np.maximum([r["rel_error"] for r in sel], 1e-16)
            c = f"C{sources.index(src)}"
            style = dict(color=c) if fill else dict(facecolors="none", edgecolors=c)
            ax.scatter(x, y, marker=markers[src], s=28, alpha=0.7,
                       label=f"{src} ({mode})", **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\|\hat z - z\| / \|z\|$")
    ax.set_ylabel(r"final $\|\hat x - x\| / \|x\|$")
    ax.set_title("z accuracy vs final error (filled: reduced, hollow: full)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    save(fig, out, "zsource_error")


def plot_tradeoff(rows, out):
    tr = [r for r in rows if r["experiment"] == "tradeoff"]
    if not tr:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    configs = sorted({(int(r["N"]), r["kappa_nominal"]) for r in tr})
    for ci, (N, kap) in enumerate(configs):
        for mi, (mode, ls) in enumerate((("reduced", "-"), ("full", "--"))):
            sel = [r for r in tr if r["N"] == N and r["kappa_nominal"] == kap
                   and r["mode"] == mode]
            x, y = agg(sel, "delta_rel", "rel_error")
            axes[0].plot(np.maximum(x, 1e-3), np.maximum(y, 1e-8), ls, marker="o",
                         color=f"C{ci}", label=fr"N={N} $\kappa$={kap:g} {mode}")
            _, c = agg(sel, "delta_rel", "cx")
            if len(c):
                axes[1].bar(ci + (mi - 0.5) * 0.4, np.nanmean(c), width=0.4,
                            color="C0" if mode == "reduced" else "C3",
                            label=mode if ci == 0 else None)
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels([f"N={N}\n$\\kappa$={k:g}" for N, k in configs], fontsize=7)
    axes[1].legend(fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\|\delta\| / \|z\|$ on hard mode (0 plotted at 1e-3)")
    axes[0].set_ylabel(r"$\|\hat x - x\| / \|x\|$")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    axes[1].set_ylabel("CX count (transpiled)")
    axes[1].set_yscale("log")
    save(fig, out, "tradeoff_reduced_vs_full")


# ------------------------------------------------ modern_compare.csv figures

MODERN_NUM = {"N", "kappa_nominal", "eps", "w", "n_systems", "kappa", "kappa_eff",
              "budget_mean", "queries", "matvecs", "total_cost", "rel_error_median",
              "rel_error_max", "speedup_vs_qsvt", "speedup_vs_dalzell"}
MODERN_STYLE = {
    "hhl_honest": ("textbook HHL", "C7", ":"),
    "qsvt": ("QSVT", "C0", "--"),
    "jennings": ("Jennings et al. 2024", "C4", "--"),
    "dalzell": ("Dalzell 2024 (optimal)", "C3", "--"),
    "hybrid_exact": ("deflated QSVT, exact z (quantum only)", "C2", ":"),
    "hybrid_adaptive": ("deflated QSVT, Lanczos adaptive", "C2", "-"),
    "hybrid_dalzell": ("deflated Dalzell, Lanczos adaptive", "C1", "-"),
    "cg_bound": ("classical CG bound", "k", "-."),
}


def load_modern(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in MODERN_NUM:
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None", "nan") else np.nan
            rows.append(r)
    return rows


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def plot_modern_cost_vs_kappa(rows, out, w=0.1):
    N, eps = max(r["N"] for r in rows), min(r["eps"] for r in rows)
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(2, (len(fams) + 1) // 2, figsize=(4.4 * ((len(fams) + 1) // 2), 7),
                             squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        for m, (label, color, ls) in MODERN_STYLE.items():
            sel = sorted(_sel(rows, family=fam, N=N, eps=eps, method=m, w=w),
                         key=lambda r: r["kappa"])
            if sel:
                ax.plot([r["kappa"] for r in sel], [r["total_cost"] for r in sel], ls,
                        marker="o", ms=3, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{fam}  (N={N:g})", fontsize=9)
        ax.set_xlabel(r"median $\kappa$")
        ax.set_ylabel(f"total cost (queries + {w:g} matvecs)")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=6)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    save(fig, out, "modern_cost_vs_kappa")


def plot_modern_breakeven(rows, out):
    N, eps = max(r["N"] for r in rows), min(r["eps"] for r in rows)
    kap = max(r["kappa_nominal"] for r in rows)
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, fam in enumerate(fams):
        for ax, m, field in ((axes[0], "hybrid_adaptive", "speedup_vs_qsvt"),
                             (axes[1], "hybrid_dalzell", "speedup_vs_dalzell")):
            sel = sorted(_sel(rows, family=fam, N=N, eps=eps, kappa_nominal=kap, method=m),
                         key=lambda r: r["w"])
            if sel:
                ax.plot([r["w"] for r in sel], [r[field] for r in sel], "o-",
                        color=f"C{i}", label=fam)
    for ax, title in ((axes[0], "deflated QSVT vs QSVT"),
                      (axes[1], "deflated Dalzell vs Dalzell (optimal)")):
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("w (cost of one classical matvec in queries)")
        ax.set_ylabel("median per-system speedup")
        ax.set_title(f"{title}\nN={N:g}, kappa={kap:g}, eps={eps:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    save(fig, out, "modern_breakeven_w")


def plot_modern_speedup_vs_N(rows, out, w=0.1):
    eps = min(r["eps"] for r in rows)
    kap = max(r["kappa_nominal"] for r in rows)
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, fam in enumerate(fams):
        for ax, m, field in ((axes[0], "hybrid_adaptive", "speedup_vs_qsvt"),
                             (axes[1], "hybrid_dalzell", "speedup_vs_dalzell")):
            sel = sorted(_sel(rows, family=fam, eps=eps, kappa_nominal=kap, method=m, w=w),
                         key=lambda r: r["N"])
            if sel:
                ax.plot([r["N"] for r in sel], [r[field] for r in sel], "o-",
                        color=f"C{i}", label=fam)
    for ax, title in ((axes[0], "deflated QSVT vs QSVT"),
                      (axes[1], "deflated Dalzell vs Dalzell (optimal)")):
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("median per-system speedup")
        ax.set_title(f"{title}\nkappa={kap:g}, eps={eps:g}, w={w:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    save(fig, out, "modern_speedup_vs_N")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "results" / "bench.csv")
    p.add_argument("--modern", type=Path, nargs="?", const=ROOT / "results" / "modern_compare.csv",
                   help="plot results/modern_compare.csv instead of bench.csv")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args(argv)
    if args.modern is not None:
        rows = load_modern(args.modern)
        plot_modern_cost_vs_kappa(rows, args.out)
        plot_modern_breakeven(rows, args.out)
        plot_modern_speedup_vs_N(rows, args.out)
        return
    rows = load(args.csv)
    plot_vs_kappa(rows, args.out, "p_success", r"$P(\mathrm{ancilla}=1)$", "p_success_vs_kappa")
    plot_vs_kappa(rows, args.out, "depth", "transpiled depth", "depth_vs_kappa")
    plot_vs_kappa(rows, args.out, "cx", "CX count", "cx_vs_kappa")
    plot_vs_kappa(rows, args.out, "n_clock", "clock qubits", "clock_vs_kappa", logy=False)
    plot_vs_kappa(rows, args.out, "fidelity", "solution fidelity", "fidelity_vs_kappa",
                  logy=False)
    plot_kappa_eff(rows, args.out)
    plot_zsource(rows, args.out)
    plot_tradeoff(rows, args.out)


if __name__ == "__main__":
    main()

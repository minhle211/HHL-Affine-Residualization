"""Train the z-predicting MLP and compare it with classical baselines.

Loss = relative MSE(z_hat, z) + w_phys * ||P_hard (b - A z_hat)|| / ||b||
       + w_log * (log ||z||_pred - log ||z||)^2   (auxiliary, helps the scale).

Evaluation on the held-out test split reports, for each z source:
* ``z_rel_err``  = ||z_hat - z|| / ||z||
* ``x_err_reduced`` = ||P_hard (z_hat - z)|| / ||x||: final solution error with
  the reduced (kappa_eff) circuit, which cannot correct hard-mode errors
  (assuming the HHL step itself is exact);
* ``x_err_full`` = 0 in exact arithmetic: the full-kappa circuit solves the raw
  residual and corrects any z error, at full-kappa circuit cost.

Usage:
    python ml/train.py --data data/dataset_N4_k1_energy.npz --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baselines import BASELINES  # noqa: E402
from ml.model import (  # noqa: E402
    ZNet,
    features,
    lognorm_mse,
    physics_loss,
    relative_mse,
    save_model,
)


def split_indices(n: int, seed: int = 0, frac=(0.8, 0.1)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def solution_errors(z_hat: np.ndarray, z: np.ndarray, P_hard: np.ndarray, x: np.ndarray) -> dict:
    dz = z_hat - z
    hard = np.einsum("nij,nj->ni", P_hard, dz)
    z_rel = np.linalg.norm(dz, axis=1) / np.linalg.norm(z, axis=1)
    red = np.linalg.norm(hard, axis=1) / np.linalg.norm(x, axis=1)
    return {
        "z_rel_err_median": float(np.median(z_rel)),
        "z_rel_err_mean": float(np.mean(z_rel)),
        "x_err_reduced_median": float(np.median(red)),
        "x_err_reduced_mean": float(np.mean(red)),
        "x_err_reduced_p90": float(np.quantile(red, 0.9)),
        "x_err_full": 0.0,
    }


def train(data_path: Path, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
          w_phys: float = 1.0, w_log: float = 0.1, hidden: int = 256, depth: int = 3,
          seed: int = 0,
          out_dir: Path = ROOT / "ml" / "checkpoints", results_dir: Path = ROOT / "results",
          verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    d = np.load(data_path)
    A, b, z, P, x = d["A"], d["b"], d["z"], d["P_hard"], d["x"]
    margin = d["selection_margin"]
    N = A.shape[1]
    tr, va, te = split_indices(len(A), seed)

    def tens(ix):
        return (torch.tensor(A[ix], dtype=torch.float32), torch.tensor(b[ix], dtype=torch.float32),
                torch.tensor(z[ix], dtype=torch.float32), torch.tensor(P[ix], dtype=torch.float32))

    Atr, btr, ztr, Ptr = tens(tr)
    Ava, bva, zva, Pva = tens(va)
    Ftr, Fva = features(Atr, btr), features(Ava, bva)

    model = ZNet(N, hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        tot = tot_mse = tot_phys = 0.0
        for i in range(0, len(tr), batch_size):
            j = perm[i : i + batch_size]
            dh, sh = model.parts(Ftr[j])
            zh = torch.exp(sh.clamp(max=20.0))[:, None] * dh
            l_mse = relative_mse(zh, ztr[j])
            l_phys = physics_loss(Atr[j], btr[j], Ptr[j], zh)
            loss = l_mse + w_phys * l_phys + w_log * lognorm_mse(sh, ztr[j])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n = len(j)
            tot += loss.item() * n
            tot_mse += l_mse.item() * n
            tot_phys += l_phys.item() * n
        sched.step()
        model.eval()
        with torch.no_grad():
            zh = model(Fva)
            v_mse = relative_mse(zh, zva).item()
            v_phys = physics_loss(Ava, bva, Pva, zh).item()
        rec = dict(epoch=ep + 1, train_loss=tot / len(tr), train_mse=tot_mse / len(tr),
                   train_phys=tot_phys / len(tr), val_mse=v_mse, val_phys=v_phys,
                   val_loss=v_mse + w_phys * v_phys)
        history.append(rec)
        if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep + 1 == epochs):
            print(f"[N={N}] epoch {ep + 1:3d}  train={rec['train_loss']:.4f} "
                  f"(mse={rec['train_mse']:.4f}, phys={rec['train_phys']:.4f})  "
                  f"val={rec['val_loss']:.4f}")
    train_time = time.time() - t0

    # ---------------------------------------------------------------- test
    with torch.no_grad():
        z_mlp = model(features(torch.tensor(A[te], dtype=torch.float32),
                               torch.tensor(b[te], dtype=torch.float32))).double().numpy()
    budget = int(d["budget"])
    results = {"MLP": solution_errors(z_mlp, z[te], P[te], x[te])}
    for name, fn in BASELINES.items():
        zb = np.stack([fn(A[i], b[i], budget) for i in te])
        results[name] = solution_errors(zb, z[te], P[te], x[te])

    # error stratified by how close the hard-mode selection is to a tie
    tie = margin[te] < 1.5
    dz = np.linalg.norm(z_mlp - z[te], axis=1) / np.linalg.norm(z[te], axis=1)
    strat = {"frac_near_tie": float(tie.mean()),
             "mlp_z_rel_err_near_tie": float(np.median(dz[tie])) if tie.any() else None,
             "mlp_z_rel_err_clear": float(np.median(dz[~tie])) if (~tie).any() else None}

    ckpt = out_dir / f"mlp_N{N}.pt"
    save_model(model, ckpt, meta={"data": str(data_path), "budget": budget,
                                  "criterion": str(d["criterion"])})
    summary = {"N": N, "n_train": len(tr), "n_test": len(te), "epochs": epochs,
               "train_time_s": train_time, "w_phys": w_phys, "checkpoint": str(ckpt),
               "final": history[-1], "test": results, "tie_stratification": strat,
               "history": history}
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"train_N{N}.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[N={N}] trained in {train_time:.1f}s -> {ckpt}")
        print(f"[N={N}] test (median):  source        z_rel_err   x_err_reduced   x_err_full")
        for k, r in results.items():
            print(f"         {k:>14}   {r['z_rel_err_median']:10.3e}   "
                  f"{r['x_err_reduced_median']:13.3e}   {r['x_err_full']:.1f}")
        print(f"[N={N}] near-tie fraction={strat['frac_near_tie']:.1%}  MLP z err near-tie="
              f"{strat['mlp_z_rel_err_near_tie']}  clear={strat['mlp_z_rel_err_clear']}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-phys", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=0.1, help="auxiliary log-norm loss weight")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "ml" / "checkpoints")
    args = p.parse_args(argv)
    for path in args.data:
        train(path, args.epochs, args.batch_size, args.lr, args.w_phys, args.w_log,
              args.hidden, args.depth, args.seed, args.out)


if __name__ == "__main__":
    main()

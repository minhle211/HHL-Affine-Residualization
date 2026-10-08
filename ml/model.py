"""MLP that predicts the deflation vector z from (A, b).

``z`` spans several orders of magnitude (``|z| ~ |beta|/lambda_min`` up to
~1e4), so the network predicts a direction and a log-norm separately:
``z_hat = exp(s) * d / ||d||``. ``z`` is invariant to eigenvector sign flips
but jumps when the hard-mode selection changes (near-tied mode energies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def n_features(N: int) -> int:
    return N * (N + 1) + 3 * N


def features(A: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Cheap polynomial features (matrix products / matvecs only, no solves).

    Upper triangles of ``A`` and ``A^2`` plus ``b``, ``A b``, ``A^2 b``.
    Batched (``A: (B, N, N)``) or single (``A: (N, N)``).
    """
    A = torch.as_tensor(A, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    single = A.ndim == 2
    if single:
        A, b = A[None], b[None]
    N = A.shape[-1]
    iu = torch.triu_indices(N, N)
    A2 = A @ A
    Ab = torch.einsum("bij,bj->bi", A, b)
    A2b = torch.einsum("bij,bj->bi", A, Ab)
    f = torch.cat([A[:, iu[0], iu[1]], A2[:, iu[0], iu[1]], b, Ab, A2b], dim=1)
    return f[0] if single else f


class ZNet(nn.Module):
    def __init__(self, N: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.N = N
        d_in = n_features(N)
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.dir_head = nn.Linear(d, N)
        self.lognorm_head = nn.Linear(d, 1)

    def parts(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit direction ``d`` and log-norm ``s`` (shape ``(B,)``)."""
        h = self.body(feats)
        d = self.dir_head(h)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-12)
        return d, self.lognorm_head(h).squeeze(-1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        d, s = self.parts(feats)
        return torch.exp(s.clamp(max=20.0))[..., None] * d


def physics_loss(A: torch.Tensor, b: torch.Tensor, P_hard: torch.Tensor,
                 z_hat: torch.Tensor) -> torch.Tensor:
    """Mean ``||P_hard (b - A z_hat)|| / ||b||``: hard-mode residual left by z_hat."""
    r = b - torch.einsum("bij,bj->bi", A, z_hat)
    rh = torch.einsum("bij,bj->bi", P_hard, r)
    return (rh.norm(dim=-1) / b.norm(dim=-1)).mean()


def relative_mse(z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (((z_hat - z) ** 2).sum(-1) / (z**2).sum(-1).clamp_min(1e-30)).mean()


def lognorm_mse(s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss on the predicted log-norm (better conditioned than z itself)."""
    return ((s - torch.log(z.norm(dim=-1).clamp_min(1e-30))) ** 2).mean()


def save_model(model: ZNet, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"N": model.N, "state_dict": model.state_dict(),
                "hidden": model.body[0].out_features,
                "depth": sum(isinstance(m, nn.Linear) for m in model.body),
                "meta": meta or {}}, path)


def load_model(path: str | Path) -> ZNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ZNet(ck["N"], hidden=ck["hidden"], depth=ck["depth"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


class MLPZProvider:
    """z-provider ``(A, b, budget) -> z`` backed by a trained ZNet.

    The network was trained for a fixed budget/criterion; ``budget`` is
    accepted for interface compatibility only.
    """

    def __init__(self, model: ZNet):
        self.model = model.eval()

    @classmethod
    def from_path(cls, path: str | Path) -> "MLPZProvider":
        return cls(load_model(path))

    def __call__(self, A: np.ndarray, b: np.ndarray, budget: int) -> np.ndarray:
        if A.shape[0] != self.model.N:
            raise ValueError(f"model trained for N={self.model.N}, got N={A.shape[0]}")
        with torch.no_grad():
            z = self.model(features(A, b)[None])[0]
        return z.double().numpy()

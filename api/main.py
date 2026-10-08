"""Thin FastAPI wrapper around ``hhl.solve_hybrid``.

Run:  uvicorn api.main:app --port 8000
POST /solve  {"A": [[...]], "b": [...], "budget": 1, "z_source": "exact"}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hhl import solve_hybrid  # noqa: E402
from ml.baselines import BASELINES  # noqa: E402

MAX_DIM = int(os.environ.get("HHL_MAX_DIM", "8"))
CKPT_DIR = Path(os.environ.get("HHL_CKPT_DIR", ROOT / "ml" / "checkpoints"))

app = FastAPI(title="Hybrid affine-residualized HHL", version="0.1.0")


class SolveRequest(BaseModel):
    A: list[list[float]]
    b: list[float]
    budget: int = Field(1, ge=0)
    criterion: Literal["energy", "smallest"] = "energy"
    mode: Literal["reduced", "full"] = "reduced"
    solver: Literal["hhl", "classical"] = "hhl"
    z_source: Literal["exact", "lanczos", "inverse_iter", "rand_svd", "mlp"] = "exact"
    extra_bits: int = Field(2, ge=0, le=4)
    max_clock: int = Field(10, ge=2, le=12)
    compute_resources: bool = False


class SolveResponse(BaseModel):
    x: list[float]
    z: list[float]
    y: list[float]
    b_res_norm: float
    metrics: dict


def _provider(name: str, N: int):
    if name == "exact":
        return None
    if name == "mlp":
        from ml.model import MLPZProvider

        path = CKPT_DIR / f"mlp_N{N}.pt"
        if not path.exists():
            raise HTTPException(400, f"no MLP checkpoint for N={N} at {path}")
        return MLPZProvider.from_path(path)
    return BASELINES[name]


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    A = np.asarray(req.A, dtype=float)
    b = np.asarray(req.b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise HTTPException(422, "A must be square (N x N) and b of length N")
    if A.shape[0] > MAX_DIM:
        raise HTTPException(422, f"N={A.shape[0]} exceeds HHL_MAX_DIM={MAX_DIM}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)) or not np.any(b):
        raise HTTPException(422, "A and b must be finite and b non-zero")
    try:
        out = solve_hybrid(
            A, b,
            z_provider=_provider(req.z_source, A.shape[0]),
            budget=req.budget,
            criterion=req.criterion,
            mode=req.mode,
            solver=req.solver,
            extra_bits=req.extra_bits,
            max_clock=req.max_clock,
            compute_resources=req.compute_resources,
        )
    except np.linalg.LinAlgError as e:
        raise HTTPException(422, f"linear algebra error: {e}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return SolveResponse(
        x=np.real(out.x).tolist(),
        z=np.real(out.z).tolist(),
        y=np.real(out.y).tolist(),
        b_res_norm=out.b_res_norm,
        metrics={k: _jsonable(v) for k, v in out.metrics.items()},
    )

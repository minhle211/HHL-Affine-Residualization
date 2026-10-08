import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from ml.baselines import BASELINES, exact_z
from ml.model import MLPZProvider, ZNet, features, physics_loss

client = TestClient(app)


def _system(n=4, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(np.logspace(-2, 0, n)) @ Q.T
    b = rng.normal(size=n)
    return A, b / np.linalg.norm(b)


def test_api_solve_hhl():
    A, b = _system()
    r = client.post("/solve", json={"A": A.tolist(), "b": b.tolist(), "budget": 1})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"x", "z", "y", "metrics"}
    assert d["metrics"]["fidelity"] > 0.95
    x = np.linalg.solve(A, b)
    assert np.linalg.norm(np.array(d["x"]) - x) / np.linalg.norm(x) < 0.2


def test_api_classical_exact():
    A, b = _system(seed=1)
    r = client.post("/solve", json={"A": A.tolist(), "b": b.tolist(), "budget": 2,
                                    "solver": "classical"})
    assert r.status_code == 200
    np.testing.assert_allclose(r.json()["x"], np.linalg.solve(A, b), rtol=1e-8)


def test_api_rejects_bad_input():
    r = client.post("/solve", json={"A": [[1, 2, 3]], "b": [1]})
    assert r.status_code == 422


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_exact_when_subspace_is_full(name):
    A, b = _system(n=2, seed=3)
    np.testing.assert_allclose(BASELINES[name](A, b, 1), exact_z(A, b, 1), rtol=1e-6, atol=1e-8)


def test_mlp_provider_shapes_and_physics_loss_zero_for_exact_z():
    A, b = _system()
    model = ZNet(4, hidden=16, depth=2)
    z = MLPZProvider(model)(A, b, 1)
    assert z.shape == (4,)
    assert features(A, b).shape[0] == 4 * 5 + 12
    from hhl.residualization import budgeted_affine_residualization

    r = budgeted_affine_residualization(A, b, 1)
    P = r.eigvecs[:, r.removed] @ r.eigvecs[:, r.removed].T
    t = lambda v: torch.tensor(v[None], dtype=torch.float64)
    assert physics_loss(t(A), t(b), t(P), t(r.z)).item() < 1e-10

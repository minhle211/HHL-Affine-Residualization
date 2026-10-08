import numpy as np

from hhl_algorithm import budgeted_affine_residualization, run_hybrid_quantum_linear_solver


def test_residualization_wrapper_shapes():
    A = np.diag([1e-4, 5e-3])
    z, b_res, top = budgeted_affine_residualization(A, np.array([1.0, 1.0]), budget=1)
    assert z.shape == (2,) and b_res.shape == (2,) and len(top) == 1
    np.testing.assert_allclose(z + np.linalg.solve(A, b_res), np.linalg.solve(A, [1.0, 1.0]))


def test_original_example_skips_hhl_and_is_exact():
    A = np.diag([1e-4, 5e-3])
    b = np.array([0.0, 100.0])
    x, z, y = run_hybrid_quantum_linear_solver(A, b, budget=1)
    np.testing.assert_allclose(x, np.linalg.solve(A, b))
    np.testing.assert_allclose(y, 0.0)


def test_multimode_runs_hhl_and_matches_exact():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    A = Q @ np.diag([1e-3, 0.05, 0.4, 1.0]) @ Q.T
    b = rng.normal(size=4)
    x, z, y = run_hybrid_quantum_linear_solver(A, b, budget=1)
    assert x.shape == z.shape == y.shape == (4,)
    assert np.linalg.norm(y) > 0
    x_exact = np.linalg.solve(A, b)
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 0.05

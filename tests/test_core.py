import numpy as np
import pytest

from hhl import (
    budgeted_affine_residualization,
    choose_hhl_params,
    fidelity,
    hermitian_embedding,
    run_hhl,
    solve_hybrid,
    solve_standard,
)


def spd_from_spectrum(eigs, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(eigs) @ Q.T


def spread_rhs(n, seed=1):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n)
    return b / np.linalg.norm(b)


# ---------------------------------------------------------------- classical path


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("budget", [0, 1, 2])
def test_classical_recombination_exact(n, budget):
    eigs = np.logspace(-3, 0, n)
    A = spd_from_spectrum(eigs, seed=n)
    b = spread_rhs(n, seed=n + 7)
    out = solve_hybrid(A, b, budget=budget, solver="classical")
    np.testing.assert_allclose(out.z + out.b_res_norm * out.y, np.linalg.solve(A, b),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.x, np.linalg.solve(A, b), rtol=1e-9, atol=1e-9)


def test_classical_path_non_hermitian_embedding():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(2, 2)) + 2 * np.eye(2)
    b = rng.normal(size=2)
    out = solve_hybrid(A, b, budget=1, solver="classical")
    assert out.metrics["embedded"]
    np.testing.assert_allclose(out.x, np.linalg.solve(A, b), rtol=1e-9, atol=1e-9)


def test_classical_path_non_power_of_two():
    A = spd_from_spectrum([0.01, 0.3, 1.0], seed=5)
    b = spread_rhs(3)
    out = solve_hybrid(A, b, budget=1, solver="classical")
    np.testing.assert_allclose(out.x, np.linalg.solve(A, b), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------- residualization


@pytest.mark.parametrize("criterion", ["energy", "smallest"])
def test_residual_orthogonal_to_removed_modes(criterion):
    eigs = np.logspace(-4, 0, 8)
    A = spd_from_spectrum(eigs, seed=11)
    b = spread_rhs(8, seed=12)
    r = budgeted_affine_residualization(A, b, budget=3, criterion=criterion)
    U_rem = r.eigvecs[:, r.removed]
    assert np.max(np.abs(U_rem.T @ r.b_res)) < 1e-12 * np.linalg.norm(b)
    # z lives entirely in the removed subspace and z + A^{-1} b_res = A^{-1} b
    np.testing.assert_allclose(r.z + np.linalg.solve(A, r.b_res), np.linalg.solve(A, b),
                               rtol=1e-8, atol=1e-8)


def test_kappa_eff_decreases_with_smallest_criterion():
    eigs = np.logspace(-3, 0, 8)
    A = spd_from_spectrum(eigs, seed=2)
    b = spread_rhs(8)
    kappas = [budgeted_affine_residualization(A, b, k, "smallest").kappa_eff
              for k in range(4)]
    assert all(k1 < k0 for k0, k1 in zip(kappas, kappas[1:]))
    np.testing.assert_allclose(kappas[0], 1e3, rtol=1e-6)


def test_predicted_z_error_on_hard_mode_not_corrected_by_reduced_circuit():
    eigs = np.array([1e-3, 0.2, 0.5, 1.0])
    A = spd_from_spectrum(eigs, seed=4)
    b = spread_rhs(4, seed=9)
    exact = budgeted_affine_residualization(A, b, 1, "smallest")
    delta = 0.05 * np.linalg.norm(exact.z) * exact.eigvecs[:, exact.removed[0]]
    z_hat = exact.z + delta
    prov = lambda A_, b_, k: z_hat
    red = solve_hybrid(A, b, prov, 1, "smallest", mode="reduced", solver="classical")
    full = solve_hybrid(A, b, prov, 1, "smallest", mode="full", solver="classical")
    x = np.linalg.solve(A, b)
    np.testing.assert_allclose(np.linalg.norm(red.x - x), np.linalg.norm(delta), rtol=1e-6)
    assert np.linalg.norm(full.x - x) < 1e-9


# ---------------------------------------------------------------- HHL circuit


def test_hhl_n2_grid_aligned():
    A = spd_from_spectrum([1.0, 3.0], seed=0)
    b = np.array([1.0, 0.4])
    p = choose_hhl_params(1.0, 3.0)
    out = run_hhl(A, b, p)
    x = np.linalg.solve(A, b / np.linalg.norm(b))
    assert fidelity(out.y, x) > 0.999
    np.testing.assert_allclose(np.real(out.y), x, atol=1e-6)


def test_hhl_n4_grid_aligned():
    A = spd_from_spectrum([1.0, 2.0, 3.0, 4.0], seed=1)
    b = spread_rhs(4)
    p = choose_hhl_params(1.0, 4.0)
    out = run_hhl(A, b, p)
    assert fidelity(out.y, np.linalg.solve(A, b)) > 0.999


@pytest.mark.parametrize("n", [2, 4])
def test_hhl_generic_spectrum_fidelity(n):
    A = spd_from_spectrum(np.logspace(-1, 0, n) * np.array([1.0, 1.07, 0.93, 1.0])[:n],
                          seed=n)
    b = spread_rhs(n, seed=2 * n)
    out = solve_standard(A, b, extra_bits=3)
    assert out.metrics["fidelity"] > 0.95


def test_hhl_signed_spectrum_via_embedding():
    A = np.array([[2.0, 0.5], [-0.3, 1.0]])
    b = np.array([0.6, 0.8])
    out = solve_hybrid(A, b, budget=0, extra_bits=3)
    assert out.metrics["embedded"]
    assert out.metrics["fidelity"] > 0.9


def test_hybrid_raises_success_probability():
    eigs = np.array([0.01, 0.4, 0.7, 1.0])
    A = spd_from_spectrum(eigs, seed=6)
    b = spread_rhs(4, seed=3)
    std = solve_standard(A, b, extra_bits=2)
    hyb = solve_hybrid(A, b, budget=1, criterion="smallest", extra_bits=2)
    assert hyb.metrics["kappa_eff"] < std.metrics["kappa"]
    assert hyb.metrics["n_clock"] < std.metrics["n_clock"]
    assert hyb.metrics["p_success"] > std.metrics["p_success"]
    assert hyb.metrics["fidelity"] > 0.95


def test_hhl_skipped_when_residual_vanishes():
    A = np.diag([1e-3, 1.0])
    b = np.array([1.0, 0.0])
    out = solve_hybrid(A, b, budget=1)
    assert out.metrics["hhl_skipped"]
    np.testing.assert_allclose(out.x, [1e3, 0.0])


def test_non_hermitian_rejected_by_residualization():
    with pytest.raises(ValueError):
        budgeted_affine_residualization(np.array([[1.0, 2.0], [0.0, 1.0]]), np.ones(2), 1)


def test_embedding_shape():
    H, be = hermitian_embedding(np.eye(2), np.ones(2))
    assert H.shape == (4, 4) and be.shape == (4,)
    np.testing.assert_allclose(H, H.T)

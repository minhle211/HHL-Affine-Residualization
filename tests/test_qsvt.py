import math

import numpy as np
import pytest

from data.matrices import MATRIX_FAMILIES, spectrum
from hhl import (
    choose_budget,
    dalzell_queries,
    inverse_poly,
    jennings_queries,
    solve_deflated_qsvt,
    solve_hybrid,
    solve_qsvt,
)
from hhl.cost import aa_rounds, hhl_queries
from hhl.qsvt import clenshaw_apply
from ml.baselines import DiagonalOperator, lanczos_deflation


def spd_from_spectrum(eigs, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(eigs) @ Q.T


def rhs(n, seed=1):
    b = np.random.default_rng(seed).normal(size=n)
    return b / np.linalg.norm(b)


# ---------------------------------------------------------------- polynomial


@pytest.mark.parametrize("kappa", [10, 100, 1000])
@pytest.mark.parametrize("eps", [1e-2, 1e-3])
def test_inverse_poly_accurate_and_bounded(kappa, eps):
    P = inverse_poly(kappa, eps)
    x = np.linspace(1 / kappa, 1, 4001)
    rel = np.abs(P(x) * P.M * x - 1)
    assert rel.max() <= eps
    xs = np.linspace(-1, 1, 20001)
    assert np.abs(P(xs)).max() <= 1.0
    # odd polynomial
    np.testing.assert_allclose(P(-x), -P(x), atol=1e-12)


def test_inverse_poly_degree_scales_like_kappa_log():
    eps = 1e-3
    ratios = [inverse_poly(k, eps).degree / (k * math.log(k / eps)) for k in (10, 100, 1000)]
    assert all(0.5 < r < 3 for r in ratios)
    assert inverse_poly(1000, eps).degree > 5 * inverse_poly(100, eps).degree


def test_trig_evaluation_matches_chebval():
    from numpy.polynomial import chebyshev as C

    P = inverse_poly(300, 1e-3)
    x = np.random.default_rng(0).uniform(-1, 1, 50)
    np.testing.assert_allclose(P(x), C.chebval(x, P.coeffs), atol=1e-12)


def test_surrogate_within_truncation_bound():
    P = inverse_poly(200, 1e-3)
    x = np.linspace(-1, 1, 20001)
    assert np.abs(P(x) - P.surrogate(x)).max() <= P.eps / (2 * P.M) * 1.01


# ---------------------------------------------------------------- QSVT solve


@pytest.mark.parametrize("n", [4, 8, 16])
def test_solve_qsvt_matches_solve(n):
    eigs = np.logspace(-2, 0, n)
    A = spd_from_spectrum(eigs, seed=n)
    b = rhs(n, seed=n + 1)
    eps = 1e-3
    out = solve_qsvt(A, b, eigs.min(), eps)
    x = np.linalg.solve(A, b)
    assert np.linalg.norm(out.y - x) / np.linalg.norm(x) <= eps
    assert 0 < out.p_success <= 1
    assert out.queries_aa == out.degree * (2 * out.aa_rounds + 1)


def test_clenshaw_matches_eig_and_counts_queries():
    eigs = np.array([0.05, 0.2, 0.6, 1.0])
    A = spd_from_spectrum(eigs, seed=3)
    b = rhs(4, seed=5)
    eig = solve_qsvt(A, b, eigs.min(), 1e-3, method="eig")
    cl = solve_qsvt(A, b, eigs.min(), 1e-3, method="clenshaw")
    np.testing.assert_allclose(cl.y, eig.y, atol=1e-10)
    _, matvecs = clenshaw_apply(A, b, eig.poly.coeffs)
    assert matvecs == eig.degree


def test_signed_spectrum():
    eigs = np.array([-1.0, -0.05, 0.02, 0.7])
    A = spd_from_spectrum(eigs, seed=8)
    b = rhs(4, seed=2)
    out = solve_qsvt(A, b, 0.02, 1e-3)
    x = np.linalg.solve(A, b)
    assert np.linalg.norm(out.y - x) / np.linalg.norm(x) <= 1e-3


def test_leakage_on_removed_modes_is_damped():
    # polynomial tuned to kappa_eff = 10 applied to a mode at 1e-4 (outside its domain)
    eigs = np.array([1e-4, 0.1, 0.5, 1.0])
    P = inverse_poly(10, 1e-3)
    amp_poly = abs(P(eigs[0])) * P.M  # effective gain of QSVT (rescaled to 1/x units)
    amp_exact = 1 / eigs[0]
    assert abs(P(eigs[0])) <= 1.0
    assert amp_poly < 1e-2 * amp_exact


# ---------------------------------------------------------------- pipeline


def test_pipeline_qsvt_solver_and_deflation_reduce_degree():
    eigs = np.array([1e-3, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    A = spd_from_spectrum(eigs, seed=4)
    b = rhs(8, seed=6)
    m0 = solve_hybrid(A, b, budget=0, solver="qsvt", eps=1e-3).metrics
    m1 = solve_hybrid(A, b, budget=1, solver="qsvt", eps=1e-3).metrics
    assert m1["degree"] < m0["degree"] / 10
    assert m1["queries_aa"] < m0["queries_aa"]
    assert m0["rel_error"] <= 1e-3 and m1["rel_error"] <= 1e-3


def test_pipeline_hhl_reports_honest_cost():
    A = spd_from_spectrum([0.1, 0.4, 0.7, 1.0], seed=1)
    m = solve_hybrid(A, rhs(4), budget=0, extra_bits=2).metrics
    n = m["n_clock"]
    assert m["u_applications"] == 2 * (2**n - 1)
    assert m["queries_per_run"] >= m["u_applications"]
    assert m["queries_aa"] >= m["queries_per_run"]


def test_hhl_query_count_grows_with_clock():
    q = [hhl_queries(n, 1.0, 1.0, 1e-3)["queries_per_run"] for n in (3, 5, 7)]
    assert q[0] < q[1] < q[2]


def test_aa_rounds():
    assert aa_rounds(1.0) == 0
    assert aa_rounds(0.25) == 1  # one Grover iterate rotates sqrt(p)=0.5 to 1
    assert aa_rounds(1e-4) > aa_rounds(1e-2)


def test_optimal_references():
    assert dalzell_queries(100, 1e-10) < jennings_queries(100, 1e-10)
    # Jennings et al. abstract: ~837 kappa at eps = 1e-10 (alpha = 1, Hermitian)
    assert 830 < jennings_queries(1e4, 1e-10) / 1e4 < 870


# ---------------------------------------------------------------- Lanczos


def test_lanczos_deflation_recovers_exact_smallest_modes():
    eigs = np.concatenate([[1e-3, 2e-3], np.linspace(0.3, 1.0, 14)])
    A = spd_from_spectrum(eigs, seed=9)
    b = rhs(16, seed=10)
    d = lanczos_deflation(A, b, 2, tol=1e-8)
    lam, U = np.linalg.eigh(A)
    z_exact = U[:, :2] @ ((U[:, :2].T @ b) / lam[:2])
    assert d.converged
    assert 0 < d.matvecs <= 16
    np.testing.assert_allclose(np.sort(d.removed_ritz), lam[:2], rtol=1e-6)
    np.testing.assert_allclose(d.z, z_exact, rtol=1e-5, atol=1e-8)
    assert d.next_ritz <= lam[2] * (1 + 1e-8)


def test_lanczos_on_diagonal_operator_matches_dense():
    eigs = np.array([1e-3, 0.01, 0.2, 0.5, 0.7, 1.0])
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    b = rhs(6, seed=3)
    dense = lanczos_deflation(Q @ np.diag(eigs) @ Q.T, b, 1, tol=1e-6)
    diag = lanczos_deflation(DiagonalOperator(eigs), Q.T @ b, 1, tol=1e-6)
    assert dense.matvecs == diag.matvecs
    np.testing.assert_allclose(dense.z, Q @ diag.z, atol=1e-10)


# ---------------------------------------------------------------- adaptive budget


@pytest.mark.parametrize("w", [0.01, 1.0, 1e3])
def test_adaptive_budget_never_estimated_worse_than_plain_qsvt(w):
    eigs = np.concatenate([[1e-3, 3e-3], np.logspace(-1.5, 0, 14)])
    A = spd_from_spectrum(eigs, seed=0)
    b = rhs(16, seed=0)
    ch = choose_budget(A, b, 1e-3, w)
    assert ch.est_total[ch.k] <= ch.est_total[0]


def test_deflated_qsvt_beats_plain_on_separated_small_modes():
    eigs = np.concatenate([[1e-3, 3e-3], np.logspace(-1.5, 0, 14)])
    A = spd_from_spectrum(eigs, seed=0)
    b = rhs(16, seed=0)
    plain = solve_deflated_qsvt(A, b, eps=1e-3, budget=0).metrics
    adaptive = solve_deflated_qsvt(A, b, eps=1e-3, w=0.1).metrics
    assert adaptive["budget"] >= 2
    assert adaptive["total_cost"] < plain["total_cost"] / 10
    assert adaptive["rel_error"] <= 1e-3


def test_fixed_budget_deflated_qsvt_accurate():
    eigs = np.concatenate([[1e-3], np.linspace(0.2, 1.0, 7)])
    A = spd_from_spectrum(eigs, seed=2)
    b = rhs(8, seed=4)
    m = solve_deflated_qsvt(A, b, eps=1e-3, budget=1).metrics
    assert m["budget"] == 1
    assert m["kappa_eff"] < 10
    assert m["rel_error"] <= 1e-3


# ---------------------------------------------------------------- matrices


@pytest.mark.parametrize("family", list(MATRIX_FAMILIES))
def test_matrix_families_spd_and_spectrum_consistent(family):
    rng = np.random.default_rng(0)
    A = MATRIX_FAMILIES[family](16, 100.0, rng)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    lam = np.linalg.eigvalsh(A)
    assert lam.min() > 0
    s = spectrum(family, 16, 100.0, np.random.default_rng(0))
    assert len(s) == 16 and s.min() > 0
    np.testing.assert_allclose(np.sort(s), lam, rtol=1e-8, atol=1e-10)
    if family in ("clustered_small", "log_uniform", "sparse_banded"):
        np.testing.assert_allclose(lam.max() / lam.min(), 100.0, rtol=1e-6)

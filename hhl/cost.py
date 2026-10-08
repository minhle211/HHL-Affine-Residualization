"""Query-cost model for quantum linear solvers.

The unit of cost is one call to a block-encoding ``U_A`` of ``A / alpha``
(``alpha >= ||A||``), the unit used by modern QLSA papers. HHL's phase
estimation calls ``exp(i A t 2^j)``; each such evolution is converted to
block-encoding queries with the near-optimal QSP Hamiltonian-simulation cost
``ceil(e * alpha * tau / 2 + ln(1 / eps_sim))`` (Low & Chuang 2017/2019).

Amplitude amplification (AA) boosts a success probability ``p`` to ~1 with
``r = ceil(pi / (4 asin(sqrt(p))) - 1/2)`` rounds; each round calls the
underlying algorithm and its inverse once, so the cost is ``(2r + 1)`` runs.
"""

from __future__ import annotations

import math


def aa_rounds(p: float) -> int:
    """Amplitude-amplification rounds needed to boost success probability ``p``."""
    if p <= 0:
        raise ValueError("success probability must be positive")
    if p >= 1:
        return 0
    return max(0, math.ceil(math.pi / (4 * math.asin(math.sqrt(p))) - 0.5))


def aa_cost(queries_per_run: float, p: float) -> float:
    """Queries to obtain the postselected state with probability ~1 via AA."""
    return queries_per_run * (2 * aa_rounds(p) + 1)


def repeat_cost(queries_per_run: float, p: float) -> float:
    """Expected queries with plain repeat-until-success postselection."""
    return queries_per_run / p


def hamiltonian_sim_queries(alpha: float, tau: float, eps_sim: float) -> int:
    """Block-encoding queries to implement ``exp(i A tau)`` to error ``eps_sim``."""
    return math.ceil(math.e * alpha * abs(tau) / 2 + math.log(1 / eps_sim))


def hhl_queries(n_clock: int, t: float, alpha: float, eps: float) -> dict:
    """Honest query count of one HHL run (QPE + inverse QPE).

    QPE applies controlled ``exp(i A t 2^j)`` for ``j = 0..n-1``; the
    simulation error budget ``eps`` is split evenly over the ``2n`` calls.
    """
    eps_sim = eps / (2 * n_clock)
    per_qpe = sum(hamiltonian_sim_queries(alpha, t * 2**j, eps_sim) for j in range(n_clock))
    return {
        "u_applications": 2 * (2**n_clock - 1),
        "queries_per_run": 2 * per_qpe,
        "inversion_rotations": 2**n_clock,
    }


def hhl_cost_metrics(n_clock: int, t: float, alpha: float, eps: float, p_success: float) -> dict:
    """Per-run and success-adjusted HHL query costs (AA and repeat-until-success)."""
    q = hhl_queries(n_clock, t, alpha, eps)
    p = max(p_success, 1e-300)
    return {
        "u_applications": q["u_applications"],
        "inversion_rotations": q["inversion_rotations"],
        "queries_per_run": q["queries_per_run"],
        "queries_aa": aa_cost(q["queries_per_run"], p),
        "queries_repeat": repeat_cost(q["queries_per_run"], p),
        "aa_rounds": aa_rounds(p),
    }

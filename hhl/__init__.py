"""Hybrid affine-residualized HHL (simulator-scale research code)."""

from .circuit import (
    HHLParams,
    HHLResult,
    build_hhl_circuit,
    choose_hhl_params,
    circuit_resources,
    fidelity,
    run_hhl,
)
from .pipeline import (
    BudgetChoice,
    HybridResult,
    choose_budget,
    solve_deflated_qsvt,
    solve_hybrid,
    solve_standard,
)
from .qsvt import (
    QSVTResult,
    dalzell_queries,
    inverse_poly,
    jennings_queries,
    optimal_reference_queries,
    solve_qsvt,
)
from .residualization import (
    Residualization,
    budgeted_affine_residualization,
    hard_projector,
    hermitian_embedding,
    pad_to_power_of_two,
)

__all__ = [
    "BudgetChoice",
    "HHLParams",
    "HHLResult",
    "HybridResult",
    "QSVTResult",
    "Residualization",
    "budgeted_affine_residualization",
    "build_hhl_circuit",
    "choose_budget",
    "choose_hhl_params",
    "circuit_resources",
    "dalzell_queries",
    "fidelity",
    "hard_projector",
    "hermitian_embedding",
    "inverse_poly",
    "jennings_queries",
    "optimal_reference_queries",
    "pad_to_power_of_two",
    "run_hhl",
    "solve_deflated_qsvt",
    "solve_hybrid",
    "solve_qsvt",
    "solve_standard",
]

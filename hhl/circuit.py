"""Custom HHL circuit for Qiskit 1.x / 2.x.

``qiskit.algorithms.HHL`` was removed in Qiskit 1.0, so this module builds the
circuit explicitly:

1. amplitude-encode the normalized right-hand side ``|b>``;
2. quantum phase estimation (QPE) with exact ``UnitaryGate`` powers of
   ``U = exp(i A t)``;
3. eigenvalue inversion: a uniformly controlled RY rotation of an ancilla by
   ``2 * arcsin(C / lambda_k)`` for every clock basis state ``k``;
4. inverse QPE (uncomputation);
5. exact statevector simulation and postselection on ``ancilla = 1`` (and
   ``clock = 0``).

All QPE parameters (clock qubits, evolution time ``t``, rotation constant
``C``) are derived from an *explicit eigenvalue range* that the caller passes
in. That is what lets the hybrid pipeline tune the circuit to the remaining
(deflated) spectrum instead of the full one.

Qubit layout (little endian, qubit 0 is the least significant bit):
``[system (m qubits) | clock (n qubits) | ancilla (1 qubit)]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, StatePreparation, UCRYGate, UnitaryGate
from qiskit.quantum_info import Statevector

# Fraction of the (half-)phase range occupied by the largest |eigenvalue|.
# The headroom keeps QPE leakage around lambda_max from wrapping around to
# k ~ 0, where it would be misread as a tiny eigenvalue and inverted.
_PHASE_HEADROOM = 0.75


@dataclass
class HHLParams:
    """QPE / inversion parameters derived from an eigenvalue range."""

    n_clock: int
    t: float
    C: float
    lam_min: float  # smallest |eigenvalue| the circuit is tuned to resolve
    lam_max: float  # largest |eigenvalue| the circuit is tuned to resolve
    signed: bool  # True if negative eigenvalues must be represented
    clock_capped: bool = False  # True if n_clock hit max_clock

    @property
    def kappa(self) -> float:
        return self.lam_max / self.lam_min

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kappa"] = self.kappa
        return d


def choose_hhl_params(
    lam_min: float,
    lam_max: float,
    signed: bool = False,
    extra_bits: int = 2,
    n_clock: int | None = None,
    max_clock: int = 12,
    c_factor: float = 1.0,
) -> HHLParams:
    """Choose ``n_clock``, ``t`` and ``C`` from an eigenvalue magnitude range.

    ``lam_max`` is mapped to the clock integer ``K = 0.75 * 2^n`` (or
    ``0.75 * 2^(n-1)`` when signed eigenvalues use two's complement), so
    ``lam_min`` lands near ``K / kappa``. ``extra_bits`` adds resolution below
    that. ``C = c_factor * lam_min`` maximizes the postselection probability
    for the targeted range; estimates below ``C`` are clipped to a full
    rotation.
    """
    lam_min = float(abs(lam_min))
    lam_max = float(abs(lam_max))
    if lam_min <= 0 or lam_max <= 0 or lam_min > lam_max:
        raise ValueError(f"invalid eigenvalue range [{lam_min}, {lam_max}]")
    kappa = lam_max / lam_min
    capped = False
    if n_clock is None:
        n_clock = math.ceil(math.log2(kappa / _PHASE_HEADROOM)) + extra_bits + int(signed)
        n_clock = max(n_clock, 2 + int(signed))
        if n_clock > max_clock:
            n_clock, capped = max_clock, True
    half = 2 ** (n_clock - 1) if signed else 2**n_clock
    k_max = _PHASE_HEADROOM * half
    t = 2 * math.pi * k_max / (2**n_clock * lam_max)
    return HHLParams(
        n_clock=n_clock,
        t=t,
        C=c_factor * lam_min,
        lam_min=lam_min,
        lam_max=lam_max,
        signed=signed,
        clock_capped=capped,
    )


def clock_eigenvalues(params: HHLParams) -> np.ndarray:
    """Eigenvalue estimate associated with each clock basis state ``k``."""
    n = params.n_clock
    k = np.arange(2**n, dtype=float)
    if params.signed:
        k = np.where(k >= 2 ** (n - 1), k - 2**n, k)
    return 2 * math.pi * k / (params.t * 2**n)


def inversion_angles(params: HHLParams) -> list[float]:
    """RY angles ``2 arcsin(C / lambda_k)`` (clipped), 0 for ``k = 0``."""
    lam = clock_eigenvalues(params)
    angles = np.zeros_like(lam)
    nz = lam != 0
    ratio = np.clip(params.C / lam[nz], -1.0, 1.0)
    angles[nz] = 2 * np.arcsin(ratio)
    return angles.tolist()


def _check_hermitian(A: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError(
            "HHL requires a Hermitian A; use hhl.residualization.hermitian_embedding"
        )
    n = A.shape[0]
    if n & (n - 1):
        raise ValueError("dimension must be a power of two; pad first")


def build_hhl_circuit(A: np.ndarray, b: np.ndarray, params: HHLParams) -> QuantumCircuit:
    """Build the full HHL circuit for Hermitian ``A`` and right-hand side ``b``."""
    A = np.asarray(A)
    _check_hermitian(A)
    b = np.asarray(b, dtype=complex)
    nb = np.linalg.norm(b)
    if nb == 0:
        raise ValueError("b must be non-zero")
    m = int(round(math.log2(A.shape[0])))
    n = params.n_clock

    sys_q = QuantumRegister(m, "sys")
    clk_q = QuantumRegister(n, "clk")
    anc_q = QuantumRegister(1, "anc")
    qc = QuantumCircuit(sys_q, clk_q, anc_q, name="HHL")

    qc.append(StatePreparation(b / nb), sys_q)

    qpe = QuantumCircuit(sys_q, clk_q, name="QPE")
    qpe.h(clk_q)
    for j in range(n):
        U = expm(1j * A * params.t * 2**j)
        cu = UnitaryGate(U, label=f"U^{2**j}").control(1)
        qpe.append(cu, [clk_q[j], *sys_q])
    qpe.append(QFTGate(n).inverse(), clk_q)

    qc.compose(qpe, qubits=[*sys_q, *clk_q], inplace=True)
    qc.append(UCRYGate(inversion_angles(params)), [anc_q[0], *clk_q])
    qc.compose(qpe.inverse(), qubits=[*sys_q, *clk_q], inplace=True)
    return qc


@dataclass
class HHLResult:
    y: np.ndarray  # estimate of A^{-1} b_hat for normalized b_hat (with magnitude)
    p_success: float  # P(ancilla = 1)
    p_success_clock0: float  # P(ancilla = 1 and clock = 0)
    params: HHLParams
    circuit: QuantumCircuit
    resources: dict | None = None


def circuit_resources(qc: QuantumCircuit, optimization_level: int = 1) -> dict:
    """Transpile to {u, cx} and report width, depth and CX count."""
    tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=optimization_level)
    ops = tqc.count_ops()
    return {
        "num_qubits": tqc.num_qubits,
        "depth": tqc.depth(),
        "cx": int(ops.get("cx", 0)),
        "u": int(ops.get("u", 0)),
    }


def run_hhl(
    A: np.ndarray,
    b: np.ndarray,
    params: HHLParams,
    compute_resources: bool = False,
) -> HHLResult:
    """Simulate HHL exactly with ``Statevector`` and postselect.

    The postselected amplitude on ``(ancilla=1, clock=0)`` equals
    ``C * A^{-1} b_hat`` in the ideal case, so dividing by ``C`` recovers the
    solution *including its magnitude* for the normalized input ``b_hat``. On
    hardware the magnitude would be estimated from ``p_success`` instead.
    """
    qc = build_hhl_circuit(A, b, params)
    sv = Statevector(qc).data
    m = int(round(math.log2(A.shape[0])))
    n = params.n_clock
    dim_sys = 2**m
    anc_offset = 2 ** (m + n)
    anc1 = sv[anc_offset:]
    p_success = float(np.sum(np.abs(anc1) ** 2))
    amp = anc1[:dim_sys]  # clock = 0
    p0 = float(np.sum(np.abs(amp) ** 2))
    y = amp / params.C
    res = circuit_resources(qc) if compute_resources else None
    return HHLResult(
        y=y,
        p_success=p_success,
        p_success_clock0=p0,
        params=params,
        circuit=qc,
        resources=res,
    )


def fidelity(x_est: np.ndarray, x_true: np.ndarray) -> float:
    """State fidelity |<x_est|x_true>|^2 / (||x_est||^2 ||x_true||^2)."""
    ne, nt = np.linalg.norm(x_est), np.linalg.norm(x_true)
    if ne == 0 or nt == 0:
        return float(ne == nt)
    return float(abs(np.vdot(x_est, x_true)) ** 2 / (ne**2 * nt**2))

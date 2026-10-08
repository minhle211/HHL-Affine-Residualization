# Hybrid Affine-Residualized HHL

A hybrid solver for small linear systems (`N` = 2, 4, or 8). A classical step removes the hardest eigenmodes of `A`. An HHL circuit, tuned only to what remains, solves the residual.

## Features

- **Budgeted deflation.** Remove `k` eigenmodes classically and leave a smaller residual for the quantum step.
- **Retuned HHL.** Clock qubits, evolution time, and the rotation constant come from the remaining spectrum, so the circuit shrinks when that spectrum is better conditioned.
- **Two circuit modes.** `reduced` keeps those savings and cannot fix errors in the classical part. `full` can fix those errors, using a circuit sized for the original spectrum.
- **Choice of classical estimate.** The removed part can be exact, or come from Lanczos, inverse iteration, randomized SVD, or a small neural net.
- **QSVT option.** The residual can be inverted with an emulated QSVT polynomial instead of HHL.
- **Resource and error readout.** Clock qubits, circuit depth, CX count, success probability, fidelity, and relative error.

## How it works

For a Hermitian matrix `A` and right-hand side `b`:

1. **Pick the hard modes.** Choose `k` modes, either the ones that contribute most to the solution or the ones with the smallest eigenvalues.
2. **Explain them classically.** Build a vector `z` from those modes. This is the part of the solution the quantum circuit does not compute.
3. **Form the residual.** Subtract `A z` from `b`, then drop any leftover component on the removed modes so tiny numerical noise is not amplified.
4. **Tune the circuit.** Set the phase-estimation parameters from the eigenvalues still in play. A smaller remaining condition number means fewer clock qubits.
5. **Solve and recombine.** Run HHL (or QSVT) on the normalized residual to get `y`. The solution is `x = z + ||b_res|| * y`.

A non-Hermitian `A` is rewritten as a larger Hermitian block matrix with a signed spectrum. Sizes that are not powers of two are padded.

If `z` is only approximate, the reduced circuit leaves that error in the answer. The full circuit sees the error in the residual and can correct it, at the cost of the original circuit size.

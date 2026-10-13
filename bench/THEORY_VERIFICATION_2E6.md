# Exact-arithmetic verification, two million systems

Date of the run: 13 October 2026.

This document records a larger check than `bench/theory_scale_log.csv` (one million systems, sizes 2, 4, and 8, budget 1). The new log is `bench/theory_scale_log_2e6.csv`. The checker is `bench/theory_scale.py`. The residual solver is `numpy.linalg.solve`. No HHL circuit is run.

## What was drawn

Fourteen streams. Each stream is one random sequence. The one-million row is the first half of the two-million row.

- Sizes: \(N = 2, 4, 8, 16\). Size 16 was not in the earlier million-draw log.
- Budgets: \(k = 1\), and \(k = 2\) when \(N > 2\).
- Rankings: `smallest` and `energy`.
- Spectrum: log-uniform, with \(\kappa\) log-uniform in \([10, 10^4]\). Eigenvalues are positive.
- Samples per stream: \(2 \times 10^6\). Across the fourteen streams that is \(2.8 \times 10^7\) systems.

## What this verifies

These checks pass on every system in the two-million rows.

| Check | Result at \(2 \times 10^6\) |
|---|---|
| Recombination \(\|z + A^{-1} b_{\mathrm{res}} - x\| / \|x\|\) | At most \(5.9 \times 10^{-12}\) |
| Exact split stays at or under \(\sqrt{2}\) | 0 systems exceed \(\sqrt{2}\) by more than \(10^{-6}\) |
| \(\kappa_{\mathrm{eff}} > \kappa\) | 0 systems |
| `smallest` matches \(\kappa_{\mathrm{eff}} = \|\lambda\|_{\max} / \|\lambda\|_{k+1}\) | Discrepancy 0 on every such system |
| 10% hard-mode error versus \(\|P_{\mathrm{hard}} \delta\| / \|x\|\) | At most \(5.9 \times 10^{-12}\) |
| Same error, raw residual, exact solve | At most \(1.5 \times 10^{-12}\) |
| Shift of \(40\|x\|\) along a kept mode | Exceeds \(\sqrt{2}\) on every system |

The pass is for the exact eigen-split used by the residualization code: recombination, orthogonality of that split, the condition-number formula when the removed modes are the smallest magnitudes, and the two residual modes for an inexact \(z\).

## What this does not verify

It does not verify the HHL circuit, phase estimation, a quantum speedup, or a wall-clock advantage over a dense solve. It does not say that the energy ranking reduces \(\kappa\). A non-orthogonal \(z\) is outside the \(\sqrt{2}\) ceiling, and every constructed shift of that kind exceeded it.

## Condition number left almost unchanged

The count below is the number of systems with \(\kappa_{\mathrm{eff}} > 0.99\,\kappa\), at two million draws.

| \(N\) | \(k\) | smallest | energy |
|---:|---:|---:|---:|
| 2 | 1 | 0 (0%) | 0 (0%) |
| 4 | 1 | 8,152 (0.41%) | 381,823 (19.1%) |
| 4 | 2 | 8 (0.0004%) | 74,534 (3.73%) |
| 8 | 1 | 23,956 (1.20%) | 738,912 (36.9%) |
| 8 | 2 | 138 (0.0069%) | 338,971 (16.9%) |
| 16 | 1 | 56,014 (2.80%) | 1,074,123 (53.7%) |
| 16 | 2 | 834 (0.042%) | 672,049 (33.6%) |

The one-million prefixes of the \(k = 1\) streams for \(N = 4\) and \(N = 8\) match the rates already recorded in the million-draw log: about 19% and 37% for the energy ranking. Doubling the sample does not move those rates. At \(N = 16\) and \(k = 1\), the energy ranking leaves \(\kappa_{\mathrm{eff}}\) within 1% of \(\kappa\) on more than half of the systems. Removing the smallest magnitudes still does that only when the next eigenvalue is almost as small, and that fraction grows slowly with \(N\) (0% at \(N = 2\), 0.41% at \(N = 4\), 1.20% at \(N = 8\), 2.80% at \(N = 16\)).

## Verdict

The exact residualization identities hold at this larger size. The energy ranking is still not a condition-number reduction. The quantum solver is not part of this test.

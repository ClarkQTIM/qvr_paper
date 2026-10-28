# Results — 2026-10-14 — MIT-BIH (first registered run)

Dated record; do not edit after commit. Run: `prereg-v1`, full grid, 5 final seeds,
32 workers. Source: `results/mitbih/summary.md`. Deviations: `docs/DEVIATIONS.md`
(D1–D4).

## Scorecard

| ID | Prediction | Result |
|---|---|---|
| P1 | Fourier structure exact on every trained model | **holds**: 15/15 models; min R² ⟨Z⟩ @K = 1.0000000000, max @K−1 = 0.868, min R² score @2K = 1.0000000000 |
| P2 | Damping formula matches simulation | **holds** |
| P3 | Tuned QVR does not beat the best classical method | **fails** (D1): val-selected PCA Δ +0.022 [−0.079, +0.146]; test-best Mahalanobis Δ +0.086 [+0.010, +0.216] |
| P4 | trig_t (QVR's class, classically) ≥ QVR − 0.02 | **holds**: 0.732 vs 0.690 |
| P5 | D = 0 within 0.05 of QVR | **holds**: 0.660 vs 0.690 |
| P6 | Learning σ does not help | **holds** (weakly): σ0 − QVR −0.024 [−0.135, +0.080] |
| P7 | A better readout of the same circuit helps | **holds**: 0.786 vs 0.690 |
| P8 | p̄_b ≤ six moments + 0.02 | **holds**: 0.813 vs 0.815 |
| P9 | Null guard ≤ 0.10 | **fails** as registered (0.118); **holds** calibrated (16th percentile; D2) |

## Main numbers (pooled test AUC, 95% unit CI, 22 test records)

| Method | AUC |
|---|---|
| QVR circuit, mean-⟨Z⟩ Mahalanobis readout | 0.786 [0.646, 0.899] |
| Mahalanobis (whole beat) | 0.775 [0.592, 0.913] |
| Isolation Forest / kNN | 0.742 / 0.734 |
| QVR, scored with D = μ | 0.734 [0.561, 0.893] |
| per-timepoint trig (QVR's class) | 0.732 |
| per-timepoint z² | 0.722 |
| six moments | 0.717 |
| **QVR tuned (native score)** | **0.690 [0.455, 0.881]** |
| QVR σ = 0 trained | 0.666 |
| QVR D = 0 (cos x) | 0.660 |
| QVR reference config | 0.569 |

Selected QVR: 5 layers, N_E = 10, cost inside, ry_all, lr 0.003 (val AUC 0.708).
Seeds agree closely (0.692 ± 0.005).

## Readings

1. **The stochastic draws cost performance.** σ did not move from its random
   initialisation (3.15 → 3.24). At that size, the draws remove nearly all
   time-dependent terms by late timepoints (median coherence damping ≈ 0).
   Scoring the same models with D = μ raises AUC from 0.692 to 0.732 (seed
   means). Training with σ fixed at 0 is about the same (0.665) and stops earlier
   (best batch 4,610 vs 5,350).
2. **The circuit's features are informative; the native score discards them.**
   The best row in the table is a classical readout of the trained circuit's
   ⟨Z⟩ (0.786), against 0.690 for the native score.
3. **No use of order.** Time-permuting the test beats changes QVR AUC by 0.009
   (Fact 2).
4. **The fingerprint is the moments.** p̄_b 0.808–0.821 over seeds vs six moments
   0.815 vs raw beat 0.945; the earlier result reproduces to the third decimal.
5. **Supraventricular beats sit below 0.5 for most methods.** Single-beat windows
   carry no timing (prematurity) information; expected.
6. **Patient heterogeneity dominates the uncertainty.** With 22 test records, the
   intervals are about 0.4 wide and the null guard's typical chance deviation is about 0.16
   (D2). Differences between methods of less than about 0.1 cannot be resolved on
   MIT-BIH alone.

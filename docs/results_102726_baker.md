# Results — 2026-10-27 — Baker (crypto, normals-only scaling)

Dated record; do not edit after commit. Run: `prereg-v1` code with report
amendments D1–D3; full multichannel grid (48 configs × 2 seeds); 5 final seeds;
32 workers. Source: `results/baker/summary.md`.

Data: `prepare.py baker` from `bi_train782` (rebuilt by
`forensics/baker_build_sets.py`; `bi_train100` equals the release). Channels: open,
volume (2 qubits, one channel each). Global per-feature min–max refit on 665
training normals; 117 validation normals carved as one contiguous block; released
Xval (60 anomalies); test: 60 normals and six anomaly sets of 19–60 windows.
**Windows overlap and have no unit ids, so CIs are per-window and optimistic**
(stated in the registration).

## Scorecard

| ID | Prediction | Result |
|---|---|---|
| P1 | Fourier structure exact on every trained model | **holds**: 15/15; per-channel degree K = 1; min R² ⟨Z⟩ @K = 1.0000000000; score @2K = 1.0000000000 |
| P2 | Damping matches simulation | **holds** |
| P3 | Tuned QVR does not beat the best classical method | **fails**: QVR ahead of val-selected PCA by 0.038 [−0.002, +0.078]; ahead of test-best kNN by 0.016 [−0.016, +0.050] |
| P4 | trig_t ≥ QVR − 0.02 | **fails**: 0.696 vs 0.729 |
| P5 | D = 0 within 0.05 of QVR | **holds**: D = 0 is better, 0.764 vs 0.729 (Δ +0.035 [+0.002, +0.074]) |
| P6 | Learning σ does not help | **fails** (borderline): σ0 − QVR −0.024 [−0.047, −0.001]; seed SD 0.03–0.05 |
| P7 | A better readout of the same circuit helps | **fails**: best readout 0.628 vs native 0.729 |
| P8 | p̄_b ≤ moments + 0.02 | not applicable (no subtypes) |
| P9 | Null guard ≤ 0.10 | **fails** as registered (0.107, 60 units); **holds** calibrated (44th percentile; median 0.114, 95th 0.173) |

## Main numbers (pooled test AUC, 95% per-window CI)

| Method | AUC |
|---|---|
| **QVR with D = 0 (cos x; trained W cancels)** | **0.764 [0.699, 0.825]** |
| QVR, D = μ | 0.735 |
| **QVR tuned (native score)** | **0.729 [0.658, 0.791]** |
| kNN (best classical) | 0.713 |
| QVR reference config | 0.706 |
| QVR σ = 0 trained | 0.705 |
| one number per series | 0.698 |
| trig_t / Mahalanobis / PCA / Isolation Forest | 0.696 / 0.692 / 0.691 / 0.691 |
| six moments / temporal / per-timepoint z² | 0.665 / 0.661 / 0.652 |
| QVR circuit readouts (tz2_z / p̄_b / mean ⟨Z⟩) | 0.628 / 0.623 / 0.568 |

Selected QVR: 3 layers, N_E = 10, cost outside, rx_per_qubit, StronglyEntanglingLayers
ansatz, lr 0.003 (val AUC 0.682). Per-seed AUC 0.711 ± 0.039.

## Readings

1. **The best detector is QVR's classical reduction.** With D = 0 the trained W
   cancels and ⟨Z_q⟩ = cos x_q. The score is
   mean_t (η₀ − mean_q cos x_q,t)² / 4, a fixed closed-form function of the scaled
   series. It gives 0.763 ± 0.000 for every seed and for the reference config, so
   training does not affect it. It tops the table and beats the full model. This
   replicates the earlier Baker audit ("identity = full model") under the
   registered protocol, and here it is strictly better.
2. **Removing or scrambling the time dependence helps.** D = 0 gives 0.763 and
   time-permuted input gives 0.760, against 0.711 for the model as trained
   (seed means). On Baker, the circuit's time-dependent phases are a liability.
3. **σ shrank during training** (2.84 → 1.42 best; → 0.72 reference). The small
   σ = 0 deficit (P6) is borderline and comes with no mechanistic explanation; a
   hypothesis that σ helps by erasing time dependence is not supported (σ did not
   grow).
4. **Global beats per-timepoint on Baker.** Methods that fit separate statistics
   at each of the 180 timepoints (trig_t, tz2, the QVR readouts) are among the
   weakest. QVR's native score and its D = 0 form use one global centre η₀ across
   time. This is the opposite of MIT-BIH, where per-timepoint detail helped. P4
   and P7 therefore fail on estimator choice, not on the function-class claim.
5. **QVR beats the generic classical detectors here** (P3 fails), unlike
   MIT-BIH. The advantage comes entirely from the classical cos x transform with a
   single global centre.

## Cross-dataset reading so far (MIT-BIH, Baker)

- The Fourier structure is exact everywhere (P1 on 30/30 models).
- QVR's behaviour is explained by its classical Fourier reduction: where it loses
  (MIT-BIH), generic detectors beat it; where it wins (Baker), its D = 0
  reduction wins by more.
- The quantum time evolution never helps detection: D = μ ≥ MC draws on both,
  D = 0 better on Baker, and order is unused (MIT-BIH) or harmful (Baker).

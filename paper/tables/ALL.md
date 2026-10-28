# T1_main

Pooled test AUC [95% unit-level CI]. QVR rows: rank-averaged over seeds.

| method | mitbih |
|---|---|
| QVR (tuned) | 0.690 [0.455, 0.881] |
| QVR (reference config) | 0.569 [0.466, 0.673] |
| QVR, D = 0 (cos x) | 0.660 [0.394, 0.870] |
| QVR circuit, per-t z² readout | 0.710 [0.450, 0.909] |
| QVR circuit, p̄_b Mahalanobis | 0.711 [0.509, 0.881] |
| one number per series | 0.669 [0.579, 0.753] |
| one temporal number (lag-1 / MSD) | 0.546 [0.334, 0.763] |
| per-timepoint z² | 0.722 [0.496, 0.897] |
| per-timepoint trig (QVR class) | 0.732 [0.507, 0.903] |
| six moments (fingerprint) | 0.717 [0.522, 0.885] |
| Mahalanobis (whole series) | 0.775 [0.592, 0.913] |
| PCA reconstruction | 0.712 [0.512, 0.871] |
| kNN distance | 0.734 [0.503, 0.914] |
| Isolation Forest | 0.742 [0.513, 0.916] |

# T2_dmodes

Pooled test AUC, mean ± SD over seeds.

| tuned QVR, scored with | mitbih |
|---|---|
| MC draws (as trained) | 0.692 ± 0.005 |
| D = μ | 0.732 ± 0.056 |
| D = 0 | 0.660 ± 0.002 |
| averaged D̄ | 0.666 ± 0.012 |
| σ = 0 during training | 0.665 ± 0.030 |
| reference config (MC) | 0.568 ± 0.021 |
| mean |σ| final (init) | 3.24 (3.15) |

# T3_readouts

| readout of the same trained circuit | mitbih |
|---|---|
| QVR (tuned) | 0.690 |
| QVR circuit, per-t z² readout | 0.710 |
| QVR circuit, mean-⟨Z⟩ Mahalanobis | 0.786 |
| QVR circuit, p̄_b Mahalanobis | 0.711 |

# T4_fourier

Across every trained model (best, ref, σ0; all seeds) and every D mode.

| dataset | K | models where all predictions hold | min R² ⟨Z⟩ @K | max R² ⟨Z⟩ @K−1 | min R² score @2K | median damping, last t |
|---|---|---|---|---|---|---|
| mitbih | [3] | 100% | 1.0000000000 | 0.8679 | 1.0000000000 | 0 |

# T5_probes

Macro AUC of a logistic probe fit on anomalies from training units, evaluated on test anomalies; p = unit-level permutation null.

| dataset | raw series | six moments | QVR p̄_b (mean over seeds) | note |
|---|---|---|---|---|
| mitbih | 0.945 (p 0.005) | 0.815 (p 0.050) | 0.813 |  |

# T6_protocol

| dataset | null guard: max deviation of AUC from 0.5 | units | QVR AUC, time-permuted | ΔAUC (orig − permuted) |
|---|---|---|---|---|
| mitbih | 0.118 | 22 | 0.683 | 0.009 |

# T7_predictions

Registered predictions (docs/REGISTRATION.md), definitions per docs/DEVIATIONS.md.

| prediction | mitbih |
|---|---|
| P1 | holds: 1.0 |
| P2 | holds: part of P1 verdicts |
| P3 | FAILS: pca (val-selected): Δ +0.022 [-0.079, +0.146] |
| P4 | holds: trig_t 0.732 vs QVR 0.690 |
| P5 | holds: identity 0.660 vs QVR 0.690 |
| P6 | holds: σ0 − QVR -0.024 [-0.135, +0.080] |
| P7 | holds: best readout 0.786 vs QVR 0.690 |
| P8 | holds: p̄_b 0.813 vs moments 0.815 |
| P9 | FAILS: max deviation 0.118 (22 units) (calibrated: holds, percentile 16% of 200 unit splits (median 0.162, 95th 0.273)) |

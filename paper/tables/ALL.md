# T1_main

Pooled test AUC [95% unit-level CI]. QVR rows: rank-averaged over seeds.

| method | baker | gw | mitbih |
|---|---|---|---|
| QVR (tuned) | 0.729 [0.658, 0.791] | 0.721 [0.629, 0.809] | 0.690 [0.455, 0.881] |
| QVR (reference config) | 0.706 [0.636, 0.768] | 0.424 [0.328, 0.531] | 0.569 [0.466, 0.673] |
| QVR, D = 0 (cos x) | 0.764 [0.699, 0.825] | 0.743 [0.658, 0.827] | 0.660 [0.394, 0.870] |
| QVR circuit, per-t z² readout | 0.628 [0.560, 0.693] | 0.817 [0.759, 0.885] | 0.710 [0.450, 0.909] |
| QVR circuit, p̄_b Mahalanobis | 0.623 [0.557, 0.684] | 0.634 [0.525, 0.741] | 0.711 [0.509, 0.881] |
| one number per series | 0.698 [0.630, 0.761] | 0.663 [0.544, 0.778] | 0.669 [0.579, 0.753] |
| one temporal number (lag-1 / MSD) | 0.661 [0.591, 0.735] | 0.516 [0.396, 0.647] | 0.546 [0.334, 0.763] |
| per-timepoint z² | 0.652 [0.585, 0.715] | 0.790 [0.726, 0.859] | 0.722 [0.496, 0.897] |
| per-timepoint trig (QVR class) | 0.696 [0.625, 0.758] | 0.774 [0.707, 0.846] | 0.732 [0.507, 0.903] |
| six moments (fingerprint) | 0.665 [0.592, 0.730] | 0.609 [0.489, 0.734] | 0.717 [0.522, 0.885] |
| Mahalanobis (whole series) | 0.692 [0.615, 0.760] | 0.798 [0.730, 0.868] | 0.775 [0.592, 0.913] |
| PCA reconstruction | 0.691 [0.614, 0.763] | 0.781 [0.714, 0.853] | 0.712 [0.512, 0.871] |
| kNN distance | 0.713 [0.639, 0.779] | 0.786 [0.712, 0.852] | 0.734 [0.503, 0.914] |
| Isolation Forest | 0.691 [0.620, 0.755] | 0.743 [0.660, 0.822] | 0.742 [0.513, 0.916] |

# T2_dmodes

Pooled test AUC, mean ± SD over seeds.

| tuned QVR, scored with | baker | gw | mitbih |
|---|---|---|---|
| MC draws (as trained) | 0.711 ± 0.039 | 0.709 ± 0.022 | 0.692 ± 0.005 |
| D = μ | 0.716 ± 0.051 | 0.608 ± 0.070 | 0.732 ± 0.056 |
| D = 0 | 0.763 ± 0.000 | 0.743 ± 0.018 | 0.660 ± 0.002 |
| averaged D̄ | 0.700 ± 0.052 | 0.608 ± 0.046 | 0.666 ± 0.012 |
| σ = 0 during training | 0.698 ± 0.026 | 0.562 ± 0.024 | 0.665 ± 0.030 |
| reference config (MC) | 0.693 ± 0.051 | 0.427 ± 0.014 | 0.568 ± 0.021 |
| mean |σ| final (init) | 1.42 (2.84) | 4.07 (3.15) | 3.24 (3.15) |

# T3_readouts

| readout of the same trained circuit | baker | gw | mitbih |
|---|---|---|---|
| QVR (tuned) | 0.729 | 0.721 | 0.690 |
| QVR circuit, per-t z² readout | 0.628 | 0.817 | 0.710 |
| QVR circuit, mean-⟨Z⟩ Mahalanobis | 0.568 | 0.628 | 0.786 |
| QVR circuit, p̄_b Mahalanobis | 0.623 | 0.634 | 0.711 |

# T4_fourier

Across every trained model (best, ref, σ0; all seeds) and every D mode.

| dataset | K | models where all predictions hold | min R² ⟨Z⟩ @K | max R² ⟨Z⟩ @K−1 | min R² score @2K | median damping, last t |
|---|---|---|---|---|---|---|
| baker | [1] | 100% | 1.0000000000 | -0.0000 | 1.0000000000 | 4.37e-175 |
| gw | [3] | 100% | 1.0000000000 | 0.8998 | 1.0000000000 | 0 |
| mitbih | [3] | 100% | 1.0000000000 | 0.8679 | 1.0000000000 | 0 |

# T5_probes

Macro AUC of a logistic probe fit on anomalies from training units, evaluated on test anomalies; p = unit-level permutation null.

| dataset | raw series | six moments | QVR p̄_b (mean over seeds) | note |
|---|---|---|---|---|
| baker | — | — | — | no Xprobe / subtypes |
| gw | — | — | — | no Xprobe / subtypes |
| mitbih | 0.945 (p 0.005) | 0.815 (p 0.050) | 0.813 |  |

# T6_protocol

| dataset | null guard: max deviation of AUC from 0.5 | units | QVR AUC, time-permuted | ΔAUC (orig − permuted) |
|---|---|---|---|---|
| baker | 0.107 | 60 | 0.760 | -0.049 |
| gw | 0.066 | 26 | 0.718 | -0.008 |
| mitbih | 0.118 | 22 | 0.683 | 0.009 |

# T7_predictions

Registered predictions (docs/REGISTRATION.md), definitions per docs/DEVIATIONS.md.

| prediction | baker | gw | mitbih |
|---|---|---|---|
| P1 | holds: 1.0 | holds: 1.0 | holds: 1.0 |
| P2 | holds: part of P1 verdicts | holds: part of P1 verdicts | holds: part of P1 verdicts |
| P3 | FAILS: pca (val-selected): Δ -0.038 [-0.078, +0.002] | holds: tz2 (val-selected): Δ +0.069 [-0.010, +0.146] | FAILS: pca (val-selected): Δ +0.022 [-0.079, +0.146] |
| P4 | FAILS: trig_t 0.696 vs QVR 0.729 | holds: trig_t 0.774 vs QVR 0.721 | holds: trig_t 0.732 vs QVR 0.690 |
| P5 | holds: identity 0.764 vs QVR 0.729 | holds: identity 0.743 vs QVR 0.721 | holds: identity 0.660 vs QVR 0.690 |
| P6 | FAILS: σ0 − QVR -0.024 [-0.047, -0.001] | FAILS: σ0 − QVR -0.155 [-0.241, -0.068] | holds: σ0 − QVR -0.024 [-0.135, +0.080] |
| P7 | FAILS: best readout 0.628 vs QVR 0.729 | holds: best readout 0.817 vs QVR 0.721 | holds: best readout 0.786 vs QVR 0.690 |
| P8 | —: no probes | —: no probes | holds: p̄_b 0.813 vs moments 0.815 |
| P9 | FAILS: max deviation 0.107 (60 units) (calibrated: holds, percentile 44% of 200 unit splits (median 0.114, 95th 0.173)) | holds: max deviation 0.066 (26 units) (calibrated: holds, percentile 68% of 200 unit splits (median 0.050, 95th 0.131)) | FAILS: max deviation 0.118 (22 units) (calibrated: holds, percentile 16% of 200 unit splits (median 0.162, 95th 0.273)) |

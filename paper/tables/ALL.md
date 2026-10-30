# T1_main

Pooled test AUC [95% unit-level CI]. QVR rows: rank-averaged over seeds.

| method | mitbih | baker | gw |
|---|---|---|---|
| QVR (tuned) | 0.690 [0.455, 0.881] | 0.729 [0.658, 0.791] | 0.721 [0.629, 0.809] |
| QVR (reference config) | 0.569 [0.466, 0.673] | 0.706 [0.636, 0.768] | 0.424 [0.328, 0.531] |
| QVR, D = 0 (cos x) | 0.660 [0.394, 0.870] | 0.764 [0.699, 0.825] | 0.743 [0.658, 0.827] |
| QVR circuit, per-t z² readout | 0.710 [0.450, 0.909] | 0.628 [0.560, 0.693] | 0.817 [0.759, 0.885] |
| QVR circuit, mean-⟨Z⟩ Mahalanobis | 0.786 [0.646, 0.899] | 0.568 [0.509, 0.633] | 0.628 [0.507, 0.744] |
| QVR circuit, p̄_b Mahalanobis | 0.711 [0.509, 0.881] | 0.623 [0.557, 0.684] | 0.634 [0.525, 0.741] |
| one number per series | 0.669 [0.579, 0.753] | 0.698 [0.630, 0.761] | 0.663 [0.544, 0.778] |
| one temporal number (lag-1 / MSD) | 0.546 [0.334, 0.763] | 0.661 [0.591, 0.735] | 0.516 [0.396, 0.647] |
| per-timepoint z² | 0.722 [0.496, 0.897] | 0.652 [0.585, 0.715] | 0.790 [0.726, 0.859] |
| per-timepoint trig (QVR class) | 0.732 [0.507, 0.903] | 0.696 [0.625, 0.758] | 0.774 [0.707, 0.846] |
| six moments (fingerprint) | 0.717 [0.522, 0.885] | 0.665 [0.592, 0.730] | 0.609 [0.489, 0.734] |
| Mahalanobis (whole series) | 0.775 [0.592, 0.913] | 0.692 [0.615, 0.760] | 0.798 [0.730, 0.868] |
| PCA reconstruction | 0.712 [0.512, 0.871] | 0.691 [0.614, 0.763] | 0.781 [0.714, 0.853] |
| kNN distance | 0.734 [0.503, 0.914] | 0.713 [0.639, 0.779] | 0.786 [0.712, 0.852] |
| Isolation Forest | 0.742 [0.513, 0.916] | 0.691 [0.620, 0.755] | 0.743 [0.660, 0.822] |

# T2_dmodes

Pooled test AUC, mean ± SD over seeds.

| tuned QVR, scored with | mitbih | baker | gw |
|---|---|---|---|
| MC draws (as trained) | 0.692 ± 0.005 | 0.711 ± 0.039 | 0.709 ± 0.022 |
| D = μ | 0.732 ± 0.056 | 0.716 ± 0.051 | 0.608 ± 0.070 |
| D = 0 | 0.660 ± 0.002 | 0.763 ± 0.000 | 0.743 ± 0.018 |
| averaged D̄ | 0.666 ± 0.012 | 0.700 ± 0.052 | 0.608 ± 0.046 |
| σ = 0 during training | 0.665 ± 0.030 | 0.698 ± 0.026 | 0.562 ± 0.024 |
| reference config (MC) | 0.568 ± 0.021 | 0.693 ± 0.051 | 0.427 ± 0.014 |
| mean |σ| final (init) | 3.24 (3.15) | 1.42 (2.84) | 4.07 (3.15) |

# T3_readouts

| readout of the same trained circuit | mitbih | baker | gw |
|---|---|---|---|
| QVR (tuned) | 0.690 | 0.729 | 0.721 |
| QVR circuit, per-t z² readout | 0.710 | 0.628 | 0.817 |
| QVR circuit, mean-⟨Z⟩ Mahalanobis | 0.786 | 0.568 | 0.628 |
| QVR circuit, p̄_b Mahalanobis | 0.711 | 0.623 | 0.634 |

# T4_fourier

Across every trained model (best, ref, σ0; all seeds) and every D mode.

| dataset | K | models where all predictions hold | min R² ⟨Z⟩ @K | max R² ⟨Z⟩ @K−1 | min R² score @2K | median damping, last t |
|---|---|---|---|---|---|---|
| mitbih | [3] | 100% | 1.0000000000 | 0.8679 | 1.0000000000 | 0 |
| baker | [1] | 100% | 1.0000000000 | -0.0000 | 1.0000000000 | 4.37e-175 |
| gw | [3] | 100% | 1.0000000000 | 0.8998 | 1.0000000000 | 0 |

# T5_probes

Macro AUC of a logistic probe fit on anomalies from training units, evaluated on test anomalies; p = unit-level permutation null.

| dataset | raw series | six moments | QVR p̄_b (mean over seeds) | note |
|---|---|---|---|---|
| mitbih | 0.945 (p 0.005) | 0.815 (p 0.050) | 0.813 |  |
| baker | — | — | — | no Xprobe / subtypes |
| gw | — | — | — | no Xprobe / subtypes |

# T6_protocol

| dataset | null guard: max deviation of AUC from 0.5 | units | QVR AUC, time-permuted | ΔAUC (orig − permuted) |
|---|---|---|---|---|
| mitbih | 0.118 | 22 | 0.683 | 0.009 |
| baker | 0.107 | 60 | 0.760 | -0.049 |
| gw | 0.066 | 26 | 0.718 | -0.008 |

# T7_predictions

Registered predictions (docs/REGISTRATION.md), definitions per docs/DEVIATIONS.md.

| prediction | mitbih | baker | gw |
|---|---|---|---|
| P1 | holds: 1.0 | holds: 1.0 | holds: 1.0 |
| P2 | holds: part of P1 verdicts | holds: part of P1 verdicts | holds: part of P1 verdicts |
| P3 | FAILS: pca (val-selected): Δ +0.022 [-0.079, +0.146] | FAILS: pca (val-selected): Δ -0.038 [-0.078, +0.002] | holds: tz2 (val-selected): Δ +0.069 [-0.010, +0.146] |
| P4 | holds: trig_t 0.732 vs QVR 0.690 | FAILS: trig_t 0.696 vs QVR 0.729 | holds: trig_t 0.774 vs QVR 0.721 |
| P5 | holds: identity 0.660 vs QVR 0.690 | holds: identity 0.764 vs QVR 0.729 | holds: identity 0.743 vs QVR 0.721 |
| P6 | holds: σ0 − QVR -0.024 [-0.135, +0.080] | FAILS: σ0 − QVR -0.024 [-0.047, -0.001] | FAILS: σ0 − QVR -0.155 [-0.241, -0.068] |
| P7 | holds: best readout 0.786 vs QVR 0.690 | FAILS: best readout 0.628 vs QVR 0.729 | holds: best readout 0.817 vs QVR 0.721 |
| P8 | holds: p̄_b 0.813 vs moments 0.815 | —: no probes | —: no probes |
| P9 (registered: ≤ 0.10) | FAILS: max deviation 0.118 (22 units) | FAILS: max deviation 0.107 (60 units) | holds: max deviation 0.066 (26 units) |
| P9 (calibrated, D2: below 95th pct of 200 unit splits) | holds: percentile 16% of 200 unit splits (median 0.162, 95th 0.273) | holds: percentile 44% of 200 unit splits (median 0.114, 95th 0.173) | holds: percentile 68% of 200 unit splits (median 0.050, 95th 0.131) |

# T8_forensics

Published pipelines run as written (forensics/). BA = balanced accuracy with the threshold tuned as published. GW one-number baseline = per-series std after the rung's scaling (0 or 1 = perfect separation).

| claim | pipeline | published BA | reproduced BA | QVR, other | one-number baseline AUC |
|---|---|---|---|---|---|
| GW | paper, Table 3 (filtered; 5 runs) | 0.919 | — | AUC 0.917 (their saved scores) | — |
| GW | rung A: their protocol: separate scaling, validation events = test events | — | 0.903 | AUC 0.949 (folds 1.000, 0.988, 0.835, 1.000, 0.924) | 0.000 |
| GW | rung A_their: their protocol, the authors' own events | — | 0.877 | AUC 0.922 (folds 1.000, 0.976, 0.746, 1.000, 0.887) | 0.000 |
| GW | rung A0: their protocol, NO SIGNAL (noise windows as "events") | — | 0.898 | AUC 0.944 (folds 1.000, 0.984, 0.869, 0.994, 0.875) | 0.000 |
| GW | rung B: one scaler fit on training noise | — | 0.506 | AUC 0.489 (folds 0.526, 0.432, 0.459, 0.494, 0.534) | 0.714 |
| GW | rung B0: one scaler, NO SIGNAL | — | 0.535 | AUC 0.534 (folds 0.615, 0.519, 0.503, 0.502, 0.530) | 0.504 |
| GW | rung C: one scaler, disjoint validation / test events | — | 0.512 | AUC 0.464 (folds 0.582, 0.310, 0.507, 0.425, 0.496) | 0.715 |

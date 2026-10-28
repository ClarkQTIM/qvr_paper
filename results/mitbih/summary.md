# Summary — mitbih

Split dir `data_splits/mitbih` · T = 126 · channels = 1 · seeds [0, 1, 2, 3, 4] · test sets Xte_fusion, Xte_supraventricular, Xte_ventricular

Sizes: Xprobe 4,442, Xte_fusion 388, Xte_norm 44,240, Xte_supraventricular 1,837, Xte_ventricular 3,220, Xtr 36,424, Xval 704, Xval_norm 9,424

## Registered predictions (docs/REGISTRATION.md; definitions per docs/DEVIATIONS.md)

| ID | holds | value |
|---|---|---|
| P1 | ✅ | 1.0 |
| P2 | ✅ | part of P1 verdicts |
| P3 | ❌ | pca (val-selected): Δ +0.022 [-0.079, +0.146] · also mahal (best on test): Δ +0.086 [+0.010, +0.216] |
| P4 | ✅ | trig_t 0.732 vs QVR 0.690 |
| P5 | ✅ | identity 0.660 vs QVR 0.690 |
| P6 | ✅ | σ0 − QVR -0.024 [-0.135, +0.080] |
| P7 | ✅ | best readout 0.786 vs QVR 0.690 |
| P8 | ✅ | p̄_b 0.813 vs moments 0.815 |
| P9 | ❌ | max deviation 0.118 (22 units) · calibrated ✅: percentile 16% of 200 unit splits (median 0.162, 95th 0.273) |

## Selection

QVR winner (best validation AUC): `{"n_layers": 5, "N_E": 10, "cost": "inside", "embedding": "ry_all", "ansatz": "chain", "lr": 0.003, "n_qubits": 3}`

| QVR config | val AUC |
|---|---|
| L5_NE10_inside_ry_all_chain_lr0.003 | 0.708 |
| L5_NE1_outside_ry_all_chain_lr0.01 | 0.670 |
| L5_NE1_inside_ry_all_chain_lr0.01 | 0.670 |
| L5_NE10_inside_ry_all_chain_lr0.01 | 0.663 |
| L5_NE1_outside_ry_all_chain_lr0.003 | 0.662 |
| L5_NE1_inside_ry_all_chain_lr0.003 | 0.662 |
| L5_NE10_outside_ry_all_chain_lr0.003 | 0.657 |
| L3_NE1_outside_ry_all_chain_lr0.01 | 0.650 |

## C4 — Main comparison (test, pooled anomalies)

| method | AUC | 95% unit CI | Δ vs qvr_best [CI] | Xte_fusion | Xte_supraventricular | Xte_ventricular |
|---|---|---|---|---|---|---|
| qvr_best_ro_mahal_zmean | 0.786 | [0.646, 0.899] | +0.096 [-0.017, +0.241] | 0.896 | 0.565 | 0.898 |
| mahal | 0.775 | [0.592, 0.913] | +0.086 [+0.010, +0.216] | 0.903 | 0.481 | 0.928 |
| iforest | 0.742 | [0.513, 0.916] | +0.053 [-0.027, +0.153] | 0.956 | 0.373 | 0.927 |
| knn | 0.734 | [0.503, 0.914] | +0.045 [-0.031, +0.143] | 0.950 | 0.384 | 0.908 |
| qvr_best_mean | 0.734 | [0.561, 0.893] | +0.045 [-0.016, +0.111] | 0.928 | 0.406 | 0.897 |
| trig_t | 0.732 | [0.507, 0.903] | +0.042 [-0.042, +0.146] | 0.941 | 0.369 | 0.913 |
| tz2 | 0.722 | [0.496, 0.897] | +0.033 [-0.056, +0.131] | 0.946 | 0.356 | 0.904 |
| moments | 0.717 | [0.522, 0.885] | +0.028 [-0.070, +0.125] | 0.597 | 0.402 | 0.911 |
| pca | 0.712 | [0.512, 0.871] | +0.022 [-0.079, +0.146] | 0.895 | 0.422 | 0.855 |
| qvr_best_ro_mahal_pbar | 0.711 | [0.509, 0.881] | +0.021 [-0.069, +0.119] | 0.604 | 0.385 | 0.909 |
| qvr_best_ro_tz2_z | 0.710 | [0.450, 0.909] | +0.021 [-0.056, +0.088] | 0.939 | 0.283 | 0.927 |
| qvr_best | 0.690 | [0.455, 0.881] | — | 0.903 | 0.291 | 0.891 |
| qvr_best_averaged | 0.681 | [0.487, 0.848] | -0.008 [-0.045, +0.033] | 0.877 | 0.344 | 0.851 |
| onenum | 0.669 | [0.579, 0.753] | -0.020 [-0.247, +0.241] | 0.677 | 0.636 | 0.688 |
| qvr_sigma0 | 0.666 | [0.512, 0.813] | -0.024 [-0.135, +0.080] | 0.899 | 0.404 | 0.788 |
| qvr_best_identity | 0.660 | [0.394, 0.870] | -0.030 [-0.067, +0.005] | 0.724 | 0.249 | 0.886 |
| qvr_ref | 0.569 | [0.466, 0.673] | -0.121 [-0.345, +0.099] | 0.822 | 0.486 | 0.586 |
| temporal | 0.546 | [0.334, 0.763] | -0.143 [-0.274, -0.018] | 0.233 | 0.247 | 0.755 |

QVR rows are rank-averaged over seeds. Per-seed AUC (mean ± SD): best 0.692 ± 0.005, ref 0.568 ± 0.021, sigma0 0.665 ± 0.030

## C3 — D modes, σ = 0 ablation, Hamiltonian

| D mode | best (mean ± SD over seeds) | ref |
|---|---|---|
| mc | 0.692 ± 0.005 | 0.568 ± 0.021 |
| mean | 0.732 ± 0.056 | 0.568 ± 0.031 |
| identity | 0.660 ± 0.002 | 0.649 ± 0.003 |
| averaged | 0.666 ± 0.012 | 0.591 ± 0.015 |

σ frozen at 0: AUC 0.665 vs learned σ 0.692 (ensemble Δ -0.024 [-0.135, +0.080]). Training length (best batch): σ0 4610 vs learned 5350.

- best: mean |σ| 3.239 (init 3.154), mean |Δμ| 0.800, S_r 1: 0.17, 2: 0.67, 3: 0.16 (init 1: 0.18, 2: 0.66, 3: 0.16)
- ref: mean |σ| 3.255 (init 3.193), mean |Δμ| 3.667, S_r 1: 0.45, 2: 0.46, 3: 0.08 (init 1: 0.45, 2: 0.43, 3: 0.11)

## C2 — Fourier exactness and order

- All predictions hold on 100% of trained models (K = [3]).
- min R² of z at degree K: 1.0000000000; max R² at K−1: 0.8679; min R² of the score term at 2K: 1.0000000000.
- Median coherence damping at the last timepoint (best models): 0.
- Order: AUC on time-permuted copies 0.683; ΔAUC (orig − perm) 0.009; Spearman(orig, perm) 0.658.

## C5 — Subtype probes (fit on Xprobe, evaluated on test anomalies)

| features | macro AUC (logreg) |
|---|---|
| raw series | 0.945 (null p 0.005) |
| six moments | 0.815 (null p 0.050) |
| QVR p̄_b (first seed) | 0.821 (null p 0.065) |
| QVR p̄_b (mean over seeds) | 0.813 |

## Protocol — null guard (test normals vs test normals, split by unit)

Max deviation of AUC from 0.5 over all methods: 0.118 (22 units). Calibration over 200 random unit splits: median 0.162, 95th percentile 0.273; observed at the 16% percentile.

## Baseline hyperparameters (selected on validation)

| method | hp | val AUC |
|---|---|---|
| onenum | {'stat': 'absmax', 'channel': 0} | 0.746 |
| temporal | {'stat': 'lag1', 'channel': 0} | 0.652 |
| tz2 | {} | 0.842 |
| trig_t | {'K': 3} | 0.843 |
| moments | {'K': 3} | 0.755 |
| mahal | {} | 0.790 |
| pca | {'n': 2} | 0.865 |
| knn | {'k': 20} | 0.840 |
| iforest | {'max_samples': 256} | 0.841 |

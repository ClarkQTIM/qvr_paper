# Summary — baker

Split dir `data_splits/baker` · T = 180 · channels = 2 · seeds [0, 1, 2, 3, 4] · test sets Xte_clean_btc, Xte_clean_usdt, Xte_dirty_btc_neg, Xte_dirty_btc_pos, Xte_dirty_usdt_neg, Xte_dirty_usdt_pos

Sizes: Xte_clean_btc 34, Xte_clean_usdt 19, Xte_dirty_btc_neg 60, Xte_dirty_btc_pos 60, Xte_dirty_usdt_neg 60, Xte_dirty_usdt_pos 60, Xte_norm 60, Xtr 665, Xval 60, Xval_norm 117

## Registered predictions (docs/REGISTRATION.md; definitions per docs/DEVIATIONS.md)

| ID | holds | value |
|---|---|---|
| P1 | ✅ | 1.0 |
| P2 | ✅ | part of P1 verdicts |
| P3 | ❌ | pca (val-selected): Δ -0.038 [-0.078, +0.002] · also knn (best on test): Δ -0.016 [-0.050, +0.016] |
| P4 | ❌ | trig_t 0.696 vs QVR 0.729 |
| P5 | ✅ | identity 0.764 vs QVR 0.729 |
| P6 | ❌ | σ0 − QVR -0.024 [-0.047, -0.001] |
| P7 | ❌ | best readout 0.628 vs QVR 0.729 |
| P8 | — | no probes |
| P9 | ❌ | max deviation 0.107 (60 units) · calibrated ✅: percentile 44% of 200 unit splits (median 0.114, 95th 0.173) |

## Selection

QVR winner (best validation AUC): `{"n_layers": 3, "N_E": 10, "cost": "outside", "embedding": "rx_per_qubit", "ansatz": "sel", "lr": 0.003, "n_qubits": 2}`

| QVR config | val AUC |
|---|---|
| L3_NE10_outside_rx_per_qubit_sel_lr0.003 | 0.682 |
| L3_NE10_outside_rx_per_qubit_sel_lr0.01 | 0.666 |
| L5_NE10_outside_rx_per_qubit_sel_lr0.003 | 0.666 |
| L3_NE10_inside_rx_per_qubit_chain_lr0.01 | 0.665 |
| L5_NE10_outside_rx_per_qubit_chain_lr0.003 | 0.656 |
| L3_NE1_outside_rx_per_qubit_sel_lr0.003 | 0.655 |
| L3_NE1_inside_rx_per_qubit_sel_lr0.003 | 0.655 |
| L5_NE10_inside_rx_per_qubit_sel_lr0.003 | 0.652 |

## C4 — Main comparison (test, pooled anomalies)

| method | AUC | 95% unit CI | Δ vs qvr_best [CI] | Xte_clean_btc | Xte_clean_usdt | Xte_dirty_btc_neg | Xte_dirty_btc_pos | Xte_dirty_usdt_neg | Xte_dirty_usdt_pos |
|---|---|---|---|---|---|---|---|---|---|
| qvr_best_identity | 0.764 | [0.699, 0.825] | +0.035 [+0.002, +0.074] | 0.670 | 0.654 | 0.747 | 0.750 | 0.776 | 0.873 |
| qvr_best_mean | 0.735 | [0.662, 0.800] | +0.006 [-0.008, +0.022] | 0.655 | 0.616 | 0.687 | 0.728 | 0.759 | 0.843 |
| qvr_best | 0.729 | [0.658, 0.791] | — | 0.665 | 0.630 | 0.676 | 0.714 | 0.754 | 0.834 |
| qvr_best_averaged | 0.723 | [0.650, 0.789] | -0.006 [-0.020, +0.010] | 0.641 | 0.597 | 0.662 | 0.716 | 0.754 | 0.835 |
| knn | 0.713 | [0.639, 0.779] | -0.016 [-0.050, +0.016] | 0.636 | 0.541 | 0.675 | 0.698 | 0.740 | 0.838 |
| qvr_ref | 0.706 | [0.636, 0.768] | -0.023 [-0.047, +0.000] | 0.635 | 0.577 | 0.660 | 0.694 | 0.739 | 0.824 |
| qvr_sigma0 | 0.705 | [0.637, 0.766] | -0.024 [-0.047, -0.001] | 0.610 | 0.539 | 0.656 | 0.703 | 0.741 | 0.831 |
| onenum | 0.698 | [0.630, 0.761] | -0.031 [-0.092, +0.037] | 0.577 | 0.495 | 0.708 | 0.689 | 0.705 | 0.823 |
| trig_t | 0.696 | [0.625, 0.758] | -0.033 [-0.072, +0.009] | 0.583 | 0.476 | 0.688 | 0.689 | 0.709 | 0.834 |
| mahal | 0.692 | [0.615, 0.760] | -0.037 [-0.076, +0.001] | 0.609 | 0.524 | 0.631 | 0.683 | 0.746 | 0.809 |
| pca | 0.691 | [0.614, 0.763] | -0.038 [-0.078, +0.002] | 0.612 | 0.522 | 0.629 | 0.679 | 0.752 | 0.804 |
| iforest | 0.691 | [0.620, 0.755] | -0.038 [-0.074, +0.000] | 0.580 | 0.480 | 0.663 | 0.685 | 0.712 | 0.833 |
| moments | 0.665 | [0.592, 0.730] | -0.065 [-0.115, -0.017] | 0.554 | 0.497 | 0.677 | 0.649 | 0.664 | 0.783 |
| temporal | 0.661 | [0.591, 0.735] | -0.068 [-0.133, -0.003] | 0.505 | 0.540 | 0.633 | 0.660 | 0.704 | 0.775 |
| tz2 | 0.652 | [0.585, 0.715] | -0.078 [-0.123, -0.025] | 0.522 | 0.418 | 0.604 | 0.658 | 0.682 | 0.810 |
| qvr_best_ro_tz2_z | 0.628 | [0.560, 0.693] | -0.101 [-0.166, -0.028] | 0.510 | 0.384 | 0.624 | 0.611 | 0.647 | 0.768 |
| qvr_best_ro_mahal_pbar | 0.623 | [0.557, 0.684] | -0.107 [-0.163, -0.047] | 0.505 | 0.403 | 0.609 | 0.623 | 0.637 | 0.769 |
| qvr_best_ro_mahal_zmean | 0.568 | [0.509, 0.633] | -0.161 [-0.242, -0.079] | 0.474 | 0.467 | 0.557 | 0.499 | 0.601 | 0.702 |

QVR rows are rank-averaged over seeds. Per-seed AUC (mean ± SD): best 0.711 ± 0.039, ref 0.693 ± 0.051, sigma0 0.698 ± 0.026

## C3 — D modes, σ = 0 ablation, Hamiltonian

| D mode | best (mean ± SD over seeds) | ref |
|---|---|---|
| mc | 0.711 ± 0.039 | 0.693 ± 0.051 |
| mean | 0.716 ± 0.051 | 0.696 ± 0.053 |
| identity | 0.763 ± 0.000 | 0.763 ± 0.000 |
| averaged | 0.700 ± 0.052 | 0.692 ± 0.051 |

σ frozen at 0: AUC 0.698 vs learned σ 0.711 (ensemble Δ -0.024 [-0.047, -0.001]). Training length (best batch): σ0 4240 vs learned 4990.

- best: mean |σ| 1.421 (init 2.836), mean |Δμ| 0.296, S_r 1: 0.84, 2: 0.16 (init 1: 0.82, 2: 0.18)
- ref: mean |σ| 0.717 (init 2.836), mean |Δμ| 0.354, S_r 1: 0.77, 2: 0.23 (init 1: 0.82, 2: 0.18)

## C2 — Fourier exactness and order

- All predictions hold on 100% of trained models (K = [1]).
- min R² of z at degree K: 1.0000000000; max R² at K−1: -0.0000; min R² of the score term at 2K: 1.0000000000.
- Median coherence damping at the last timepoint (best models): 4.37e-175.
- Order: AUC on time-permuted copies 0.760; ΔAUC (orig − perm) -0.049; Spearman(orig, perm) 0.870.

## C5 — Subtype probes (fit on Xprobe, evaluated on test anomalies)

Skipped: no Xprobe / subtypes

## Protocol — null guard (test normals vs test normals, split by unit)

Max deviation of AUC from 0.5 over all methods: 0.107 (60 units). Calibration over 200 random unit splits: median 0.114, 95th percentile 0.173; observed at the 44% percentile.

## Baseline hyperparameters (selected on validation)

| method | hp | val AUC |
|---|---|---|
| onenum | {'stat': 'mean', 'channel': 1} | 0.653 |
| temporal | {'stat': 'msd', 'channel': 0} | 0.668 |
| tz2 | {} | 0.619 |
| trig_t | {'K': 3} | 0.649 |
| moments | {'K': 2} | 0.670 |
| mahal | {} | 0.675 |
| pca | {'n': 20} | 0.681 |
| knn | {'k': 5} | 0.646 |
| iforest | {'max_samples': 256} | 0.651 |

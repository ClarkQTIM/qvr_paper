# Results — 2026-10-30 — GW (rebuilt from GWOSC, D5)

Dated record; do not edit after commit. Run: `prereg-v1` code with report
amendments D1–D3; registered univariate grid (48 configs × 2 seeds); 5 final seeds;
32 workers on ophlslcec08 (finished 03:02, about 5.5 h, sharing the machine with another
29-worker job). Source: `results/gw/summary.md`.

Data: `prepare.py gw-gwosc` (D5). L1, O3, band-pass 50–250 Hz plus notches, 1-s
windows decimated ×4 to T = 1024. 86 catalog events found, 3 skipped for data gaps
(GW190514_065416, GW190719_215514, 200214_224526), 83 used. Units are event
neighbourhoods, split 31/26/26 into train/val/test: 3,100 training noise windows;
validation 2,600 noise + 26 events; test 2,600 noise + 26 events (disjoint from
validation). Normalizer fit on training noise; about 1% of noise samples clipped vs
2.0–2.4% of event samples.
**26 test events: intervals are wide** (about ±0.09 on QVR's AUC).

## Scorecard

| ID | Prediction | Result |
|---|---|---|
| P1 | Fourier structure exact on every trained model | **holds**: 15/15; K = 3; min R² ⟨Z⟩ @K = 1.0000000000, max @K−1 = 0.900; score @2K = 1.0000000000 |
| P2 | Damping matches simulation | **holds** |
| P3 | Tuned QVR does not beat the best classical method | **holds**: val-selected tz2 ahead of QVR by 0.069 [−0.010, +0.146]; test-best Mahalanobis by 0.077 [−0.005, +0.167] |
| P4 | trig_t ≥ QVR − 0.02 | **holds**: 0.774 vs 0.721 |
| P5 | D = 0 within 0.05 of QVR | **holds**: D = 0 is better, 0.743 vs 0.721 (Δ +0.022 [−0.058, +0.111]) |
| P6 | Learning σ does not help | **fails, clearly**: σ0 − QVR −0.155 [−0.241, −0.068] |
| P7 | A better readout of the same circuit helps | **holds**: per-timepoint z² readout 0.817 vs native 0.721 (Δ +0.097 [+0.031, +0.161]) |
| P8 | p̄_b ≤ moments + 0.02 | not applicable (no subtypes) |
| P9 | Null guard ≤ 0.10 | **holds** as registered (0.066, 26 units) and calibrated (68th percentile; median 0.050, 95th 0.131) |

## Main numbers (pooled test AUC, 95% unit CI, 26 test events)

| Method | AUC |
|---|---|
| QVR circuit, per-timepoint z² readout of ⟨Z⟩ | 0.817 [0.759, 0.885] |
| Mahalanobis (whole window) | 0.798 [0.730, 0.868] |
| per-timepoint z² (val-selected best classical) | 0.790 [0.726, 0.859] |
| kNN / PCA | 0.786 / 0.781 |
| per-timepoint trig (QVR's class) | 0.774 |
| **QVR with D = 0 (cos x; trained W cancels)** | **0.743 [0.658, 0.827]** |
| Isolation Forest | 0.743 |
| **QVR tuned (native score)** | **0.721 [0.629, 0.809]** |
| QVR, D averaged before the circuit | 0.691 |
| one number per series (std) | 0.663 |
| QVR, D = μ | 0.640 |
| six moments | 0.609 |
| QVR σ = 0 trained | 0.566 |
| temporal (lag-1) | 0.516 |
| QVR reference config | 0.424 (below chance; seeds 0.427 ± 0.014) |

Selected QVR: 5 layers, N_E = 10, cost inside, ry_all, lr 0.01 (val AUC 0.788).
Per-seed AUC 0.709 ± 0.022. All four top validation configs are N_E = 10 with cost
inside (0.769–0.788); the best N_E = 1 configs reach 0.705. With N_E = 1 the two costs
are the same function, and their validation AUCs are identical (0.705 / 0.705,
0.702 / 0.702), which is a built-in consistency check.

## Readings

1. **The best QVR model has removed its own time evolution.** σ grew during
   training (3.15 → 4.08). Median coherence damping at the last timepoint is 0.
   Under the draws, the time-dependent terms of ⟨Z⟩ average away, leaving a
   time-invariant function of each x_t. The deterministic D modes, which keep the
   coherent phases, score worse than the draws: D = μ 0.608 and D averaged 0.608,
   against MC 0.709 (seed means). D = 0, which has no time evolution at all,
   scores 0.743, the best of any QVR scoring mode.
2. **This is why learning σ helps here (P6 fails).** The σ = 0 model must keep
   the coherent phases at every timepoint and scores 0.562. The learned-σ model
   escapes them by dephasing. So σ helps on GW only by erasing the dynamics,
   which is the mechanism we wrongly proposed for Baker, where σ shrank. On GW
   it is supported by three facts: σ grew, damping is complete, and D = 0 ≥ QVR.
   Exploratory, not registered: scoring the best models in the fully dephased limit
   (σ → ∞, analytic) would confirm it directly.
3. **The draw variance is the trained model's extra feature.** With the inside
   cost, the score is mean_t[(η₀ − E m)² + Var_draws(m)]. Both parts are
   time-invariant trigonometric polynomials of x_t (degree ≤ 2K = 6; P1 at 2K holds
   exactly). That puts the selected model in trig_t's function class, and trig_t
   does better (0.774 vs 0.721; P4).
4. **No use of order, even for chirps.** Time-permuted test windows give 0.718
   vs 0.721 (|Δ| < 0.01). A GW chirp is a temporal pattern, yet this detector
   treats each window as a bag of samples. The best classical detectors here are
   per-timepoint power (tz2) and global atypicality (Mahalanobis, kNN, PCA).
5. **The circuit's features are informative; the native score discards them.**
   A per-timepoint z² readout of the trained ⟨Z⟩ gives 0.817, the best row. That
   is 0.027 above the same readout on raw x (tz2 0.790), within noise.
6. **The reference config is below chance (0.424).** It uses the outside cost,
   and it is stable across seeds. Under D = 0 it rises to 0.550. Reported, not
   interpreted further.

## Relation to the published GW claim

The paper reports a mean test **accuracy** of 0.919 on filtered L1 data (Table 3; five
runs, 0.596 and four 1.000, at a threshold tuned on the same events). The AUCs from
their saved test scores are close: 0.587 and four 1.000, mean 0.917. Under our
protocol on comparably built data, tuned QVR gives an AUC of 0.721 [0.629, 0.809]. This
is **not a reproduction of their pipeline**; that is the separate forensic analysis
(`forensics/gw_ladder.py`, which rebuilds their pipeline on this data and removes its
choices one at a time). Our protocol differs in four ways:
- disjoint validation and test events (theirs overlap);
- one normalizer fit on training noise (their code scales sets separately);
- T = 1024 after anti-aliased decimation (theirs is 4,096 samples);
- the registered grid, which uses ry embeddings and does not include their exact
  rx_first / chain_full circuit.

Which of these accounts for the gap is not separated here. Under our protocol, five
generic classical detectors match or exceed tuned QVR, and QVR's own D = 0 reduction
beats it.

## Cross-dataset reading (MIT-BIH, Baker, GW)

- **Fourier structure:** exact on every trained model (P1: 45/45).
- **D = 0 (no time evolution) vs tuned QVR:** −0.030 on MIT-BIH, +0.035 on Baker,
  +0.022 on GW. Removing the quantum dynamics never costs more than 0.03, and helps on
  two of three datasets.
- **Order:** unused (MIT-BIH, GW) or harmful (Baker).
- **σ:** its effect depends on the dataset. On MIT-BIH it did not move and did not
  help. On Baker it shrank, with a borderline benefit. On GW it grew and clearly
  helped, through dephasing, i.e. by removing the dynamics.
- **QVR vs classical:** tuned QVR is never the best detector. MIT-BIH and GW: generic
  detectors match or beat it. Baker: it beats them, but its own D = 0 reduction beats
  it.
- **Null guard:** passes on all three datasets under the calibrated definition (D2).

# Forensics — 2026-10-30 — GW: the published pipeline rebuilt on public LIGO data

Dated record; do not edit after commit. Exploratory forensics, not a registered
analysis. Code: `forensics/gw_ladder.py` (commit fa8e8c7). Output:
`results/forensics/gw_ladder/` and `logs/gw_ladder.out`.

## What was run

The QVR4GW pipeline as written in their code (`rewinder.py`, `utils.py`):
- 2 qubits, RX(x) on qubit 0, 3 layers, D = Z0, Z1, Z0Z1, N_E = 10, 256 shots;
- Powell with maxfev 2000 on a 10-series × 10-timepoint minibatch objective, with
  the arctan σ penalty (τ = 15);
- per-timepoint min-max scaling;
- 4,000 / 78 / 78 noise windows for train / validation / test;
- a balanced-accuracy threshold grid search on validation.

Data: our GWOSC rebuild (D5) at the original rate: 83 L1 O3 events and 8,300 filtered
noise windows, 1 s at 4,096 Hz (T = 4,096). Each fold draws a 5,000-window noise
pool, standing in for their noise file. Five folds.

Rungs remove the protocol choices one at a time. Rung A0 replaces the events with
noise windows from 8 held-out event neighbourhoods. Those windows contain **no
gravitational-wave signal**.

## Results (test AUC; BA = balanced accuracy at the validation-tuned threshold)

| Rung | Protocol | QVR AUC per fold | Mean AUC | Mean BA | Std-only AUC | Scaled std (noise / events) |
|---|---|---|---|---|---|---|
| — | **Paper, Table 3 (filtered)** | 0.587, 1, 1, 1, 1 (from their saved scores) | 0.917 | **0.919** | — | — |
| A | their protocol: separate scaling, validation events = test events | 1.000, 0.988, 0.835, 1.000, 0.924 | 0.949 | 0.903 | 0.000 | 1.72 / 1.32 |
| A_their | the same, with the authors' own 78 events | 1.000, 0.976, 0.746, 1.000, 0.887 | 0.922 | 0.877 | 0.000 | 1.72 / 1.34 |
| **A0** | **the same, NO SIGNAL** | 1.000, 0.984, 0.869, 0.994, 0.875 | **0.944** | **0.898** | 0.000 | 1.72 / 1.33 |
| B | joint scaling, fit on training noise | 0.526, 0.432, 0.459, 0.494, 0.534 | 0.489 | 0.506 | 0.714 | 1.71 / 1.73 |
| B0 | joint scaling, no signal | 0.615, 0.519, 0.503, 0.502, 0.530 | 0.534 | 0.535 | 0.504 | 1.71 / 1.71 |
| C | joint scaling, disjoint validation / test events | 0.582, 0.310, 0.507, 0.425, 0.496 | 0.464 | 0.512 | 0.715 | 1.71 / 1.73 |

The std-only column is the AUC of each series' standard deviation after scaling.
0.000 means perfect separation, with events having the *smaller* spread.

## Readings

1. **The published result reproduces.** Under their protocol, their pipeline on public
   data gives a mean accuracy of 0.903, or 0.877 with their own events, against the
   reported 0.919. Fold-to-fold variation (0.75–1.00) also matches theirs.
2. **It reproduces equally without any gravitational-wave signal** (A0: AUC 0.944,
   BA 0.898). The separation comes from scaling the two sets separately. After
   per-set min-max scaling, the 5,000-window noise set and the ~80-window event set
   end up with different spreads (1.72 vs 1.33). Any score that depends on spread
   separates them; a one-number standard-deviation detector separates them
   perfectly. This resolves the open item from the 2026-10-06 forensics: with real
   noise, the direction matches their filtered runs.
3. **With one scaler for all sets, their trained model detects nothing.** AUC 0.489
   (B) and 0.464 (C), indistinguishable from the no-signal rung (0.534). The events
   are detectable under fair scaling: the std-only detector reaches 0.714, and our
   registered run's classical detectors reach 0.74–0.80. Their circuit, trained their
   way (Powell; loss about 98% penalty), does not pick it up.
4. **Decomposition of the gap.** The paper's ~0.92 comes from the scaling artifact.
   Their model under fair scaling: ~0.49. Our registered, properly trained QVR under
   fair scaling: 0.721. Generic classical detectors: 0.74–0.80.

## Caveats

- Their noise file was never released. Ours follows the paper's recipe from GWOSC;
  filter design and window positions differ in detail.
- Shots are simulated as multinomial sampling; circuit numerics are exact otherwise.
- **Not checked yet:** why the 5,000-window noise set scales to a *wider* spread (1.72,
  close to the 1.81 of a uniform distribution) than the 83-window sets. That suggests
  light-tailed, bounded per-timepoint values in the filtered noise; a histogram would
  confirm it. The artifact needs only that the spreads differ, so the conclusions don't
  depend on the explanation.

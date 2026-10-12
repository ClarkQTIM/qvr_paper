# Proposed §12.4 — paper pipeline (to append to ANALYSIS_PLAN.md §12 before the first real run)

Status: PROPOSED 2026-10-12. Review, edit, then paste into `ANALYSIS_PLAN.md` §12
(above §11) and commit before `./run_all.sh` touches real data. Once committed,
changes need a §11 entry.

## 12.4.1 Pipeline

All datasets run through `run.py` (stages select → train → score → baselines →
teardown → report) with `configs/grid.json` at the commit that registers this
section. Datasets: MIT-BIH, Baker (`prepare.py baker` from `bi_train782`),
GW once noise data exist; synthetic and memory in the appendix.

## 12.4.2 Selection

- **QVR grid** (`configs/grid.json`):
  - univariate: layers {1, 3, 5} × N_E {1, 10} × cost {outside, inside} ×
    embedding {ry_all, ry_first} × lr {0.01, 0.003};
  - multichannel: embedding rx_per_qubit, ansatz {chain, sel} in place of the
    embedding axis;
  - selection seeds {0, 1}; winner = best mean validation AUC (ties → fewer
    layers, then fewer draws).
- **Final models:** seeds 0–4 × {best, reference config, best with σ frozen at 0}.
- **Classical methods** (`qvr/baselines.py`): each tuned on the same validation
  sets with its own grid: onenum, temporal, tz2, trig_t, moments, mahal, pca,
  knn, iforest.
- Test sets are read only by the `report` stage.

## 12.4.3 Registered predictions

| ID | Prediction | Pass criterion |
|---|---|---|
| P1 | Proposition 1–2 hold on every trained model (⟨Z⟩ exactly degree K, score term exactly 2K, not exact at K−1; D = 0 gives ⟨Z⟩ = cos x) | `fourier.all_hold_fraction` = 1.0 on every dataset |
| P2 | Proposition 3: analytic damping matches MC | included in P1 verdicts |
| P3 | Tuned QVR does not beat the best classical method | on every dataset, the paired CI of AUC(best classical) − AUC(qvr_best) has lower bound > −0.02 |
| P4 | QVR's own function class, fit classically, is at least as good | AUC(trig_t) ≥ AUC(qvr_best) − 0.02 on every dataset |
| P5 | D contributes little | AUC(qvr_best_identity) within 0.05 of AUC(qvr_best) on every main dataset that runs |
| P6 | Learning σ does not help | paired CI of AUC(qvr_sigma0) − AUC(qvr_best) contains 0 or is positive on every main dataset that runs |
| P7 | A better readout of the same circuit helps | max(readout AUCs) > AUC(qvr_best) on every main dataset that runs |
| P8 | The fingerprint carries nothing beyond six moments | probe macro AUC(p̄_b, mean over seeds) ≤ AUC(moments) + 0.02 where probes run |
| P9 | The protocol is clean | null guard max deviation from 0.5 ≤ 0.1 on every dataset (with ≥ 20 units) |

Main datasets = MIT-BIH, Baker, GW (if run). Any failed prediction is
reported as failed, with the numbers.

## 12.4.4 Known limitations, stated in advance

- Baker windows overlap and have no unit ids, so its CIs are per-window
  (optimistic).
- The GW event pool is 78 events, so validation and test anomalies are about 39
  each.
- Cervical (DYSIS) is not part of this study (decision 2026-10-12).

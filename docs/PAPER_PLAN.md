# Paper plan — QVR audit (living document)

*Updated 2026-10-12: cervical dropped; new minimal repo `qvr_paper`.*

Supersedes the per-experiment scripts as the organising document for the paper.
`ANALYSIS_PLAN.md` stays the frozen record of what was registered and when;
changes to the protocol below are logged there (§11 / §12) before results are
read.

---

## 1. What we want to show (the five claims)

| # | Claim | Evidence we have | Evidence still needed |
|---|---|---|---|
| C1 | **The published results don't hold up; the evaluation produced them.** | Baker: not reproduced (best 0.717 over about 400 models); pooled global scaling; a one-number baseline reaches the published BA. GW: separate per-set scaling plus thresholds tuned on the test anomalies; saved scores reproduced exactly from the authors' weights; signal-free surrogates "detect" equally well. | Server re-run of the GW forensics; the GW noise file (closes the filtered-noise offset). |
| C2 | **QVR is exactly a per-timepoint Fourier model with no memory.** | E-F1: ⟨Z⟩ is exactly degree K in each value, the score term exactly degree 2K, `identity` gives ⟨Z⟩ = cos x; the score is additive over t (Fact 2). Confirmed on the GW authors' trained weights (degree 1 / 2). p̄_b is a linear map of six moments (Fact 1). | E-F1 on every trained model of every dataset (automated in `run.py`). |
| C3 | **D, the draws and σ contribute little.** | In expectation the draws only damp coherences by exp(−½ Var·t²) (derived and verified). `identity` = full model on Baker. σ collapses under the paper objective and is uninformative relative to init (E5, MIT-BIH). | D-mode table on all datasets. **σ = 0 training ablation** (does learning σ help training at all?). |
| C4 | **Simple classical models match or beat QVR, including QVR's own reduced form.** | MIT-BIH: native 0.572 vs classical 0.70–0.72; Baker (normals-only scaling): `identity` = full. Test A: better readouts of the same circuit reach 0.70–0.75. | One table, all datasets, under one protocol, with QVR tuned (selection grid) and comparators given the same tuning budget. |
| C5 | **The QVR fingerprint carries nothing beyond six Fourier moments of the input.** | MIT-BIH: p̄_b 0.810 ≈ six moments 0.815 (raw beat 0.945); Fact 1 says it cannot do better. | Same probe on every dataset with subtypes, automated. |

Not claimed: anything about quantum hardware, the Fisher analysis, or sQVR
(follow-on paper; one discussion paragraph only).

## 2. Paper outline

*What Quantum Variational Rewinding Computes: A Fourier Account and a
Cross-Dataset Audit*

1. **Introduction.** QVR is promoted for time-series anomaly detection. We ask
   what it computes, whether reported results hold, and how it compares with
   simple baselines.
2. **Background.** Circuit and objective (short). Related work: encoding
   circuits compute Fourier series (Schuld, Sweke & Meyer 2021); QML
   benchmarking (Bowles, Ahmed & Schuld 2024). We claim the consequences for
   QVR, not the Fourier fact.
3. **What QVR computes (C2, C3, analytic).**
   - Proposition 1: ⟨Z⟩ is a degree-K trig polynomial per timepoint; the score
     term is degree 2K; the score is a sum over t.
   - Proposition 2: `identity` gives ⟨Z⟩ = cos x.
   - Proposition 3: expected draws = Gaussian damping of coherences;
     cost-inside collapses σ.
   - Corollary: every D mode lies in the same function class; tuning cannot
     leave it.
4. **Empirical confirmation (C2, C3).** E-F1 exactness on all trained models;
   damping check; D-mode table; σ = 0 ablation; σ and Hamiltonian statistics.
5. **Comparison under a correct protocol (C4).** Main table: datasets ×
   methods (tuned QVR, its D modes, best readout, Fourier surrogates,
   per-timepoint z², one-number baselines, standard detectors). Unit-level CIs,
   paired differences vs QVR. Signal-free null shows the protocol is clean.
6. **Subtype structure (C5).** Fingerprint vs six moments vs raw; unit-level
   permutation nulls.
7. **Evaluation forensics (C1).** Baker; GW.
8. **Discussion.** What a QVR-style model would need (memory, richer readout,
   higher-degree features); one paragraph on sQVR; an evaluation checklist for
   QML time series (scale on normals only, no threshold tuning on test
   anomalies, signal-free nulls, one-number baselines, unit-level statistics).
9. **Appendix.** Backend equivalence; per-dataset tables; synthetic and memory
   sanity sets; selection grid and chosen configs; reproduction attempts.

Figures: (1) circuit with the Fourier decomposition; (2) E-F1 R² by degree;
(3) damping (analytic vs simulated) and σ under inside/outside cost;
(4) main comparison; (5) Baker scaling forensics; (6) GW null.

## 3. Protocol (applies to every dataset)

- **Splits by independent unit** (patient, record, window block): `Xtr`
  (normals only), `Xval_norm` + `Xval` (validation, unit-disjoint from test),
  `Xte_norm` + `Xte_<set>` (test). `Xprobe` = anomalies from training units
  (subtype probes only).
- **Normalizer fit on `Xtr` only**, applied identically to every set.
- **Model selection on validation only.** QVR: the registered grid (§5). Each
  classical method: its own small grid, same rule (best validation AUC).
  Selection never reads test.
- **Final QVR:** the selected config over 5 seeds plus the fixed reference
  config (continuity with earlier records). The σ = 0 ablation uses the
  selected config.
- **Statistics:** AUC with a 95% unit-level cluster bootstrap; paired bootstrap
  of AUC differences vs QVR native; seeds summarised as mean ± SD plus the
  rank-averaged ensemble.
- **Null guard:** test normals split by unit into two halves, one half labelled
  "anomalous". Every method should give ≈ 0.5; if not, the pipeline leaks.

## 4. Datasets

| Dataset | Role | Channels | Status |
|---|---|---|---|
| Baker (crypto) | published claim (C1); main table with normals-only scaling | 2–3 (per-qubit embedding) | sets exist; needs `Xval_norm` carved from the 782 training normals |
| GW (QVR4GW) | published claim (C1); main table once noise data exist | 1 | forensics done on events + authors' weights; needs noise data |
| MIT-BIH | independent real data | 1 | split dir exists |
| Synthetic, memory | appendix sanity sets | 1 | split dirs exist |

## 5. QVR selection grid (to register in `ANALYSIS_PLAN.md` §12 before running)

Univariate: layers {1, 3, 5} × N_E {1, 10} × cost {outside, inside} ×
embedding {ry_all, ry_first} × lr {0.01, 0.003} = 48 configs × 2 seeds.
Multichannel (Baker): embedding fixed to per-qubit (one channel per qubit), so
24 configs × 2 seeds. Everything else as the reference config (k = Q, mc draws,
σ_λ = 1e-2, τ = 1, batch 8, ≤ 50,000 samples, monitor early stopping).

## 6. Repository design

One repo (`qvr_paper`), five entry points, one bash driver. `qvr_final` stays
frozen as the history (plan-v1 tag, dated progress records, exploratory scripts).

```
qvr_paper/
├── qvr/                      library (no CLIs except self-tests)
│   ├── circuit.py            the one circuit (+ GW options: rx_first, chain_full)
│   ├── train.py              trainer (+ freeze_sigma; git check before mkdir)
│   ├── infer.py              scoring under D modes, CRN draws
│   ├── data.py               Normalizer, t_grid, roc_auc, load_split (split dirs)
│   ├── baselines.py          classical suite: fit on Xtr, tune on val, score
│   ├── teardown.py           QVR analyses: Fourier exactness, damping, D modes,
│   │                         readouts, order, fingerprint probes, Hamiltonian
│   └── stats.py              unit bootstrap, paired bootstrap, permutation null
├── prepare.py                raw data → split dirs: mitbih | baker | gw | synthetic | memory
├── run.py                    one dataset, stages: select → train → score → baselines → teardown → report
├── collate.py                all datasets → paper tables (md + csv) and figures
├── run_all.sh                the driver: prepare (optional) → run.py per dataset → collate.py
├── configs/
│   ├── datasets.json         per dataset: split dir, channels, notes
│   └── grid.json             the registered QVR selection grid
├── forensics/
│   ├── baker_*.py            Baker: rebuild sets, E0 reproduction, rescore, data forensics
│   └── gw_*.py               QVR4GW: exact simulator, score reproduction, null test, Fourier check
├── tests/                    equivalence tests (unchanged)
└── docs/
```

### What each `run.py` stage writes (under `results/<dataset>/`)

| Stage | Output | Feeds |
|---|---|---|
| `select` | `select/<cfg>_s<seed>/` models, `select/selection.json` (val AUC per config, winner) | C4 |
| `train` | `models/{best,ref,sigma0}_s<seed>/` | all |
| `score` | `scores/<model>.npz`: every set × D mode, permuted copies, z_t, p̄_b, readouts | C3, C4 |
| `baselines` | `baselines/scores.npz`, `baselines/selection.json` | C4 |
| `teardown` | `teardown/<model>.json`: E-F1 exactness, damping, Hamiltonian/σ; `teardown/probes.json` | C2, C3, C5 |
| `report` | `summary.json`, `summary.md` (one table per claim) | collate |

`collate.py` reads every `results/*/summary.json` and writes `paper/tables/*.md|csv`
(T1 main comparison, T2 D modes + σ ablation, T3 readouts, T4 Fourier exactness,
T5 subtypes, T6 null guard) and `paper/figures/*.png`.

## 7. Reuse map

| New home | Comes from | Change |
|---|---|---|
| `qvr/circuit.py`, `infer.py` | as is | add `rx_first` embedding and `cnot_last` ansatz flag (GW circuit) |
| `qvr/train.py` | as is | `freeze_sigma` option; git check before creating the folder |
| `qvr/data.py` | `data.py` + `run_suite.load_split` / `anomaly_sets` + `fourier_moments` | multichannel-aware loader |
| `qvr/stats.py` | `run_suite._unit_boot`, `_rank_avg`; `score_audit.paired_boot`; `e4_probe` null | consolidated |
| `qvr/baselines.py` | `score_audit` comparators, `run_e6.untrained_scores`, `readout_test._mahal` | one fit/tune/score interface; new: kNN, PCA, IsolationForest, trig surrogates |
| `qvr/teardown.py` | `fourier_check.py`, `readout_test.readouts`, `run_suite.e4_probe` / `e5_hamiltonian` | functions only |
| `prepare.py` | `scripts/prepare_dataset.py` + new `baker`, `gw` converters | one CLI |
| `run.py` | `run_suite.py` (train/score/analyze pattern, resumable, worker pool) | stages above |
| `forensics/*` | `baker_e0/forensics/rescore`, `scripts/gw/*` | moved, not rewritten |
| not carried over | exploratory and cervical scripts, sQVR, old progress records | stay in `qvr_final` (history) |

## 8. Build order

1. Library modules with self-tests (`data`, `stats`, `baselines`, `teardown`;
   small changes to `circuit`, `train`).
2. `run.py` + configs; end-to-end on synthetic.
3. `prepare.py` (Baker conversion is new; GW waits for data).
4. `collate.py` + `run_all.sh`.
5. Register the grid (§5) in `ANALYSIS_PLAN.md` §12, commit, then run.

## 9. Open items

- GW noise data (authors or GWOSC). Until then GW enters C1 and C2 only.
- Cervical (DYSIS) dropped 2026-10-12: too costly to describe for what it adds;
  MIT-BIH carries the independent-real-data role. A further public dataset may be added.
- Baker: carve `Xval_norm` from the 782 training normals (by time block, so
  windows don't overlap across splits).
- E6 loose ends (prediction (e) numbers; wqvr 0.817 on the memory set): the
  memory set stays in the appendix until they are resolved.

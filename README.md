# qvr_paper — what Quantum Variational Rewinding computes

Everything needed to reproduce the paper, and nothing else: prepare each dataset,
train QVR and a classical comparison suite on it, take QVR apart, reproduce the
published pipelines, and collate the tables.

- Plan (claims, outline, protocol): [`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md)
- Registered predictions: `docs/ANALYSIS_PLAN.md` §12.4 (proposed text in
  [`docs/ANALYSIS_PLAN_12_4_proposed.md`](docs/ANALYSIS_PLAN_12_4_proposed.md))
- History (exploration, progress records, plan-v1): the `qvr_final` repo

## Reproduce

```bash
conda create -n qml python=3.12 && conda activate qml && pip install -r requirements.txt
SELFTEST=1 DATASETS="" ./run_all.sh                      # every self-test

# data → split folders (once)
python prepare.py mitbih    --db-dir <mitdb 1.0.0 dir> --out data_splits/mitbih
python forensics/baker_build_sets.py --repo-data <QuantumVariationalRewinding/data> --out-root data_raw/baker_sets
python prepare.py baker     --sets-dir data_raw/baker_sets/bi_train782 --out data_splits/baker
python prepare.py gw        --events <events.pkl> --noise <noise.pkl> --out data_splits/gw
python prepare.py synthetic --out data_splits/synthetic
python prepare.py memory    --out data_splits/memory

./run_all.sh                                              # all datasets, all stages, then tables
```

Outputs: `results/<dataset>/summary.md` (one per dataset) and `paper/tables/ALL.md`,
`paper/figures/`. Every stage resumes where it stopped. Environment variables:
`DATASETS`, `WORKERS`, `GRID`, `STAGES`, `RESULTS`, `PAPER`, `STRICT=1` (refuse a dirty
git tree). Smoke test in minutes: `GRID=configs/grid_quick.json RESULTS=results_quick
PAPER=paper_quick DATASETS=synthetic ./run_all.sh`.

Forensics (published pipelines), run separately:

```bash
python forensics/baker_e0.py --out results/forensics/baker_e0          # reproduction attempt
python forensics/baker_forensics.py ...                                # scaling / provenance audit
python forensics/gw_repro.py   <QVR4GW clone>                          # their scores from their weights
python forensics/gw_null_test.py <QVR4GW clone>                        # signal-free surrogates
python forensics/gw_fourier.py <QVR4GW clone>                          # their circuit is a + b cos x + c sin x
```

## Layout

```
qvr/            library — every module self-tests with python -m qvr.<module>
  circuit.py    the one circuit; matrix backend; D modes (mc, mean, identity, averaged)
  train.py      seeded trainer with manifests; freeze_sigma for the σ = 0 ablation
  infer.py      scores, ⟨Z⟩, p̄_b under any D mode; common random numbers
  data.py       Normalizer, t_grid, roc_auc, load_split / write_split, fourier_moments
  baselines.py  9 classical methods: fit on Xtr, tuned on validation
  teardown.py   Fourier exactness, damping, readouts, Hamiltonian/σ, subtype probes
  stats.py      unit-level bootstrap, paired differences, rank ensembles, null guard
prepare.py      raw data → split folders (mitbih, baker, gw, synthetic, memory)
run.py          one dataset: select → train → score → baselines → teardown → report
report.py       summary.json / summary.md for one dataset
collate.py      all datasets → paper/tables (T1–T6), paper/figures
run_all.sh      the driver
configs/        datasets.json, grid.json (registered), grid_quick.json (tests only)
forensics/      reproductions of the published Baker and QVR4GW pipelines
tests/          backend equivalence vs the old qvr_cervix code and Baker's notebook
docs/           plan and registration
```

## Split folder (how any dataset enters)

Pickles `(N, d, T)` float64, normalized to [−π, π] by a normalizer fit on `Xtr` only:
`Xtr` (training normals), `Xval_norm` / `Xval` (validation normals / anomalies),
`Xte_norm` / `Xte_<name>` (test), optional `Xprobe` (anomalies from training units,
probes only), `<set>_units`, `<set>_subtype` (`NA` = unknown), `manifest.json`.
Read with `qvr.data.load_split`, write with `qvr.data.write_split`.

## Protocol

Splits by independent unit · scaling fit on training normals only · every
hyperparameter (QVR grid and each classical method's grid) chosen on validation ·
test read once, by `report` · unit-level cluster-bootstrap CIs · null guard (test
normals vs test normals) must be ≈ 0.5 for every method.

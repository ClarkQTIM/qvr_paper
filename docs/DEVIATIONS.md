# Deviations from the registration

Every change to the registered protocol (`docs/REGISTRATION.md`, tag `prereg-v1`)
after it was frozen. Each entry says when it was made, what had been seen at that
point, and how results are reported under both the registered and the amended
definition. Entries are never edited after commit; corrections get a new entry.

---

## D1 — 2026-10-14 — P3: "best classical method" was undefined

**Seen at the time:** MIT-BIH results (first dataset). Nothing from Baker or GW.

**Problem.** P3 ("tuned QVR does not beat the best classical method") did not say
how the best classical method is chosen. The two natural readings disagree on
MIT-BIH:

| Reading | Method | AUC(method) − AUC(QVR) [95% unit CI] | P3 |
|---|---|---|---|
| best on **validation** | pca (val AUC 0.865) | +0.022 [−0.079, +0.146] | fails (lower bound < −0.02) |
| best on **test** | mahal (test AUC 0.775) | +0.086 [+0.010, +0.216] | holds |

**Amendment.** The best classical method is the one with the highest
**validation** AUC, consistent with the rest of the protocol (nothing is chosen on
test). The test-best reading is still reported alongside, labelled as such.

**MIT-BIH under the amendment:** P3 **fails**. The interval is too wide (22 test
records) to rule out QVR being up to 0.08 better than PCA. Reported as failed, with
the explanation.

## D2 — 2026-10-14 — P9: the null-guard threshold was miscalibrated

**Seen at the time:** MIT-BIH results.

**Registered result: P9 fails on MIT-BIH** (max deviation 0.118 > 0.10, 22 units).
The largest deviations come from the "atypicality" methods (mahal 0.618, knn 0.593,
iforest 0.591, trig_t 0.591, QVR ⟨Z⟩ readout 0.591), all in the same direction, with
`identity` mirrored at 0.385. This is consistent with one random half of the test
patients having slightly more unusual beats than the other.

**Calibration (exploratory, run after seeing the result).** The same statistic over
200 random unit splits of MIT-BIH test normals, computed by `report.py` over all 18
methods: median 0.162, 95th percentile 0.273. The observed 0.118 lies at the **16th
percentile**. (A first ad hoc check over 11 methods gave median 0.156, 95th 0.268,
22nd percentile; same conclusion.) A split of 22 patients
routinely produces larger deviations by chance. A fixed 0.10 threshold could not be
met at this unit count. The threshold was a registration error, not evidence of
leakage.

**Amendment.** From Baker onwards, P9 is also judged against its own calibration:
the observed deviation must be below the 95th percentile of 200 random unit splits.
`report.py` now computes this automatically. Both the registered verdict (≤ 0.10)
and the calibrated verdict are reported for every dataset.

**MIT-BIH:** registered P9 **fails**; calibrated P9 **holds**.

## D3 — 2026-10-14 — report code changed after `prereg-v1`

`report.py` and `collate.py` gained:
- the mechanical scorecard (P1–P9, with D1 and D2);
- the null-guard calibration;
- table T7;
- a cosmetic title fix.

No training, scoring, baseline or teardown code changed. MIT-BIH models and score
files are untouched; only the `report` stage was re-run to regenerate
`summary.json` / `summary.md`.

## D4 — 2026-10-14 — environment numerics (note, not a protocol change)

`prepare.py mitbih` in the pinned pyenv environment reproduces the earlier
`qvr_final` split with identical labels, units, subtypes and shapes. Float values
differ at about 1e-13 (normalizer bounds; zero-phase filtering numerics), so byte
hashes differ. Exact reproduction requires `environment.lock.txt`.

## D5 — 2026-10-29 — GW dataset defined (written before any GW data is prepared or run)

**Seen at the time:** MIT-BIH and Baker results; the QVR4GW paper text. No GW data
under our protocol.

**Change.** The registration listed GW "once noise data exist". The authors' noise
file is not in their repository, and we did not request it. GW is instead rebuilt
from public LIGO open data (GWOSC) by `prepare.py gw-gwosc`, following the recipe in
the paper (Rodrigues de Miranda et al., QMI 2025, Secs. 6 and 8):
- L1, observing run O3 (GPS 1238166018–1269363618), 1-s windows at 4096 Hz;
- events: every GWOSC GWTC-2.1 / GWTC-3 catalog event in O3 with L1 data (confident,
  marginal, auxiliary), placed off-centre by U(−0.25, 0.25) s;
- noise: windows starting at uniform random times 100–999 s after an event;
- filter: band-pass 50–250 Hz plus notches at 60/120/180/240 Hz.

**Choices the paper leaves open, fixed here:**
- Filter design: Butterworth order 8, zero-phase; notch Q = 30; applied to the padded
  segment before cutting.
- Units: an event's neighbourhood (the event plus its noise windows), split
  40/30/30 into train/val/test. Validation and test anomalies are **disjoint**,
  unlike the paper. Training-unit events are unused.
- 100 noise windows per event; decimation ×4 (anti-aliased FIR) to T = 1024. This
  is lossless for content ≤ 250 Hz (Nyquist 512 Hz) and makes the registered grid
  computationally feasible.
- One normalizer (per-timepoint 0.5/99.5% quantiles) fit on training noise only.

**Unchanged:** the registered grid, protocol and predictions P1–P9 (P8 not
applicable; no subtypes). The event count may differ from the paper's 78, depending
on current GWOSC catalogs; the manifest records the list. The forensic GW results
(the authors' saved scores and weights) are separate and unaffected.

## D6 — 2026-10-30 — collate.py presentation changes (note, not a protocol change)

**Seen at the time:** all three registered runs (MIT-BIH, Baker, GW) and the GW ladder.

`collate.py` only; no training, scoring, baseline, teardown or report code changed, and
no number changes. T1 gains the mean-⟨Z⟩ Mahalanobis readout row (already in T3). T7
reports P9 as two rows, the registered threshold and the calibrated verdict (D2). Columns
are ordered MIT-BIH, Baker, GW. New T8 collects the published-pipeline forensics
(`forensics/baker_e0.py`, `forensics/gw_ladder.py`) when their outputs exist.

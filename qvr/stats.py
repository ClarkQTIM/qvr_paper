"""
qvr/stats.py — the statistics every comparison in the paper uses.

All resampling is over INDEPENDENT UNITS (patient, record, window block): a unit's
series are kept together, so correlated series don't shrink the intervals.

  auc_ci          AUC + 95% unit-level cluster-bootstrap CI
  paired_diff_ci  AUC(s1) − AUC(s2) on the same series, same resamples
  rank_avg        seed ensemble: mean of within-set ranks
  null_halves     null guard: split normal units into two halves, label one
                  "anomalous" → every method should give AUC ≈ 0.5

Self-test:  python -m qvr.stats
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from qvr.data import roc_auc


def _unit_index(u):
    u = np.asarray(u).astype(str)
    uu, inv = np.unique(u, return_inverse=True)
    return uu, [np.flatnonzero(inv == i) for i in range(len(uu))]


def _resamples(u, n_boot, seed):
    uu, idx = _unit_index(u)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uu), len(uu))
        yield np.concatenate([idx[i] for i in pick])


def auc_ci(y, s, u, n_boot: int = 1000, seed: int = 0) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    vals = [roc_auc(y[ix], s[ix]) for ix in _resamples(u, n_boot, seed) if len(np.unique(y[ix])) == 2]
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return {'auc': roc_auc(y, s), 'lo': float(lo), 'hi': float(hi), 'n_units': int(len(np.unique(u)))}


def paired_diff_ci(y, s1, s2, u, n_boot: int = 1000, seed: int = 0) -> dict:
    """Same resamples for both scores (seed fixed) → paired CI of the AUC difference."""
    y, s1, s2 = np.asarray(y), np.asarray(s1, float), np.asarray(s2, float)
    vals = [roc_auc(y[ix], s1[ix]) - roc_auc(y[ix], s2[ix])
            for ix in _resamples(u, n_boot, seed) if len(np.unique(y[ix])) == 2]
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return {'diff': roc_auc(y, s1) - roc_auc(y, s2), 'lo': float(lo), 'hi': float(hi)}


def rank_avg(arrays) -> np.ndarray:
    """Seed ensemble that is scale-free: mean of normalized ranks (computed on the SAME
    concatenated evaluation vector for every seed, so ranks are comparable)."""
    return np.mean([rankdata(a) / len(a) for a in arrays], axis=0)


def null_halves(u, seed: int = 0) -> np.ndarray:
    """Labels for the null guard: units split at random into halves; one half → 1."""
    uu, inv = np.unique(np.asarray(u).astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    flag = np.zeros(len(uu), int)
    flag[rng.permutation(len(uu))[: len(uu) // 2]] = 1
    return flag[inv]


def _self_test() -> None:
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    print('=' * 64)
    print('qvr.stats self-test')
    print('=' * 64)
    rng = np.random.default_rng(0)
    # 40 units, 25 series each; unit effect makes series within a unit correlated
    u = np.repeat([f'u{i}' for i in range(40)], 25)
    y = np.repeat(rng.integers(0, 2, 40), 25)
    eff = np.repeat(rng.normal(size=40), 25)
    s = 0.6 * y + eff + 0.3 * rng.normal(size=len(y))
    r = auc_ci(y, s, u, n_boot=400)
    naive = auc_ci(y, s, np.arange(len(y)).astype(str), n_boot=400)
    check('AUC inside its CI', r['lo'] <= r['auc'] <= r['hi'], f"{r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    check('unit CI wider than series-level CI (clustering respected)',
          (r['hi'] - r['lo']) > 1.5 * (naive['hi'] - naive['lo']),
          f"{r['hi'] - r['lo']:.3f} vs {naive['hi'] - naive['lo']:.3f}")
    d = paired_diff_ci(y, s, s, u, n_boot=200)
    check('paired diff of a score with itself = 0 exactly', d['diff'] == 0 and d['lo'] == 0 and d['hi'] == 0)
    d2 = paired_diff_ci(y, s, -s, u, n_boot=200)
    check('paired diff vs reversed score > 0', d2['lo'] > 0, f"{d2['diff']:.3f} [{d2['lo']:.3f}, {d2['hi']:.3f}]")
    ra = rank_avg([s, s * 10 + 3])
    check('rank_avg is scale-free', roc_auc(y, ra) == roc_auc(y, s))
    lab = null_halves(u, seed=1)
    per_unit = {k: set(lab[u == k]) for k in np.unique(u)}
    check('null halves: one label per unit, half the units', all(len(v) == 1 for v in per_unit.values())
          and abs(lab.mean() - 0.5) < 0.05)
    null_aucs = [roc_auc(null_halves(u, seed=k), rng.normal(size=len(u))) for k in range(50)]
    check('null of a random score ≈ 0.5', abs(np.mean(null_aucs) - 0.5) < 0.03, f'{np.mean(null_aucs):.3f}')
    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()

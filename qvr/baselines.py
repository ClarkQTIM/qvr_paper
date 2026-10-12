"""
qvr/baselines.py — the classical comparison suite.

Every method is FIT ON TRAINING NORMALS ONLY, has a small hyperparameter grid that
is selected on validation (Xval_norm vs Xval) by AUC — the same rule QVR's grid
uses — and returns scores where higher = more anomalous.

  onenum    ONE number per series (mean, std, max|x|, last value) of one channel,
            two-sided robust z vs training normals.     grid: stat × channel
  temporal  one number that needs order: lag-1 autocorrelation or mean squared
            successive difference.                       grid: stat × channel
  tz2       mean over t (and channels) of per-timepoint z²: QVR's additive-over-time
            structure with a Gaussian per timepoint.     no grid
  trig_t    QVR's own function class, classically: per timepoint, Mahalanobis on
            [cos kx_t, sin kx_t]_{k≤K} (all channels), averaged over t. A quadratic
            form in degree-K trig features is a degree-2K trig polynomial per
            timepoint — exactly the class Proposition 1 puts QVR in.   grid: K
  moments   Mahalanobis on series-averaged trig moments (what p̄_b is a linear map
            of; "the fingerprint, classically").         grid: K
  mahal     Mahalanobis on the whole series (Ledoit–Wolf shrinkage).   no grid
  pca       PCA reconstruction error.                    grid: n components
  knn       mean distance to the k nearest training normals.   grid: k
  iforest   Isolation Forest.                            grid: max_samples

Inputs: (N, T) or (N, T, d) arrays/tensors, already normalized.

Self-test:  python -m qvr.baselines
"""

from __future__ import annotations

import numpy as np

from qvr.data import roc_auc

METHODS = ('onenum', 'temporal', 'tz2', 'trig_t', 'moments', 'mahal', 'pca', 'knn', 'iforest')
KNN_TRAIN_CAP = 5000


def _np3(X) -> np.ndarray:
    a = np.asarray(X.detach().numpy() if hasattr(X, 'detach') else X, dtype=np.float64)
    return a[..., None] if a.ndim == 2 else a


def _robust_z(train, x):
    med = np.median(train)
    mad = np.median(np.abs(train - med))
    return np.abs(x - med) / (mad if mad > 0 else (np.std(train) or 1.0))


class _Gauss:
    """Mahalanobis distance with Ledoit–Wolf shrinkage (stable for any dimension)."""

    def __init__(self, F):
        from sklearn.covariance import LedoitWolf
        self.mu = F.mean(0)
        self.P = LedoitWolf().fit(F).precision_

    def __call__(self, F):
        D = F - self.mu
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', D, self.P, D), 0))


def _series_stat(A, stat):            # A: (N, T) one channel
    if stat == 'mean':
        return A.mean(1)
    if stat == 'std':
        return A.std(1)
    if stat == 'absmax':
        return np.abs(A).max(1)
    if stat == 'last':
        return A[:, -1]
    if stat == 'lag1':
        a, b = A[:, :-1] - A[:, :-1].mean(1, keepdims=True), A[:, 1:] - A[:, 1:].mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    if stat == 'msd':
        return (np.diff(A, axis=1) ** 2).mean(1)
    raise ValueError(stat)


def _trig(A, K):                      # A: (N, T, d) → (N, T, 2K·d)
    return np.concatenate([f(k * A) for k in range(1, K + 1) for f in (np.cos, np.sin)], axis=-1)


# ---------------------------------------------------------------------------
# Grids and fitting
# ---------------------------------------------------------------------------

def grid(name: str, d: int, T: int, n_train: int) -> list[dict]:
    if name == 'onenum':
        return [{'stat': s, 'channel': c} for s in ('mean', 'std', 'absmax', 'last') for c in range(d)]
    if name == 'temporal':
        return [{'stat': s, 'channel': c} for s in ('lag1', 'msd') for c in range(d)]
    if name in ('tz2', 'mahal'):
        return [{}]
    if name in ('trig_t', 'moments'):
        return [{'K': k} for k in (1, 2, 3)]
    if name == 'pca':
        cap = min(T * d, n_train) - 1
        return [{'n': n} for n in (2, 5, 10, 20) if n <= cap] or [{'n': 1}]
    if name == 'knn':
        return [{'k': k} for k in (1, 5, 20) if k < n_train]
    if name == 'iforest':
        return [{'max_samples': m} for m in (256, 1024)]
    raise ValueError(f'unknown method {name!r}; choose from {METHODS}')


def fit(name: str, Xtr, **hp):
    """Fit on training normals → scorer(X) → (N,) scores (higher = more anomalous)."""
    A = _np3(Xtr)
    N, T, d = A.shape
    if name in ('onenum', 'temporal'):
        c, st = hp['channel'], hp['stat']
        ref = _series_stat(A[..., c], st)
        return lambda X: _robust_z(ref, _series_stat(_np3(X)[..., c], st))
    if name == 'tz2':
        mu, sd = A.mean(0), A.std(0) + 1e-9
        return lambda X: (((_np3(X) - mu) / sd) ** 2).mean((1, 2))
    if name == 'trig_t':
        K = hp['K']
        F = _trig(A, K)                                            # (N, T, p)
        gs = [_Gauss(F[:, t]) for t in range(T)]
        return lambda X: np.mean([g(f) for g, f in zip(gs, np.moveaxis(_trig(_np3(X), K), 1, 0))], 0)
    if name == 'moments':
        K = hp['K']
        g = _Gauss(_trig(A, K).mean(1))
        return lambda X: g(_trig(_np3(X), K).mean(1))
    if name == 'mahal':
        g = _Gauss(A.reshape(N, -1))
        return lambda X: g(_np3(X).reshape(len(X), -1))
    if name == 'pca':
        from sklearn.decomposition import PCA
        p = PCA(n_components=hp['n'], random_state=0).fit(A.reshape(N, -1))

        def score(X):
            F = _np3(X).reshape(len(X), -1)
            return ((F - p.inverse_transform(p.transform(F))) ** 2).sum(1)
        return score
    if name == 'knn':
        from sklearn.neighbors import NearestNeighbors
        F = A.reshape(N, -1)
        if N > KNN_TRAIN_CAP:
            F = F[np.random.default_rng(0).choice(N, KNN_TRAIN_CAP, replace=False)]
        nn = NearestNeighbors(n_neighbors=hp['k']).fit(F)
        return lambda X: nn.kneighbors(_np3(X).reshape(len(X), -1))[0].mean(1)
    if name == 'iforest':
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=200, max_samples=min(hp['max_samples'], N), random_state=0)
        m.fit(A.reshape(N, -1))
        return lambda X: -m.score_samples(_np3(X).reshape(len(X), -1))
    raise ValueError(f'unknown method {name!r}')


def select_and_score(Xtr, sets: dict, val: tuple | None = None, methods=METHODS) -> tuple[dict, dict]:
    """For each method: fit every grid point on Xtr, pick the best validation AUC
    (val = (X_val_normal, X_val_anomalous); without val the first grid point is used
    and flagged), score every set in `sets`.
    Returns scores {method: {set: (N,)}} and report {method: {hp, val_auc, grid}}."""
    A = _np3(Xtr)
    N, T, d = A.shape
    scores, report = {}, {}
    if val is not None:
        Vn, Va = val
        y_val = np.r_[np.zeros(len(Vn)), np.ones(len(Va))]
    for m in methods:
        g = grid(m, d, T, N)
        tried = []
        best = (None, -np.inf, None)
        for hp in g:
            scorer = fit(m, A, **hp)
            if val is None:
                best = (hp, np.nan, scorer)
                break
            auc = roc_auc(y_val, np.r_[scorer(Vn), scorer(Va)])
            tried.append({'hp': hp, 'val_auc': auc})
            if auc > best[1]:
                best = (hp, auc, scorer)
        hp, auc, scorer = best
        scores[m] = {k: scorer(v) for k, v in sets.items()}
        report[m] = {'hp': hp, 'val_auc': float(auc), 'grid': tried, 'selected_on_val': val is not None}
    return scores, report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    print('=' * 70)
    print('qvr.baselines self-test')
    print('=' * 70)
    rng = np.random.default_rng(0)
    T = 24
    base = np.sin(np.linspace(0, 2 * np.pi, T))

    def normals(n):
        return base + 0.3 * rng.normal(size=(n, T))
    Xtr, Vn, Tn = normals(400), normals(100), normals(150)
    Va = normals(80) + 0.8 * np.linspace(0, 1, T)            # drift anomaly
    Ta = normals(120) + 0.8 * np.linspace(0, 1, T)
    sets = {'Xte_norm': Tn, 'Xte_drift': Ta}
    sc, rep = select_and_score(Xtr, sets, val=(Vn, Va))
    y = np.r_[np.zeros(150), np.ones(120)]
    print(f"  {'method':9s} {'hp':28s} {'val':>6s} {'test':>6s}")
    for m in METHODS:
        auc = roc_auc(y, np.r_[sc[m]['Xte_norm'], sc[m]['Xte_drift']])
        print(f"  {m:9s} {str(rep[m]['hp']):28s} {rep[m]['val_auc']:6.3f} {auc:6.3f}")
        check(f'{m}: finite scores, one per series',
              all(np.isfinite(v).all() and len(v) == len(sets[k]) for k, v in sc[m].items()))
    auc = lambda m: roc_auc(y, np.r_[sc[m]['Xte_norm'], sc[m]['Xte_drift']])  # noqa: E731
    check('drift is easy for tz2 / mahal / onenum(mean)', min(auc('tz2'), auc('mahal'), auc('onenum')) > 0.9)
    check('onenum picks the mean (drift raises it)', rep['onenum']['hp']['stat'] in ('mean', 'last'))
    check('selection = argmax of the grid', all(
        rep[m]['val_auc'] == max(g['val_auc'] for g in rep[m]['grid']) for m in METHODS))
    # order-only anomaly: same values, shuffled in time → per-timepoint marginals differ, but
    # a series-level one-number like the mean cannot see a pure permutation
    Ps = np.array([r[rng.permutation(T)] for r in normals(120)])
    sc2, _ = select_and_score(Xtr, {'n': Tn, 'p': Ps}, val=None, methods=('onenum', 'tz2'))
    y2 = np.r_[np.zeros(150), np.ones(120)]
    a_one = roc_auc(y2, np.r_[sc2['onenum']['n'], sc2['onenum']['p']])
    a_tz2 = roc_auc(y2, np.r_[sc2['tz2']['n'], sc2['tz2']['p']])
    check('shuffle anomaly: per-timepoint tz2 sees it, series mean does not',
          a_tz2 > 0.9 and abs(a_one - 0.5) < 0.1, f'tz2 {a_tz2:.3f}, mean {a_one:.3f}')
    # multichannel
    X3 = rng.normal(size=(200, 10, 3))
    s3, r3 = select_and_score(X3, {'a': rng.normal(size=(20, 10, 3))}, val=None)
    check('multichannel (N, T, 3) runs for every method', all(len(s3[m]['a']) == 20 for m in METHODS))
    check('without val, selection is flagged', not r3['tz2']['selected_on_val'])
    print('=' * 70)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()

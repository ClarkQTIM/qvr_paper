"""
qvr/data.py — dataset-agnostic data layer.

SeriesSet   raw series + labels + independent-unit ids (patient, record, window)
Normalizer  fit on TRAINING NORMALS only; transform any split. Reports how many
            values land outside [-π, π] (p_b is 2π-periodic, so those alias).
t_grid      the time grid fed to D(ε, t): linspace(0, 2π, T), as in qvr_cervix.
roc_auc     rank-based AUC (ties averaged), no sklearn needed.

Datasets enter as split directories (prepare.py); load_split / write_split read and write them.

Normalizer kinds
  'per_timepoint'  legacy-exact qvr_cervix: each frame t≥1 mapped with its own
                   train min/max to [-π, π]; frame 0 left as-is (0 after
                   baseline subtraction). Use from_legacy() for old checkpoints.
  'per_timepoint_q' same, but with train quantiles (lo_q, hi_q) instead of
                   min/max, and optional clipping — for new datasets where
                   outliers would otherwise compress the range.

Self-test:  python -m qvr.data
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class SeriesSet:
    X: torch.Tensor                 # (N, T) raw (un-normalized) series, float64
    y: np.ndarray                   # (N,) int: 0 = normal, 1 = anomalous (task-level)
    unit: np.ndarray                # (N,) independent unit id (patient / record / window)
    name: str = ''
    extra: dict = field(default_factory=dict)   # optional per-series arrays (e.g. subtype)

    def __post_init__(self):
        self.X = torch.as_tensor(self.X, dtype=DTYPE)
        self.y = np.asarray(self.y).astype(int)
        self.unit = np.asarray(self.unit)
        n = self.X.shape[0]
        if len(self.y) != n or len(self.unit) != n:
            raise ValueError(f'{self.name}: X has {n} rows, y {len(self.y)}, unit {len(self.unit)}')
        for k, v in self.extra.items():
            if len(v) != n:
                raise ValueError(f'{self.name}: extra[{k!r}] has {len(v)} rows, expected {n}')

    @property
    def T(self) -> int:
        return self.X.shape[1]

    def normals(self) -> torch.Tensor:
        return self.X[self.y == 0]

    def subset(self, mask) -> 'SeriesSet':
        mask = np.asarray(mask)
        return SeriesSet(self.X[torch.as_tensor(mask)], self.y[mask], self.unit[mask],
                         self.name, {k: np.asarray(v)[mask] for k, v in self.extra.items()})

    def summary(self) -> str:
        n1 = int(self.y.sum())
        return (f'{self.name}: N={len(self.y):,} (normal {len(self.y) - n1:,}, anomalous {n1:,}), '
                f'T={self.T}, units={len(np.unique(self.unit)):,}')


def t_grid(T: int) -> torch.Tensor:
    return torch.linspace(0, 2 * math.pi, T, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    def __init__(self, kind: str, lo=None, hi=None, skip_first: bool = True,
                 clip: bool = False, q: tuple[float, float] | None = None):
        self.kind, self.skip_first, self.clip, self.q = kind, skip_first, clip, q
        self.lo = None if lo is None else torch.as_tensor(lo, dtype=DTYPE)
        self.hi = None if hi is None else torch.as_tensor(hi, dtype=DTYPE)

    @classmethod
    def fit(cls, X_train_normals: torch.Tensor, kind: str = 'per_timepoint',
            skip_first: bool = True, clip: bool = False,
            q: tuple[float, float] = (0.005, 0.995)) -> 'Normalizer':
        X = torch.as_tensor(X_train_normals, dtype=DTYPE)
        Xs = X[:, 1:] if skip_first else X
        if kind == 'per_timepoint':
            return cls(kind, Xs.min(0).values, Xs.max(0).values, skip_first, clip)
        if kind == 'per_timepoint_q':
            lo = torch.quantile(Xs, q[0], dim=0)
            hi = torch.quantile(Xs, q[1], dim=0)
            return cls(kind, lo, hi, skip_first, clip, q)
        raise ValueError(f'unknown normalizer kind {kind!r}')

    @classmethod
    def from_legacy(cls, norm_config: dict) -> 'Normalizer':
        """From an old checkpoint's norm_config (norm='per_timepoint')."""
        if norm_config.get('norm') != 'per_timepoint':
            raise ValueError(f"legacy norm {norm_config.get('norm')!r} not supported")
        p = norm_config['norm_params']
        return cls('per_timepoint', p['per_tp_min'], p['per_tp_max'], skip_first=True)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=DTYPE)
        out = X.clone()
        sl = slice(1, None) if self.skip_first else slice(None)
        z = 2 * math.pi * (X[:, sl] - self.lo) / (self.hi - self.lo) - math.pi
        if self.clip:
            z = z.clamp(-math.pi, math.pi)
        out[:, sl] = z
        return out

    def out_of_range(self, Xn: torch.Tensor, tol: float = 1e-12) -> float:
        """Fraction of normalized values with |x| > π (aliasing risk)."""
        return float((Xn.abs() > math.pi + tol).double().mean())

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'lo': self.lo.tolist(), 'hi': self.hi.tolist(),
                'skip_first': self.skip_first, 'clip': self.clip, 'q': self.q}

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        return cls(d['kind'], d['lo'], d['hi'], d['skip_first'], d['clip'],
                   tuple(d['q']) if d.get('q') else None)


# ---------------------------------------------------------------------------
# Split directories (README "Drop-in dataset layout")
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """All sets of one split directory, already normalized.
    X[name]: (N, T) float64 for one channel, (N, T, d) for several."""
    X: dict
    units: dict
    subtype: dict
    manifest: dict
    path: str = ''

    @property
    def T(self) -> int:
        return int(self.X['Xtr'].shape[1])

    @property
    def n_channels(self) -> int:
        return 1 if self.X['Xtr'].ndim == 2 else int(self.X['Xtr'].shape[2])

    def test_sets(self) -> list:
        """Anomalous test sets, sorted (Xte_<name>, excluding Xte_norm)."""
        return sorted(k for k in self.X if k.startswith('Xte_') and k != 'Xte_norm')

    def has_val(self) -> bool:
        return 'Xval' in self.X and 'Xval_norm' in self.X

    def pair(self, normal: str, anomalous) -> tuple:
        """Concatenate a normal set and one or more anomalous sets → (keys, y, units)
        where keys lists (set, n) blocks in order, for slicing score dicts."""
        anomalous = [anomalous] if isinstance(anomalous, str) else list(anomalous)
        keys = [normal] + anomalous
        y = np.concatenate([np.full(len(self.X[k]), int(k != normal)) for k in keys])
        # same unit id in a normal and an anomalous set = same cluster (e.g. one MIT-BIH record)
        u = np.concatenate([self.units[k].astype(str) for k in keys])
        return keys, y, u


def load_split(data_dir, channels=None) -> Split:
    """Read X*.pickle (N, d, T) arrays. channels: None = all, or a list of indices.
    A single selected channel is returned as (N, T)."""
    import json
    import pickle
    d = Path(data_dir)
    if not (d / 'Xtr.pickle').exists():
        raise FileNotFoundError(f'{d}: no Xtr.pickle — not a split directory')
    X, units, sub = {}, {}, {}
    for p in sorted(d.glob('X*.pickle')):
        with open(p, 'rb') as f:
            a = np.asarray(pickle.load(f))
        name = p.stem
        if name.endswith('_units'):
            units[name[:-6]] = a.astype(str)
        elif name.endswith('_subtype'):
            sub[name[:-8]] = a.astype(str)
        elif a.ndim == 3:
            a = a.astype(np.float64)
            if channels is not None:
                a = a[:, list(channels), :]
            a = np.transpose(a, (0, 2, 1))                       # (N, T, d)
            X[name] = torch.as_tensor(a[..., 0] if a.shape[2] == 1 else a)
        else:
            raise ValueError(f'{p}: expected (N, d, T), got shape {a.shape}')
    for k in X:
        # no unit file: each series is its own unit, ids unique across sets
        units.setdefault(k, np.array([f'{k}#{i}' for i in range(len(X[k]))]))
        if len(units[k]) != len(X[k]):
            raise ValueError(f'{k}: {len(X[k])} series but {len(units[k])} unit ids')
    man_p = d / 'manifest.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    return Split(X, units, sub, man, str(d))


def write_split(out, sets: dict, units: dict | None = None, subtype: dict | None = None,
                manifest: dict | None = None) -> None:
    """Inverse of load_split. sets[name]: (N, T) or (N, T, d) → stored (N, d, T)."""
    import json
    import pickle
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        a = np.asarray(v, dtype=np.float64)
        a = a[:, None, :] if a.ndim == 2 else np.transpose(a, (0, 2, 1))
        with open(out / f'{k}.pickle', 'wb') as f:
            pickle.dump(a, f)
    for tag, dd in (('units', units or {}), ('subtype', subtype or {})):
        for k, v in dd.items():
            with open(out / f'{k}_{tag}.pickle', 'wb') as f:
                pickle.dump(np.asarray(v).astype(str), f)
    (out / 'manifest.json').write_text(json.dumps(manifest or {}, indent=1, default=str))


def fourier_moments(X, K: int = 3) -> np.ndarray:
    """Series-averaged trig moments mean_t cos/sin(k x_t), k = 1..K, per channel.
    (N, T) → (N, 2K);  (N, T, d) → (N, 2K·d). These are what p̄_b is a linear map of."""
    X = torch.as_tensor(X, dtype=DTYPE)
    if X.ndim == 2:
        X = X.unsqueeze(-1)
    F = [f(k * X).mean(1) for k in range(1, K + 1) for f in (torch.cos, torch.sin)]   # each (N, d)
    return torch.cat(F, 1).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc(y, s) -> float:
    """AUC with average ranks for ties. y ∈ {0,1}; higher s = more anomalous."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    g = torch.Generator().manual_seed(0)
    N, T = 1000, 17
    X = torch.randn(N, T, generator=g, dtype=DTYPE) * 0.1
    X[:, 0] = 0.0
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    unit = np.repeat(np.arange(50), 20)

    print('=' * 64)
    print('qvr.data self-test')
    print('=' * 64)

    print('SeriesSet')
    s = SeriesSet(X, y, unit, name='fake', extra={'grade': np.arange(N) % 4})
    print('  ' + s.summary())
    check('normals shape', s.normals().shape == (800, T), f'{tuple(s.normals().shape)}')
    sub = s.subset(s.y == 1)
    check('subset keeps extra aligned', len(sub.extra['grade']) == 200 and sub.y.all())
    try:
        SeriesSet(X, y[:10], unit)
        check('length mismatch raises', False)
    except ValueError:
        check('length mismatch raises', True)
    check('t_grid', torch.allclose(t_grid(17)[[0, -1]], torch.tensor([0, 2 * math.pi], dtype=DTYPE)))

    print('Normalizer (per_timepoint, legacy-exact)')
    nm = Normalizer.fit(s.normals(), 'per_timepoint')
    Xn = nm.transform(s.normals())
    check('frame 0 untouched', torch.equal(Xn[:, 0], s.normals()[:, 0]))
    check('train normals span exactly [-π, π] per frame',
          torch.allclose(Xn[:, 1:].min(0).values, torch.full((T - 1,), -math.pi, dtype=DTYPE))
          and torch.allclose(Xn[:, 1:].max(0).values, torch.full((T - 1,), math.pi, dtype=DTYPE)))
    lo, hi = s.normals()[:, 1:].min(0).values, s.normals()[:, 1:].max(0).values
    legacy = 2 * math.pi * (X[:, 1:] - lo) / (hi - lo) - math.pi
    check('matches legacy formula 2π(x−min)/(max−min)−π', torch.allclose(nm.transform(X)[:, 1:], legacy))
    leg = Normalizer.from_legacy({'norm': 'per_timepoint',
                                  'norm_params': {'per_tp_min': lo.tolist(), 'per_tp_max': hi.tolist()}})
    check('from_legacy == fit', torch.allclose(leg.transform(X), nm.transform(X)))
    rt = Normalizer.from_dict(nm.to_dict())
    check('to_dict/from_dict round trip', torch.equal(rt.transform(X), nm.transform(X)))
    X_big = X * 3
    frac = nm.out_of_range(nm.transform(X_big))
    check('out_of_range detects values past ±π', frac > 0, f'{frac:.3f} of values for 3× scaled data')
    nq = Normalizer.fit(s.normals(), 'per_timepoint_q', clip=True)
    check("'per_timepoint_q' + clip stays in [-π, π]",
          nq.out_of_range(nq.transform(X_big)) == 0)

    print('roc_auc')
    check('perfect separation = 1', roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0)
    check('reversed = 0', roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0)
    check('all ties = 0.5', roc_auc([0, 1, 0, 1], [1, 1, 1, 1]) == 0.5)
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 5000)
    ss = rng.normal(size=5000) + yy
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in ss[yy == 1][:300] for b in ss[yy == 0][:300]])
    check('matches pairwise definition', abs(roc_auc(np.r_[np.ones(300), np.zeros(300)],
                                                      np.r_[ss[yy == 1][:300], ss[yy == 0][:300]]) - brute) < 1e-12,
          f'{brute:.4f}')

    print('Split directories')
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        rng = np.random.default_rng(0)
        sets = {'Xtr': rng.normal(size=(20, 9, 2)), 'Xte_norm': rng.normal(size=(6, 9, 2)),
                'Xte_b': rng.normal(size=(4, 9, 2)), 'Xte_a': rng.normal(size=(3, 9, 2))}
        write_split(tmp, sets, units={'Xte_norm': ['p1'] * 3 + ['p2'] * 3},
                    subtype={'Xte_a': ['x', 'y', 'x']}, manifest={'name': 'fake'})
        sp = load_split(tmp)
        check('round trip (N, T, d)', np.allclose(sp.X['Xtr'].numpy(), sets['Xtr']))
        check('n_channels / T', sp.n_channels == 2 and sp.T == 9)
        check('test_sets sorted, Xte_norm excluded', sp.test_sets() == ['Xte_a', 'Xte_b'])
        s1 = load_split(tmp, channels=[1])
        check('single channel → (N, T)', s1.X['Xtr'].shape == (20, 9)
              and np.allclose(s1.X['Xtr'].numpy(), sets['Xtr'][..., 1]))
        keys, yy, uu = sp.pair('Xte_norm', sp.test_sets())
        check('pair labels', yy.tolist() == [0] * 6 + [1] * 7)
        check('units shared across sets (one cluster per unit)', uu[0] == 'p1' and len(uu) == 13)
        check('subtype loaded', sp.subtype['Xte_a'].tolist() == ['x', 'y', 'x'])
        check('default units unique per set', sp.units['Xtr'][3] == 'Xtr#3')
    F = fourier_moments(torch.zeros(5, 7, 2))
    check('fourier_moments shape / values', F.shape == (5, 12) and np.allclose(F[:, :2], 1))

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()

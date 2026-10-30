"""
gw_ladder.py — rebuild the published QVR4GW pipeline on public LIGO data, then remove its
protocol choices one at a time (exploratory forensics; not a registered analysis).

Their pipeline (QVR4GW rewinder.py / utils.py, paper Secs. 6-8):
  - 2 qubits; RX(x) on qubit 0; W = 3 layers RX·RY·RZ per qubit + CNOT(0,1); D = Z0, Z1, Z0Z1;
    observable ½(Z0 + Z1); N_E = 10 fresh draws per timepoint; t = linspace(0.1, 2π, 4096).
  - Score = mean_t (η − mean_draws ⟨H⟩)²; loss = minibatch mean + (1/π)·mean arctan(2π·15·|σ|).
  - Training: Powell, bounds [0, 2π] (η in [−1, 1]), maxfev 2000; every evaluation draws a fresh
    minibatch of 10 series × 10 random timepoints. lightning.qubit with 256 shots.
  - utils.rescale: per-timepoint min-max to [−π+0.01, π−0.01], applied SEPARATELY to the noise
    file (5000 windows) and to the events (78).
  - Noise: 4000 train, 78 validation, 78 test. Events: validation = test = all events.
  - Threshold: grid search of balanced accuracy on validation (noise + all events).

Rungs (each run over --folds folds; same circuit, trainer and data throughout):
  A   paper protocol                               separate scaling, events val = test
  A0  paper protocol, NO SIGNAL                    'events' are held-out noise windows
  B   joint scaling                                min-max fit on training noise, applied to all
  B0  joint scaling, NO SIGNAL
  C   joint scaling + disjoint val/test events     events split in half by fold
  (+ A_their: rung A with the authors' own 78 filtered events, if --qvr4gw is given)
For every rung: test AUC, balanced accuracy at the validation-tuned threshold (their
threshold_scan), score means, and a one-number classical comparator (per-series std of the
scaled series) on the same sets.

Data: the per-event caches written by `prepare.py gw-gwosc` (raw_dir/filtered/<event>.npz:
1 event window + 100 noise windows, 4096 samples at 4096 Hz, filtered).

Usage (server):
  python forensics/gw_ladder.py --raw-dir data_raw/gwosc/filtered --qvr4gw <QVR4GW clone> \
      --out results/forensics/gw_ladder
  python forensics/gw_ladder.py --self-test
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw_sim as G  # noqa: E402

H_DIAG = np.array([1.0, 0.0, 0.0, -1.0])          # ½(Z0 + Z1) on |00>,|01>,|10>,|11>
EPS = 0.01
LO_X, HI_X = -np.pi + EPS, np.pi - EPS


# ---------------------------------------------------------------------------
# Circuit with arbitrary (x, t) pairs and optional shot noise
# ---------------------------------------------------------------------------

def _probs(x, t, alphas, g):
    """x, t: (...,) ; g: (..., 3) draws. Returns outcome probabilities (..., 4)."""
    W = G.W_matrix(alphas)
    c, s = np.cos(x / 2), np.sin(x / 2)
    psi = np.zeros(x.shape + (4,), complex)
    psi[..., 0] = c
    psi[..., 2] = -1j * s
    phi = psi @ W.T
    chi = phi * G.D_eigs(t[..., None] * g)
    fin = chi @ W.conj()
    p = np.abs(fin) ** 2
    return p / p.sum(-1, keepdims=True)


def _expval(p, shots, rng):
    if not shots:
        return p @ H_DIAG
    counts = rng.multinomial(shots, p)
    return (counts[..., 0] - counts[..., 3]) / shots


def point_costs(x, t, params, n_draws, shots, rng):
    """(η − mean_draws ⟨H⟩)² for each (x, t) pair; fresh draws per point."""
    z = np.zeros(x.shape)
    for _ in range(n_draws):
        g = rng.normal(params['mus'], np.abs(params['sigmas']), size=x.shape + (3,))
        z += _expval(_probs(x, t, params['alphas'], g), shots, rng)
    return (params['eta_0'] - z / n_draws) ** 2


def scores(X, params, n_draws=10, shots=256, seed=0, chunk=16):
    """Per-series anomaly score: mean over all timepoints (their get_anomaly_scores)."""
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    t = np.broadcast_to(np.linspace(0.1, 2 * np.pi, T), (min(chunk, len(X)), T))
    out = []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        out.append(point_costs(x, t[:len(x)], params, n_draws, shots, rng).mean(1))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Their training: Powell on a stochastic minibatch objective
# ---------------------------------------------------------------------------

def unpack(v):
    return {'alphas': v[:18].reshape(3, 2, 3), 'mus': v[18:21], 'sigmas': v[21:24], 'eta_0': v[24]}


def train_paper(Xtr, seed, maxfev=2000, n_series=10, n_t=10, n_draws=10, shots=256, tau=15.0):
    rng = np.random.default_rng(seed)
    N, T = Xtr.shape
    t_all = np.linspace(0.1, 2 * np.pi, T)
    state = {'s_perm': rng.permutation(N), 's_pos': 0, 't_perm': rng.permutation(T), 't_pos': 0}

    def next_series():
        if state['s_pos'] + n_series > N:
            state['s_perm'], state['s_pos'] = rng.permutation(N), 0
        idx = state['s_perm'][state['s_pos']:state['s_pos'] + n_series]
        state['s_pos'] += n_series
        return idx

    def next_t():
        if state['t_pos'] + n_t > T:
            state['t_perm'], state['t_pos'] = rng.permutation(T), 0
        idx = state['t_perm'][state['t_pos']:state['t_pos'] + n_t]
        state['t_pos'] += n_t
        return idx

    hist = []

    def loss(v):
        p = unpack(v)
        rows = next_series()
        tix = np.stack([next_t() for _ in rows])                  # each series its own timepoints
        x = Xtr[rows[:, None], tix]
        c = point_costs(x, t_all[tix], p, n_draws, shots, rng).mean()
        pen = np.arctan(2 * np.pi * tau * np.abs(p['sigmas'])).mean() / np.pi
        hist.append(float(c + pen))
        return c + pen

    v0 = np.r_[rng.uniform(0, 2 * np.pi, 24), rng.uniform(-1, 1)]
    bounds = [(0, 2 * np.pi)] * 24 + [(-1, 1)]
    res = minimize(loss, v0, method='Powell', bounds=bounds, options={'maxfev': maxfev, 'maxiter': np.inf})
    return unpack(res.x), {'nfev': int(res.nfev), 'final_loss': hist[-1], 'init': v0.tolist()}


# ---------------------------------------------------------------------------
# Scaling and evaluation
# ---------------------------------------------------------------------------

def rescale_separate(X):
    return G.rescale(X, eps=EPS)


def minmax_fit(X):
    lo, hi = X.min(0, keepdims=True), X.max(0, keepdims=True)
    return lambda Y: np.clip((Y - lo) / (hi - lo) * (HI_X - LO_X) + LO_X, LO_X, HI_X)


def auc(neg, pos):
    s = np.r_[neg, pos]
    r = np.empty(len(s))
    order = np.argsort(s, kind='mergesort')
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                   # average ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def balanced_acc(neg, pos, thr):
    return 0.5 * (np.mean(neg < thr) + np.mean(pos >= thr))


def threshold_scan(neg, pos, steps=1000):
    """Their threshold_scan: zetas = linspace(0, max anomalous score), midpoint of the best."""
    zetas = np.linspace(0, pos.max(), steps)
    accs = np.array([balanced_acc(neg, pos, z) for z in zetas])
    best = np.flatnonzero(accs == accs.max())
    return float(zetas[best[len(best) // 2]])


def evaluate(params, sets, seed, shots, n_draws):
    """sets: dict with val_noise, val_events, test_noise, test_events (already scaled)."""
    sc = {k: scores(v, params, n_draws=n_draws, shots=shots, seed=seed + i)
          for i, (k, v) in enumerate(sets.items())}
    thr = threshold_scan(sc['val_noise'], sc['val_events'])
    std = {k: v.std(1) for k, v in sets.items()}
    return {
        'auc': auc(sc['test_noise'], sc['test_events']),
        'bal_acc': balanced_acc(sc['test_noise'], sc['test_events'], thr),
        'threshold': thr,
        'score_noise': [float(sc['test_noise'].mean()), float(sc['test_noise'].std())],
        'score_events': [float(sc['test_events'].mean()), float(sc['test_events'].std())],
        'std_auc': auc(std['test_noise'], std['test_events']),
        'scaled_std_noise': float(sets['test_noise'].std()),
        'scaled_std_events': float(sets['test_events'].std()),
    }


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def load_cache(raw_dir):
    files = sorted(glob.glob(str(Path(raw_dir) / '*.npz')))
    if not files:
        raise FileNotFoundError(f'no event caches in {raw_dir}')
    ev, no, no_u = [], [], []
    for k, f in enumerate(files):
        d = np.load(f)
        ev.append(d['event'])
        no.append(d['noise'])
        no_u += [k] * len(d['noise'])
    return np.concatenate(ev), np.concatenate(no), np.array(no_u), [Path(f).stem for f in files]


def run_ladder(events, noise, noise_units, their_events=None, folds=5, n_pool=5000, n_val=78, n_test=78,
               maxfev=2000, shots=256, n_draws=10, seed=0, log=print):
    """events (E, T), noise (M, T) raw filtered strain. Returns {rung: [per-fold dicts]}."""
    rng = np.random.default_rng(seed)
    E = len(events)
    n_tr = n_pool - n_val - n_test
    # 'no-signal events': noise windows from event neighbourhoods kept out of every pool
    units = np.unique(noise_units)
    null_units = rng.choice(units, size=max(1, len(units) // 10), replace=False)
    null_mask = np.isin(noise_units, null_units)
    null_pool, avail = noise[null_mask], np.flatnonzero(~null_mask)
    if len(avail) < n_pool:
        raise ValueError(f'only {len(avail)} noise windows available for a pool of {n_pool}')
    out = {r: [] for r in ('A', 'A0', 'A_their', 'B', 'B0', 'C')}
    for k in range(folds):
        fr = np.random.default_rng([seed, k])
        pool = noise[fr.choice(avail, n_pool, replace=False)]      # stands in for their 5000-window file
        fake = null_pool[fr.choice(len(null_pool), E, replace=False)]
        tr, va, te = pool[:n_tr], pool[n_tr:n_tr + n_val], pool[n_tr + n_val:]
        t0 = time.time()

        # --- paper protocol: whole noise file scaled together, events scaled alone
        P = rescale_separate(pool)
        params_sep, info_sep = train_paper(P[:n_tr], seed=1000 * seed + k, maxfev=maxfev, shots=shots,
                                           n_draws=n_draws)
        Ps = {'val_noise': P[n_tr:n_tr + n_val], 'test_noise': P[n_tr + n_val:]}
        ev_s = rescale_separate(events)
        fake_s = rescale_separate(fake)
        out['A'].append({**evaluate(params_sep, {**Ps, 'val_events': ev_s, 'test_events': ev_s}, 10 * k, shots,
                                    n_draws), **info_sep})
        out['A0'].append(evaluate(params_sep, {**Ps, 'val_events': fake_s, 'test_events': fake_s}, 10 * k + 1,
                                  shots, n_draws))
        if their_events is not None:
            th = rescale_separate(their_events)
            out['A_their'].append(evaluate(params_sep, {**Ps, 'val_events': th, 'test_events': th}, 10 * k + 2,
                                           shots, n_draws))

        # --- joint scaling: one min-max fit on training noise, applied to everything
        f = minmax_fit(tr)
        params_j, info_j = train_paper(f(tr), seed=1000 * seed + k + 500, maxfev=maxfev, shots=shots,
                                       n_draws=n_draws)
        Js = {'val_noise': f(va), 'test_noise': f(te)}
        ev_j, fake_j = f(events), f(fake)
        out['B'].append({**evaluate(params_j, {**Js, 'val_events': ev_j, 'test_events': ev_j}, 10 * k + 3, shots,
                                    n_draws), **info_j})
        out['B0'].append(evaluate(params_j, {**Js, 'val_events': fake_j, 'test_events': fake_j}, 10 * k + 4,
                                  shots, n_draws))
        half = fr.permutation(E)
        vi, ti = half[:E // 2], half[E // 2:]
        out['C'].append(evaluate(params_j, {**Js, 'val_events': ev_j[vi], 'test_events': ev_j[ti]}, 10 * k + 5,
                                 shots, n_draws))
        log(f'  fold {k + 1}/{folds} ({time.time() - t0:.0f}s): ' + ' | '.join(
            f"{r} AUC {out[r][-1]['auc']:.3f} BA {out[r][-1]['bal_acc']:.3f}" for r in out if out[r]))
    return {r: v for r, v in out.items() if v}


RUNG_TEXT = {
    'A': 'paper protocol (separate scaling, events val = test)',
    'A_their': 'paper protocol, the authors\' own 78 events',
    'A0': 'paper protocol, NO SIGNAL (held-out noise as "events")',
    'B': 'joint scaling (fit on training noise), events val = test',
    'B0': 'joint scaling, NO SIGNAL',
    'C': 'joint scaling, disjoint val / test events',
}


def summarize(res):
    lines = ['| Rung | Protocol | QVR AUC (per fold) | mean | QVR BA mean | std-only AUC mean | '
             'scaled std noise / events |', '|---|---|---|---|---|---|---|']
    for r in ('A', 'A_their', 'A0', 'B', 'B0', 'C'):
        if r not in res:
            continue
        v = res[r]
        a = [x['auc'] for x in v]
        lines.append(f"| {r} | {RUNG_TEXT[r]} | {', '.join(f'{x:.3f}' for x in a)} | {np.mean(a):.3f} | "
                     f"{np.mean([x['bal_acc'] for x in v]):.3f} | {np.mean([x['std_auc'] for x in v]):.3f} | "
                     f"{np.mean([x['scaled_std_noise'] for x in v]):.2f} / "
                     f"{np.mean([x['scaled_std_events'] for x in v]):.2f} |")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir', default='data_raw/gwosc/filtered')
    ap.add_argument('--qvr4gw', default=None, help='QVR4GW clone; adds rung A_their (their data/events.pkl)')
    ap.add_argument('--out', default='results/forensics/gw_ladder')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--maxfev', type=int, default=2000)
    ap.add_argument('--shots', type=int, default=256, help='0 = exact expectation values')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    events, noise, units, names = load_cache(a.raw_dir)
    their = None
    if a.qvr4gw:
        import pandas as pd
        their = np.asarray(pd.read_pickle(Path(a.qvr4gw) / 'data' / 'events.pkl'), dtype=float)
    print(f'gw_ladder: {len(events)} events, {len(noise)} noise windows from {len(names)} neighbourhoods; '
          f'T = {events.shape[1]}; folds {a.folds}; maxfev {a.maxfev}; shots {a.shots}', flush=True)
    res = run_ladder(events, noise, units, their, folds=a.folds, maxfev=a.maxfev, shots=a.shots, seed=a.seed,
                     log=lambda s: print(s, flush=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ladder.json').write_text(json.dumps(res, indent=1))
    md = summarize(res)
    (out / 'ladder.md').write_text(md + '\n')
    print(md)


def _self_test():
    print('gw_ladder self-test')
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, (3, 2, 3))
    p = {'alphas': a, 'mus': rng.normal(size=3), 'sigmas': rng.uniform(0.5, 2, 3), 'eta_0': 0.1}
    X = rng.uniform(-3, 3, (5, 64))
    # exact path equals gw_sim.scores (same draws: one (T, 3) draw per draw index there; compare in expectation)
    s_mine = scores(X, {**p, 'sigmas': np.zeros(3)}, shots=0, n_draws=3)
    s_ref = G.scores(X, {**p, 'sigmas': np.zeros(3)}, n_draws=3)
    check('sigma = 0, exact: matches gw_sim.scores', np.allclose(s_mine, s_ref, atol=1e-12),
          f'max |Δ| {np.abs(s_mine - s_ref).max():.1e}')
    s_shot = scores(X, {**p, 'sigmas': np.zeros(3)}, shots=100000, n_draws=3)
    check('shot noise is unbiased at 1e5 shots', np.allclose(s_shot, s_ref, atol=5e-3),
          f'max |Δ| {np.abs(s_shot - s_ref).max():.1e}')
    check('auc: perfect / inverted / ties', auc([0, 1], [2, 3]) == 1 and auc([2, 3], [0, 1]) == 0
          and auc([1, 1], [1, 1]) == 0.5)
    f = minmax_fit(rng.normal(size=(50, 8)))
    Y = f(rng.normal(size=(20, 8)) * 10)
    check('joint scaling clips to [-pi+eps, pi-eps]', Y.min() >= LO_X and Y.max() <= HI_X)
    pr, info = train_paper(rescale_separate(rng.normal(size=(40, 64))), seed=0, maxfev=60, shots=0)
    check('Powell trainer runs and respects maxfev', info['nfev'] <= 80 and set(pr) == {'alphas', 'mus', 'sigmas',
                                                                                        'eta_0'},
          f"nfev {info['nfev']}")
    # tiny ladder on Gaussian 'noise' and 'events' with no signal anywhere
    T = 64
    noise = rng.normal(size=(600, T))
    units = np.repeat(np.arange(60), 10)
    events = rng.normal(size=(20, T))
    res = run_ladder(events, noise, units, None, folds=1, n_pool=300, n_val=20, n_test=20, maxfev=40, shots=0,
                     log=lambda s: None)
    check('ladder runs: rungs A, A0, B, B0, C', set(res) == {'A', 'A0', 'B', 'B0', 'C'})
    sA = res['A'][0]
    check('separate scaling inflates the small set\'s spread (signal-free)',
          sA['scaled_std_events'] > 1.1 * sA['scaled_std_noise'],
          f"{sA['scaled_std_noise']:.2f} vs {sA['scaled_std_events']:.2f}")
    check('std-only detector separates signal-free sets under separate scaling', res['A'][0]['std_auc'] > 0.9,
          f"{res['A'][0]['std_auc']:.3f}")
    check('...and not under joint scaling', abs(res['B'][0]['std_auc'] - 0.5) < 0.25,
          f"{res['B'][0]['std_auc']:.3f}")
    print('RESULT:', 'ALL PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

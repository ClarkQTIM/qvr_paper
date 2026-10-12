"""
run.py — everything for ONE dataset, as resumable stages (PAPER_PLAN §6).

    python run.py --dataset mitbih --workers 8                 # all stages
    python run.py --dataset mitbih --stages score report       # some stages
    python run.py --data-dir <split dir> --out <dir> ...       # without configs/datasets.json
    python run.py --self-test                                  # whole pipeline on fake data

Stages (each skips work whose output already exists):
  select     QVR grid (configs/grid.json) × selection seeds, trained on Xtr, scored on
             validation (Xval_norm vs Xval); winner = best mean validation AUC.
             → select/<tag>_s<seed>/, select/selection.json
  train      final models over final seeds: best (winner), ref (reference config),
             sigma0 (winner with σ frozen at 0)          → models/<name>_s<seed>/
  score      every non-training set × D modes (mc, mean, identity, averaged), the
             time-permuted copy, p̄_b, and the four Test A readouts → scores/<model>.npz
  baselines  classical suite, tuned on validation, fit on Xtr → baselines/{scores.npz,selection.json}
  teardown   per model: Fourier exactness + damping + Hamiltonian/σ → teardown/<model>.json;
             subtype probes (p̄_b vs moments vs raw) → teardown/probes.json
  report     summary.json + summary.md (one block per paper claim)

Data: a split directory (README "Drop-in dataset layout"), already normalized.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qvr import circuit as C                                       # noqa: E402
from qvr.data import fourier_moments, load_split, roc_auc, t_grid  # noqa: E402

STAGES = ('select', 'train', 'score', 'baselines', 'teardown', 'report')
D_MODES = ('mc', 'mean', 'identity', 'averaged')


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def tag(c: dict) -> str:
    return (f"L{c['n_layers']}_NE{c['N_E']}_{c['cost']}_{c['embedding']}_{c['ansatz']}_lr{c['lr']}")


def grid_points(grid: dict, n_channels: int) -> list[dict]:
    """All grid configs as flat dicts (QVRConfig fields + lr)."""
    sec = grid['univariate'] if n_channels == 1 else grid['multichannel']
    keys = ('n_layers', 'N_E', 'cost', 'embedding', 'ansatz', 'lr')
    out = []
    for vals in itertools.product(*(sec[k] for k in keys)):
        c = dict(zip(keys, vals))
        c['n_qubits'] = sec.get('n_qubits', 3) if n_channels == 1 else n_channels
        out.append(c)
    return out


def reference(grid: dict, n_channels: int) -> dict:
    c = dict(grid['reference']['univariate' if n_channels == 1 else 'multichannel'])
    if n_channels > 1:
        c['n_qubits'] = n_channels
    return c


def make_cfg(c: dict, grid: dict) -> C.QVRConfig:
    f = grid['fixed']
    return C.QVRConfig(n_qubits=c['n_qubits'], n_layers=c['n_layers'], k=f['k'], N_E=c['N_E'],
                       tau=f['tau'], sigma_lambda=f['sigma_lambda'], embedding=c['embedding'],
                       ansatz=c['ansatz'], cost=c['cost'], draw_mode=f['draw_mode'])


# ---------------------------------------------------------------------------
# Workers (module level so they pickle)
# ---------------------------------------------------------------------------

def _threads():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))


def _train_job(job):
    """job: (data_dir, channels, run_dir, cfg dict, seed, grid, freeze_sigma, strict, score_val)"""
    _threads()
    data_dir, channels, run_dir, c, seed, grid, freeze, strict, score_val = job
    from qvr.train import load_run, train
    from qvr.infer import infer
    run_dir = Path(run_dir)
    sp = load_split(data_dir, channels)
    t = t_grid(sp.T)
    if not (run_dir / 'params.pt').exists():
        train(sp.X['Xtr'], t, make_cfg(c, grid), run_seed=seed, out_dir=run_dir, lr=c['lr'],
              max_samples=grid['train']['max_samples'], batch_size=grid['train']['batch_size'],
              meta={'data_dir': str(data_dir), 'channels': channels, 'grid_point': c},
              strict_git=strict, freeze_sigma=freeze, verbose=False)
        msg = 'trained'
    else:
        msg = 'exists'
    if score_val and sp.has_val() and not (run_dir / 'val.json').exists():
        params, cfg, _ = load_run(run_dir)
        bs = infer_batch(sp.T, cfg)
        sn = infer(params, cfg, sp.X['Xval_norm'], t, mode='mc', seed=0, batch_size=bs)['score']
        sa = infer(params, cfg, sp.X['Xval'], t, mode='mc', seed=0, batch_size=bs)['score']
        auc = roc_auc(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])
        (run_dir / 'val.json').write_text(json.dumps({'val_auc': auc}))
    return f'{run_dir.name}: {msg}'


def infer_batch(T: int, cfg: C.QVRConfig, budget: float = 4e7) -> int:
    """Series per inference batch so that (batch × T × N_E × dim) complex values stay ≈ budget
    (≈ 0.6 GB); 4096 for short series, small for GW-length series."""
    return int(max(8, min(4096, budget // (T * cfg.N_E * cfg.dim))))


def _score_job(job):
    _threads()
    data_dir, channels, model_dir, out_path, n_fit = job
    from qvr.infer import infer, permute_time
    from qvr.teardown import readouts
    from qvr.train import load_run
    if Path(out_path).exists():
        return f'{Path(out_path).stem}: exists'
    sp = load_split(data_dir, channels)
    params, cfg, _ = load_run(model_dir)
    t = t_grid(sp.T)
    bs = infer_batch(sp.T, cfg)
    res = {}
    sets = {k: v for k, v in sp.X.items() if k != 'Xtr'}
    for k, Xk in sets.items():
        for m in D_MODES:
            o = infer(params, cfg, Xk, t, mode=m, seed=0, batch_size=bs)
            res[f'{m}__{k}'] = o['score']
            if m == 'mc':
                res[f'pbar__{k}'] = o['pbar']
                res[f'zrepr__{k}'] = o['zrepr']
        res[f'perm__{k}'] = infer(params, cfg, permute_time(Xk, seed=7, keep_first=False), t,
                                  mode='mc', seed=0, batch_size=bs)['score']
    ro = readouts(params, cfg, sp.X['Xtr'], sets, n_fit=n_fit, batch_size=bs)
    for r, d in ro.items():
        for k, v in d.items():
            res[f'ro_{r}__{k}'] = v
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **res)
    return f'{Path(out_path).stem}: scored'


def _teardown_job(job):
    _threads()
    model_dir, out_path, T, td = job
    from qvr.teardown import fourier_report, hamiltonian
    from qvr.train import load_run
    if Path(out_path).exists():
        return f'{Path(out_path).stem}: exists'
    params, cfg, ck = load_run(model_dir)
    r = {'model': Path(model_dir).name, 'cfg': cfg.to_dict(),
         'fourier': fourier_report(params, cfg, T, n_grid=td['n_grid'], n_rand=td['n_rand'], n_mc=td['n_mc']),
         'hamiltonian': hamiltonian(params, cfg, init=ck.get('init_params'))}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(r, indent=1))
    return f'{Path(out_path).stem}: done (all predictions hold: {r["fourier"]["all_hold"]})'


def _pool(fn, jobs, workers):
    t0 = time.time()
    if not jobs:
        return
    if workers <= 1:
        for j in jobs:
            print(f'    {fn(j)}  ({time.time() - t0:.0f}s)', flush=True)
    else:
        # 'spawn': forking after torch has started its thread pool can deadlock
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
            for r in ex.map(fn, jobs):
                print(f'    {r}  ({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, data_dir, out, channels, grid, workers=1, strict_git=False):
        self.data_dir, self.out, self.channels = str(data_dir), Path(out), channels
        self.grid, self.workers, self.strict = grid, workers, strict_git
        self.sp = load_split(data_dir, channels)
        self.d = self.sp.n_channels
        self.out.mkdir(parents=True, exist_ok=True)

    # -- select ------------------------------------------------------------
    def select(self):
        pts = grid_points(self.grid, self.d)
        seeds = self.grid['selection_seeds']
        print(f'  select: {len(pts)} configs × {len(seeds)} seeds'
              + ('' if self.sp.has_val() else '  (NO validation sets → reference config is used)'))
        if not self.sp.has_val():
            sel = {'winner': reference(self.grid, self.d), 'selected_on_val': False, 'table': []}
        else:
            jobs = [(self.data_dir, self.channels, self.out / 'select' / f'{tag(c)}_s{s}', c, s,
                     self.grid, False, self.strict, True) for c in pts for s in seeds]
            _pool(_train_job, jobs, self.workers)
            table = []
            for c in pts:
                v = [json.loads((self.out / 'select' / f'{tag(c)}_s{s}' / 'val.json').read_text())['val_auc']
                     for s in seeds]
                table.append({'tag': tag(c), 'cfg': c, 'val_auc_mean': float(np.mean(v)), 'val_auc': v})
            # best mean val AUC; ties → fewer layers, then fewer draws (simpler model)
            table.sort(key=lambda r: (-round(r['val_auc_mean'], 6), r['cfg']['n_layers'], r['cfg']['N_E']))
            sel = {'winner': table[0]['cfg'], 'selected_on_val': True, 'table': table}
        sel['reference'] = reference(self.grid, self.d)
        (self.out / 'select').mkdir(parents=True, exist_ok=True)
        (self.out / 'select' / 'selection.json').write_text(json.dumps(sel, indent=1))
        print(f"  winner: {tag(sel['winner'])}")

    def _selection(self):
        p = self.out / 'select' / 'selection.json'
        if not p.exists():
            raise FileNotFoundError(f'{p}: run the select stage first')
        return json.loads(p.read_text())

    def models(self) -> dict:
        """{model name: (cfg dict, seed, freeze_sigma)} for the final models."""
        sel = self._selection()
        out = {}
        for s in self.grid['final_seeds']:
            out[f'best_s{s}'] = (sel['winner'], s, False)
            out[f'ref_s{s}'] = (sel['reference'], s, False)
            out[f'sigma0_s{s}'] = (sel['winner'], s, True)
        return out

    # -- train / score / teardown -----------------------------------------
    def train(self):
        jobs = [(self.data_dir, self.channels, self.out / 'models' / name, c, s, self.grid, frz, self.strict, False)
                for name, (c, s, frz) in self.models().items()]
        print(f'  train: {len(jobs)} models')
        _pool(_train_job, jobs, self.workers)

    def score(self):
        jobs = [(self.data_dir, self.channels, self.out / 'models' / n, self.out / 'scores' / f'{n}.npz',
                 self.grid['teardown']['readout_n_fit']) for n in self.models()]
        print(f'  score: {len(jobs)} models')
        _pool(_score_job, jobs, self.workers)

    def teardown(self):
        jobs = [(self.out / 'models' / n, self.out / 'teardown' / f'{n}.json', self.sp.T, self.grid['teardown'])
                for n in self.models()]
        print(f'  teardown: {len(jobs)} models')
        _pool(_teardown_job, jobs, self.workers)
        self.probes()

    def probes(self):
        """C5: subtype from p̄_b (each best seed) vs six moments vs the raw series."""
        from qvr.teardown import probe
        sp, n_perm = self.sp, self.grid['teardown']['n_perm']
        tests = [k for k in sp.test_sets() if k in sp.subtype]
        if 'Xprobe' not in sp.X or 'Xprobe' not in sp.subtype or not tests:
            (self.out / 'teardown' / 'probes.json').write_text(json.dumps({'skipped': 'no Xprobe / subtypes'}))
            print('  probes: skipped (no Xprobe or subtype labels)')
            return
        y_fit, u_fit = sp.subtype['Xprobe'], sp.units['Xprobe']
        y_ev = np.concatenate([sp.subtype[k] for k in tests])
        raw = lambda k: sp.X[k].reshape(len(sp.X[k]), -1).numpy()  # noqa: E731
        feats = {'moments': (fourier_moments(sp.X['Xprobe']),
                             np.concatenate([fourier_moments(sp.X[k]) for k in tests])),
                 'raw': (raw('Xprobe'), np.concatenate([raw(k) for k in tests]))}
        for n in self.models():
            if n.startswith('best_'):
                S = np.load(self.out / 'scores' / f'{n}.npz')
                feats[f'pbar_{n}'] = (S['pbar__Xprobe'], np.concatenate([S[f'pbar__{k}'] for k in tests]))
        res = {'eval_sets': tests}
        first = f"pbar_best_s{self.grid['final_seeds'][0]}"
        for name, (Ff, Fe) in feats.items():
            # permutation null for moments, raw and one QVR seed (the null is the slow part)
            res[name] = probe(Ff, y_fit, u_fit, Fe, y_ev,
                              n_perm=n_perm if name in ('moments', 'raw', first) else 0)
            r = res[name]
            print(f"  probe {name:16s} " + (f"macro AUC {r['logreg']['macro_auc']:.3f}" if r else 'n/a'))
        (self.out / 'teardown' / 'probes.json').write_text(json.dumps(res, indent=1))

    # -- baselines ---------------------------------------------------------
    def baselines(self):
        from qvr.baselines import select_and_score
        path = self.out / 'baselines' / 'scores.npz'
        if path.exists():
            print('  baselines: exist')
            return
        sp = self.sp
        sets = {k: v for k, v in sp.X.items() if k != 'Xtr'}
        val = (sp.X['Xval_norm'], sp.X['Xval']) if sp.has_val() else None
        t0 = time.time()
        sc, rep = select_and_score(sp.X['Xtr'], sets, val=val)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f'{m}__{k}': v for m, d in sc.items() for k, v in d.items()})
        (path.parent / 'selection.json').write_text(json.dumps(rep, indent=1, default=str))
        print(f'  baselines: {len(sc)} methods ({time.time() - t0:.0f}s)')

    # -- report --------------------------------------------------------------
    def report(self):
        from report import build_report
        rep = build_report(self)
        (self.out / 'summary.json').write_text(json.dumps(rep, indent=1, default=float))
        from report import to_markdown
        (self.out / 'summary.md').write_text(to_markdown(rep))
        print(f"  report → {self.out / 'summary.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', help='key in configs/datasets.json')
    ap.add_argument('--data-dir')
    ap.add_argument('--out')
    ap.add_argument('--channels', type=int, nargs='*')
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'datasets.json'))
    ap.add_argument('--grid', default=str(ROOT / 'configs' / 'grid.json'))
    ap.add_argument('--stages', nargs='+', default=list(STAGES), choices=STAGES)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--strict-git', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    channels = a.channels
    if a.dataset:
        ds = json.loads(Path(a.config).read_text())[a.dataset]
        data_dir = a.data_dir or ds['data_dir']
        channels = channels if channels is not None else ds.get('channels')
        out = a.out or f'results/{a.dataset}'
    else:
        if not (a.data_dir and a.out):
            ap.error('need --dataset, or --data-dir and --out')
        data_dir, out = a.data_dir, a.out
    grid = json.loads(Path(a.grid).read_text())
    r = Run(data_dir, out, channels, grid, a.workers, a.strict_git)
    print(f'run.py  {data_dir} → {out}   channels={r.d}  T={r.sp.T}  grid={Path(a.grid).name}')
    for st in STAGES:
        if st in a.stages:
            t0 = time.time()
            print(f'[{st}]', flush=True)
            getattr(r, st)()
            print(f'[{st}] done in {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------------------
# Self-test: the whole pipeline on a tiny fake dataset (quick grid)
# ---------------------------------------------------------------------------

def _fake_split(path, n_channels=1, seed=0, T=12):
    from qvr.data import write_split
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 2 * np.pi, T)) * 1.2

    def norm(n, units):
        X = base + 0.35 * rng.normal(size=(n, T))
        X = np.repeat(X[..., None], n_channels, -1) + 0.1 * rng.normal(size=(n, T, n_channels))
        return np.clip(X, -np.pi, np.pi), np.array([f'u{units[i % len(units)]}' for i in range(n)])

    def anom(n, units, kind):
        X, u = norm(n, units)
        X = X + (0.9 if kind == 'shift' else 0.0) + (rng.normal(size=X.shape) * 0.6 if kind == 'noise' else 0)
        return np.clip(X, -np.pi, np.pi), u
    sets, units, sub = {}, {}, {}
    sets['Xtr'], units['Xtr'] = norm(160, range(0, 16))
    sets['Xval_norm'], units['Xval_norm'] = norm(40, range(16, 20))
    a1, u1 = anom(20, range(16, 20), 'shift')
    a2, u2 = anom(20, range(16, 20), 'noise')
    sets['Xval'], units['Xval'] = np.r_[a1, a2], np.r_[u1, u2]
    sets['Xte_norm'], units['Xte_norm'] = norm(60, range(20, 32))
    sets['Xte_shift'], units['Xte_shift'] = anom(30, range(20, 32), 'shift')
    sets['Xte_noise'], units['Xte_noise'] = anom(30, range(20, 32), 'noise')
    sub['Xte_shift'], sub['Xte_noise'] = ['shift'] * 30, ['noise'] * 30
    p1, q1 = anom(30, range(0, 16), 'shift')
    p2, q2 = anom(30, range(0, 16), 'noise')
    sets['Xprobe'], units['Xprobe'] = np.r_[p1, p2], np.r_[q1, q2]
    sub['Xprobe'] = ['shift'] * 30 + ['noise'] * 30
    write_split(path, sets, units, sub, {'name': 'fake', 'note': 'run.py self-test'})


def _self_test():
    import tempfile
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    grid = json.loads((ROOT / 'configs' / 'grid_quick.json').read_text())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for d in (1, 2):
            data, out = Path(tmp) / f'data{d}', Path(tmp) / f'out{d}'
            _fake_split(data, n_channels=d, seed=d)
            print('=' * 70)
            print(f'run.py self-test: fake dataset, {d} channel(s), quick grid')
            print('=' * 70)
            main(['--data-dir', str(data), '--out', str(out), '--grid', str(ROOT / 'configs' / 'grid_quick.json'),
                  '--workers', '2'])
            rep = json.loads((out / 'summary.json').read_text())
            n_models = 3 * len(grid['final_seeds'])
            check('selection table covers the grid',
                  len(json.loads((out / 'select' / 'selection.json').read_text())['table'])
                  == len(grid_points(grid, d)))
            check(f'{n_models} final models trained', len(list((out / 'models').glob('*/params.pt'))) == n_models)
            check('sigma0 models have σ = 0', all(
                torch.count_nonzero(torch.load(p, weights_only=False)['params']['sigma']).item() == 0
                for p in (out / 'models').glob('sigma0_*/params.pt')))
            check('every model scored and torn down',
                  len(list((out / 'scores').glob('*.npz'))) == n_models
                  and len(list((out / 'teardown').glob('*_s*.json'))) == n_models)
            check('Fourier predictions hold on every trained model', rep['fourier']['all_hold_fraction'] == 1.0,
                  f"{rep['fourier']['all_hold_fraction']:.2f}")
            m = rep['main']['pooled']
            check('main table has QVR rows and every baseline', all(k in m for k in
                  ('qvr_best', 'qvr_ref', 'qvr_sigma0', 'qvr_best_identity', 'tz2', 'onenum', 'knn')))
            check('easy fake anomalies: best baseline AUC > 0.8',
                  max(v['auc'] for k, v in m.items() if not k.startswith('qvr')) > 0.8)
            check('null guard ≈ 0.5 for every method', rep['null_guard']['max_abs_dev'] < 0.25,
                  f"max |AUC − 0.5| = {rep['null_guard']['max_abs_dev']:.3f}")
            check('probes ran', 'moments' in rep['probes'])
            check('summary.md written', (out / 'summary.md').stat().st_size > 500)
            # resumability: a second run does no new training
            t0 = time.time()
            main(['--data-dir', str(data), '--out', str(out), '--grid', str(ROOT / 'configs' / 'grid_quick.json'),
                  '--stages', 'select', 'train', 'score', 'teardown'])
            check('re-run is a no-op (resumable)', time.time() - t0 < 30, f'{time.time() - t0:.1f}s')
    print('=' * 70)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""
report.py — turn one dataset's run directory into summary.json / summary.md.
Called by `run.py --stages report`; collate.py reads the JSON.

Blocks (one per paper claim):
  main        pooled + per-set AUC [unit CI] for every QVR row and every baseline;
              paired AUC difference vs QVR (best, rank-averaged over seeds)       C4
  seeds       per-seed AUC mean ± SD for best / ref / sigma0                       C4, C3
  d_modes     per-seed AUC under mc / mean / identity / averaged                  C3
  sigma0      σ frozen at 0 vs learned σ: AUC and training length                 C3
  order       AUC on time-permuted copies; ΔAUC; score correlation                C2
  fourier     Fourier-exactness verdicts across all trained models; damping       C2, C3
  hamiltonian |σ| final vs init, S_r vs init                                       C3
  probes      subtype probes: p̄_b vs six moments vs raw                           C5
  null_guard  normals vs normals (unit halves): every method should be ≈ 0.5      protocol
  selection   QVR winner + top of the grid; baseline hyperparameters
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from qvr.data import roc_auc
from qvr.stats import auc_ci, null_halves, paired_diff_ci, rank_avg

QVR_ROWS = {   # row name: (model prefix, score key)
    'qvr_best': ('best', 'mc'),
    'qvr_best_mean': ('best', 'mean'),
    'qvr_best_identity': ('best', 'identity'),
    'qvr_best_averaged': ('best', 'averaged'),
    'qvr_ref': ('ref', 'mc'),
    'qvr_sigma0': ('sigma0', 'mc'),
    'qvr_best_ro_tz2_z': ('best', 'ro_tz2_z'),
    'qvr_best_ro_mahal_zmean': ('best', 'ro_mahal_zmean'),
    'qvr_best_ro_mahal_pbar': ('best', 'ro_mahal_pbar'),
}


def build_report(r) -> dict:
    sp, out, grid = r.sp, r.out, r.grid
    seeds, n_boot = grid['final_seeds'], grid.get('n_boot', 1000)
    S = {n: dict(np.load(out / 'scores' / f'{n}.npz')) for n in r.models()}
    B = dict(np.load(out / 'baselines' / 'scores.npz'))
    base_methods = sorted({k.split('__')[0] for k in B})
    tests = sp.test_sets()

    def getter(name):
        """name → f(list of set keys) → score vector (seed ensembles rank-averaged)."""
        if name in QVR_ROWS:
            pre, key = QVR_ROWS[name]
            return lambda ks: rank_avg([np.concatenate([S[f'{pre}_s{s}'][f'{key}__{k}'] for k in ks])
                                        for s in seeds])
        return lambda ks: np.concatenate([B[f'{name}__{k}'] for k in ks])

    rows = list(QVR_ROWS) + base_methods
    rep = {'dataset': sp.manifest.get('name', sp.path), 'data_dir': sp.path, 'T': sp.T,
           'n_channels': sp.n_channels, 'test_sets': tests, 'final_seeds': seeds,
           'n': {k: int(len(v)) for k, v in sp.X.items()}}

    # -- main ------------------------------------------------------------------
    groups = {'pooled': tests, **{k: [k] for k in tests}}
    rep['main'] = {}
    for g, sets in groups.items():
        keys, y, u = sp.pair('Xte_norm', sets)
        ref = getter('qvr_best')(keys)
        rep['main'][g] = {}
        for m in rows:
            s = getter(m)(keys)
            row = auc_ci(y, s, u, n_boot=n_boot) if g == 'pooled' else {'auc': roc_auc(y, s)}
            if g == 'pooled' and m != 'qvr_best':
                row['vs_qvr_best'] = paired_diff_ci(y, s, ref, u, n_boot=n_boot)
            rep['main'][g][m] = row

    # -- per seed, D modes, σ0 ----------------------------------------------------
    keys, y, u = sp.pair('Xte_norm', tests)

    def seed_auc(pre, key):
        return [roc_auc(y, np.concatenate([S[f'{pre}_s{s}'][f'{key}__{k}'] for k in keys])) for s in seeds]
    rep['seeds'] = {pre: _ms(seed_auc(pre, 'mc')) for pre in ('best', 'ref', 'sigma0')}
    rep['d_modes'] = {m: _ms(seed_auc('best', m)) for m in ('mc', 'mean', 'identity', 'averaged')}
    rep['d_modes']['ref'] = {m: _ms(seed_auc('ref', m)) for m in ('mc', 'mean', 'identity', 'averaged')}
    hist = {}
    for pre in ('best', 'sigma0'):
        hs = [json.loads((out / 'models' / f'{pre}_s{s}' / 'history.json').read_text()) for s in seeds]
        hist[pre] = {'best_batch': _ms([h['best_batch'] for h in hs]),
                     'n_batches': _ms([h['n_batches'] for h in hs]),
                     'early_stopped': int(sum(h['stopped'] == 'early_stop' for h in hs))}
    rep['sigma0'] = {'auc_best': rep['seeds']['best'], 'auc_sigma0': rep['seeds']['sigma0'],
                     'paired_pooled': rep['main']['pooled']['qvr_sigma0']['vs_qvr_best'], 'training': hist,
                     'note': 'monitor losses are not comparable (σ = 0 removes the arctan penalty); '
                             'compare AUC and training length'}

    # -- order ------------------------------------------------------------------
    perm_auc, d_auc, rho = [], [], []
    for s in seeds:
        o = np.concatenate([S[f'best_s{s}'][f'mc__{k}'] for k in keys])
        p = np.concatenate([S[f'best_s{s}'][f'perm__{k}'] for k in keys])
        perm_auc.append(roc_auc(y, p))
        d_auc.append(roc_auc(y, o) - roc_auc(y, p))
        rho.append(float(spearmanr(o, p).statistic))
    rep['order'] = {'auc_permuted': _ms(perm_auc), 'delta_auc': _ms(d_auc), 'spearman_orig_vs_perm': _ms(rho)}

    # -- fourier / hamiltonian ------------------------------------------------------
    td = {n: json.loads((out / 'teardown' / f'{n}.json').read_text()) for n in r.models()}
    F = {}
    for n, t in td.items():
        ex, K = t['fourier']['exactness'], t['fourier']['exactness']['K']
        modes = [m for m in ('mean', 'mc', 'averaged')]
        dmp = t['fourier']['damping']
        md = dmp['median_offdiag_damping_by_t']
        F[n] = {'K': K, 'all_hold': t['fourier']['all_hold'],
                'min_r2_z_at_K': min(ex[m]['z'][str(K)][0] for m in modes),
                'max_r2_z_below_K': max(ex[m]['z'][str(K - 1)][0] for m in modes),
                'min_r2_s_at_2K': min(ex[m]['s'][str(2 * K)][0] for m in modes),
                'identity_r2_z_at_1': ex['identity']['z']['1'][0],
                'fresh_draw_r2_median': ex['mc_fresh_draws']['r2_degK_median'],
                'damping_err_over_mc_scale': dmp['max_abs_analytic_vs_mc'] / dmp['mc_se_scale'],
                'damping_mid_t': md[len(md) // 2], 'damping_last_t': md[-1],
                'analytic_vs_mean_mode_mean_abs': dmp['mean_abs_analytic_vs_mean_mode']}
    vals = list(F.values())
    rep['fourier'] = {'per_model': F, 'all_hold_fraction': float(np.mean([v['all_hold'] for v in vals])),
                      'K': sorted({v['K'] for v in vals}),
                      'min_r2_z_at_K': min(v['min_r2_z_at_K'] for v in vals),
                      'max_r2_z_below_K': max(v['max_r2_z_below_K'] for v in vals),
                      'min_r2_s_at_2K': min(v['min_r2_s_at_2K'] for v in vals),
                      'median_damping_last_t_best': float(np.median([F[f'best_s{s}']['damping_last_t'] for s in seeds]))}
    H = {n: t['hamiltonian'] for n, t in td.items()}
    rep['hamiltonian'] = {pre: {'abs_sigma_mean': _ms([np.mean(H[f'{pre}_s{s}']['abs_sigma']) for s in seeds]),
                                'abs_sigma_init_mean': _ms([np.mean(H[f'{pre}_s{s}'].get('abs_sigma_init', [np.nan]))
                                                            for s in seeds]),
                                'mean_abs_dmu': _ms([H[f'{pre}_s{s}'].get('mean_abs_dmu', np.nan) for s in seeds]),
                                'S_r': H[f'{pre}_s{seeds[0]}']['S_r'],
                                'S_r_init': H[f'{pre}_s{seeds[0]}'].get('S_r_init')}
                          for pre in ('best', 'ref')}

    # -- probes -----------------------------------------------------------------------
    pj = out / 'teardown' / 'probes.json'
    P = json.loads(pj.read_text()) if pj.exists() else {}
    if 'skipped' in P or not P:
        rep['probes'] = {'skipped': P.get('skipped', 'not run')}
    else:
        rep['probes'] = {k: v for k, v in P.items() if k in ('moments', 'raw', 'eval_sets')}
        pb = [v['logreg']['macro_auc'] for k, v in P.items() if k.startswith('pbar_best_s') and v]
        first = P.get(f'pbar_best_s{seeds[0]}')
        rep['probes']['pbar_best'] = {'macro_auc': _ms(pb), 'first_seed': first}

    # -- null guard --------------------------------------------------------------------
    un = sp.units['Xte_norm']
    yn = null_halves(un, seed=0)
    dev = {}
    if len(np.unique(yn)) == 2:
        for m in rows:
            dev[m] = roc_auc(yn, getter(m)(['Xte_norm']))
    rep['null_guard'] = {'auc': dev, 'n_units': int(len(np.unique(un))),
                         'max_abs_dev': float(max(abs(v - 0.5) for v in dev.values())) if dev else float('nan')}

    # -- selection -----------------------------------------------------------------------
    sel = json.loads((out / 'select' / 'selection.json').read_text())
    bsel = json.loads((out / 'baselines' / 'selection.json').read_text())
    rep['selection'] = {'qvr_winner': sel['winner'], 'qvr_reference': sel['reference'],
                        'selected_on_val': sel['selected_on_val'],
                        'qvr_top': [{k: t[k] for k in ('tag', 'val_auc_mean')} for t in sel['table'][:8]],
                        'baselines': {m: {'hp': v['hp'], 'val_auc': v['val_auc']} for m, v in bsel.items()}}
    return rep


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v)), 'values': [float(x) for x in v]}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(x, n=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'


def to_markdown(rep) -> str:
    L = [f"# Summary — {rep['dataset']}", '',
         f"Split dir `{rep['data_dir']}` · T = {rep['T']} · channels = {rep['n_channels']} · "
         f"seeds {rep['final_seeds']} · test sets {', '.join(rep['test_sets'])}", '',
         'Sizes: ' + ', '.join(f'{k} {v:,}' for k, v in rep['n'].items()), '']
    sel = rep['selection']
    L += ['## Selection', '',
          f"QVR winner ({'best validation AUC' if sel['selected_on_val'] else 'NOT selected: no validation sets, reference config used'}): "
          f"`{json.dumps(sel['qvr_winner'])}`", '']
    L += ['| QVR config | val AUC |', '|---|---|'] + [f"| {t['tag']} | {_f(t['val_auc_mean'])} |" for t in sel['qvr_top']]
    L += ['', '## C4 — Main comparison (test, pooled anomalies)', '',
          '| method | AUC | 95% unit CI | Δ vs qvr_best [CI] | ' + ' | '.join(rep['test_sets']) + ' |',
          '|---|---|---|---|' + '---|' * len(rep['test_sets'])]
    pooled = rep['main']['pooled']
    for m, v in sorted(pooled.items(), key=lambda kv: -kv[1]['auc']):
        d = v.get('vs_qvr_best')
        dd = f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d else '—'
        per = ' | '.join(_f(rep['main'][s][m]['auc']) for s in rep['test_sets'])
        L.append(f"| {m} | {_f(v['auc'])} | [{_f(v['lo'])}, {_f(v['hi'])}] | {dd} | {per} |")
    L += ['', 'QVR rows are rank-averaged over seeds. Per-seed AUC (mean ± SD): ' +
          ', '.join(f"{k} {_f(v['mean'])} ± {_f(v['sd'])}" for k, v in rep['seeds'].items()), '']
    L += ['## C3 — D modes, σ = 0 ablation, Hamiltonian', '',
          '| D mode | best (mean ± SD over seeds) | ref |', '|---|---|---|']
    for m in ('mc', 'mean', 'identity', 'averaged'):
        a, b = rep['d_modes'][m], rep['d_modes']['ref'][m]
        L.append(f"| {m} | {_f(a['mean'])} ± {_f(a['sd'])} | {_f(b['mean'])} ± {_f(b['sd'])} |")
    s0 = rep['sigma0']
    d = s0['paired_pooled']
    L += ['', f"σ frozen at 0: AUC {_f(s0['auc_sigma0']['mean'])} vs learned σ {_f(s0['auc_best']['mean'])} "
          f"(ensemble Δ {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]). "
          f"Training length (best batch): σ0 {s0['training']['sigma0']['best_batch']['mean']:.0f} vs "
          f"learned {s0['training']['best']['best_batch']['mean']:.0f}.", '']
    for pre, h in rep['hamiltonian'].items():
        sr = lambda d: '—' if not d else ', '.join(f'{k}: {v:.2f}' for k, v in d.items())  # noqa: E731
        L.append(f"- {pre}: mean |σ| {_f(h['abs_sigma_mean']['mean'])} (init {_f(h['abs_sigma_init_mean']['mean'])}), "
                 f"mean |Δμ| {_f(h['mean_abs_dmu']['mean'])}, S_r {sr(h['S_r'])} (init {sr(h['S_r_init'])})")
    f = rep['fourier']
    L += ['', '## C2 — Fourier exactness and order', '',
          f"- All predictions hold on {f['all_hold_fraction']:.0%} of trained models (K = {f['K']}).",
          f"- min R² of z at degree K: {f['min_r2_z_at_K']:.10f}; max R² at K−1: {f['max_r2_z_below_K']:.4f}; "
          f"min R² of the score term at 2K: {f['min_r2_s_at_2K']:.10f}.",
          f"- Median coherence damping at the last timepoint (best models): {f['median_damping_last_t_best']:.3g}.",
          f"- Order: AUC on time-permuted copies {_f(rep['order']['auc_permuted']['mean'])}; "
          f"ΔAUC (orig − perm) {_f(rep['order']['delta_auc']['mean'])}; "
          f"Spearman(orig, perm) {_f(rep['order']['spearman_orig_vs_perm']['mean'])}.", '']
    p = rep['probes']
    L += ['## C5 — Subtype probes (fit on Xprobe, evaluated on test anomalies)', '']
    if 'skipped' in p:
        L.append(f"Skipped: {p['skipped']}")
    else:
        def pr(x):
            if not x:
                return '—'
            lr = x['logreg']
            return f"{_f(lr['macro_auc'])}" + (f" (null p {lr['p_value']:.3f})" if 'p_value' in lr else '')
        L += ['| features | macro AUC (logreg) |', '|---|---|',
              f"| raw series | {pr(p.get('raw'))} |", f"| six moments | {pr(p.get('moments'))} |",
              f"| QVR p̄_b (first seed) | {pr(p['pbar_best']['first_seed'])} |",
              f"| QVR p̄_b (mean over seeds) | {_f(p['pbar_best']['macro_auc']['mean'])} |"]
    ng = rep['null_guard']
    L += ['', '## Protocol — null guard (test normals vs test normals, split by unit)', '',
          f"Max |AUC − 0.5| over all methods: {_f(ng['max_abs_dev'])} ({ng['n_units']} units).", '']
    L += ['## Baseline hyperparameters (selected on validation)', '',
          '| method | hp | val AUC |', '|---|---|---|'] + \
         [f"| {m} | {v['hp']} | {_f(v['val_auc'])} |" for m, v in sel['baselines'].items()]
    return '\n'.join(L) + '\n'

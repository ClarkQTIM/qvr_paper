"""
collate.py — every results/<dataset>/summary.json → the paper's tables and figures.

    python collate.py --results results --out paper [--datasets mitbih baker ...]

Tables (paper/tables/<name>.md and .csv), one per claim:
  T1_main          pooled test AUC [unit CI] per method × dataset                 C4
  T2_dmodes        QVR D modes, σ = 0 ablation, reference config (mean ± SD)       C3
  T3_readouts      the same circuit read out four ways                            C4
  T4_fourier       Fourier exactness and damping across all trained models        C2, C3
  T5_probes        subtype probes: p̄_b vs six moments vs raw                      C5
  T6_protocol      null guard and order sensitivity                               protocol, C2
  T7_predictions   registered predictions P1–P9, mechanically scored
Figures (paper/figures/):
  fig_main.png     dot plot of T1 with CIs
  fig_fourier.png  min R² by trig degree for z and the score term
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MAIN_ORDER = ['qvr_best', 'qvr_ref', 'qvr_best_identity', 'qvr_best_ro_tz2_z', 'qvr_best_ro_mahal_pbar',
              'onenum', 'temporal', 'tz2', 'trig_t', 'moments', 'mahal', 'pca', 'knn', 'iforest']
LABELS = {'qvr_best': 'QVR (tuned)', 'qvr_ref': 'QVR (reference config)', 'qvr_best_identity': 'QVR, D = 0 (cos x)',
          'qvr_best_ro_tz2_z': 'QVR circuit, per-t z² readout', 'qvr_best_ro_mahal_pbar': 'QVR circuit, p̄_b Mahalanobis',
          'qvr_best_ro_mahal_zmean': 'QVR circuit, mean-⟨Z⟩ Mahalanobis', 'qvr_best_mean': 'QVR, D = μ',
          'qvr_best_averaged': 'QVR, averaged D̄', 'qvr_sigma0': 'QVR, σ = 0 trained',
          'onenum': 'one number per series', 'temporal': 'one temporal number (lag-1 / MSD)',
          'tz2': 'per-timepoint z²', 'trig_t': 'per-timepoint trig (QVR class)', 'moments': 'six moments (fingerprint)',
          'mahal': 'Mahalanobis (whole series)', 'pca': 'PCA reconstruction', 'knn': 'kNN distance',
          'iforest': 'Isolation Forest'}


def _f(x, n=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'


def _write(out, name, header, rows, note=''):
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    with open(out / 'tables' / f'{name}.csv', 'w', newline='') as f:
        csv.writer(f).writerows([header] + rows)
    md = [f'# {name}', ''] + ([note, ''] if note else []) + \
         ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + \
         ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    (out / 'tables' / f'{name}.md').write_text('\n'.join(md) + '\n')
    return '\n'.join(md)


def collate(results, out, datasets=None):
    results, out = Path(results), Path(out)
    reps = {}
    for p in sorted(results.glob('*/summary.json')):
        if datasets and p.parent.name not in datasets:
            continue
        reps[p.parent.name] = json.loads(p.read_text())
    if not reps:
        raise SystemExit(f'no summary.json under {results}')
    ds = list(reps)
    print(f'collate: {len(ds)} datasets: {", ".join(ds)}')
    md = []

    # T1 main
    rows = []
    for m in MAIN_ORDER:
        row = [LABELS.get(m, m)]
        for d in ds:
            v = reps[d]['main']['pooled'].get(m)
            row.append(f"{_f(v['auc'])} [{_f(v['lo'])}, {_f(v['hi'])}]" if v else '—')
        rows.append(row)
    md.append(_write(out, 'T1_main', ['method'] + ds, rows,
                     'Pooled test AUC [95% unit-level CI]. QVR rows: rank-averaged over seeds.'))

    # T2 D modes / σ0
    rows = []
    for key, lab in (('mc', 'MC draws (as trained)'), ('mean', 'D = μ'), ('identity', 'D = 0'), ('averaged', 'averaged D̄')):
        rows.append([lab] + [f"{_f(reps[d]['d_modes'][key]['mean'])} ± {_f(reps[d]['d_modes'][key]['sd'])}" for d in ds])
    rows.append(['σ = 0 during training'] + [f"{_f(reps[d]['seeds']['sigma0']['mean'])} ± {_f(reps[d]['seeds']['sigma0']['sd'])}" for d in ds])
    rows.append(['reference config (MC)'] + [f"{_f(reps[d]['seeds']['ref']['mean'])} ± {_f(reps[d]['seeds']['ref']['sd'])}" for d in ds])
    rows.append(['mean |σ| final (init)'] + [f"{_f(reps[d]['hamiltonian']['best']['abs_sigma_mean']['mean'], 2)} "
                                             f"({_f(reps[d]['hamiltonian']['best']['abs_sigma_init_mean']['mean'], 2)})" for d in ds])
    md.append(_write(out, 'T2_dmodes', ['tuned QVR, scored with'] + ds, rows, 'Pooled test AUC, mean ± SD over seeds.'))

    # T3 readouts
    rows = []
    for m in ('qvr_best', 'qvr_best_ro_tz2_z', 'qvr_best_ro_mahal_zmean', 'qvr_best_ro_mahal_pbar'):
        rows.append([LABELS[m]] + [_f(reps[d]['main']['pooled'][m]['auc']) for d in ds])
    md.append(_write(out, 'T3_readouts', ['readout of the same trained circuit'] + ds, rows))

    # T4 fourier
    rows = []
    for d in ds:
        f = reps[d]['fourier']
        rows.append([d, str(f['K']), f"{f['all_hold_fraction']:.0%}", f"{f['min_r2_z_at_K']:.10f}",
                     f"{f['max_r2_z_below_K']:.4f}", f"{f['min_r2_s_at_2K']:.10f}", f"{f['median_damping_last_t_best']:.3g}"])
    md.append(_write(out, 'T4_fourier', ['dataset', 'K', 'models where all predictions hold', 'min R² ⟨Z⟩ @K',
                                         'max R² ⟨Z⟩ @K−1', 'min R² score @2K', 'median damping, last t'], rows,
                     'Across every trained model (best, ref, σ0; all seeds) and every D mode.'))

    # T5 probes
    rows = []
    for d in ds:
        p = reps[d]['probes']
        if 'skipped' in p:
            rows.append([d, '—', '—', '—', p['skipped']])
            continue

        def g(x):
            return _f(x['logreg']['macro_auc']) + (f" (p {x['logreg']['p_value']:.3f})" if x and 'p_value' in x['logreg'] else '') if x else '—'
        rows.append([d, g(p.get('raw')), g(p.get('moments')), _f(p['pbar_best']['macro_auc']['mean']), ''])
    md.append(_write(out, 'T5_probes', ['dataset', 'raw series', 'six moments', 'QVR p̄_b (mean over seeds)', 'note'], rows,
                     'Macro AUC of a logistic probe fit on anomalies from training units, evaluated on test anomalies; '
                     'p = unit-level permutation null.'))

    # T6 protocol / order
    rows = [[d, _f(reps[d]['null_guard']['max_abs_dev']), str(reps[d]['null_guard']['n_units']),
             _f(reps[d]['order']['auc_permuted']['mean']), _f(reps[d]['order']['delta_auc']['mean'])] for d in ds]
    md.append(_write(out, 'T6_protocol', ['dataset', 'null guard: max deviation of AUC from 0.5', 'units', 'QVR AUC, time-permuted',
                                          'ΔAUC (orig − permuted)'], rows))
    # T7 registered predictions
    mark = lambda v: '—' if v is None else ('holds' if v else 'FAILS')  # noqa: E731
    ids = sorted({k for d in ds for k in reps[d].get('scorecard', {})}, key=lambda k: int(k[1:]))
    rows = []
    for k in ids:
        row = [k]
        for d in ds:
            v = reps[d].get('scorecard', {}).get(k)
            if v is None:
                row.append('—')
                continue
            cell = f"{mark(v['pass'])}: {v['value']}"
            if k == 'P9':
                cell += f" (calibrated: {mark(v['calibrated_pass'])}, {v['calibrated']})"
            row.append(cell)
        rows.append(row)
    if rows:
        md.append(_write(out, 'T7_predictions', ['prediction'] + ds, rows,
                         'Registered predictions (docs/REGISTRATION.md), definitions per docs/DEVIATIONS.md.'))
    (out / 'tables' / 'ALL.md').write_text('\n\n'.join(md) + '\n')
    figures(reps, out)
    print(f'  → {out / "tables"} ({len(md)} tables), {out / "figures"}')
    return reps


def figures(reps, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    ds = list(reps)
    ms = MAIN_ORDER
    y = np.arange(len(ms))[::-1]
    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 5.5), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        P = reps[d]['main']['pooled']
        for yi, m in zip(y, ms):
            if m not in P:
                continue
            v = P[m]
            c = '#3b6fb6' if m.startswith('qvr') else '#555555'
            ax.plot([v['lo'], v['hi']], [yi, yi], color=c, lw=1.5)
            ax.plot(v['auc'], yi, 'o', color=c, ms=5)
        ax.axvline(0.5, color='#999', ls=':', lw=1)
        ax.set_title(d)
        ax.set_xlabel('pooled test AUC')
        ax.set_xlim(-0.02, 1.02)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([LABELS.get(m, m) for m in ms], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'fig_main.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(ds))
    ax.bar(x - 0.2, [reps[d]['fourier']['max_r2_z_below_K'] for d in ds], 0.4, label='⟨Z⟩ at degree K−1 (max)')
    ax.bar(x + 0.2, [reps[d]['fourier']['min_r2_z_at_K'] for d in ds], 0.4, label='⟨Z⟩ at degree K (min)')
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_ylabel('R² of trig fit')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'fig_fourier.png', dpi=150)
    plt.close('all')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='paper')
    ap.add_argument('--datasets', nargs='*')
    a = ap.parse_args()
    collate(a.results, a.out, a.datasets)

"""
qvr/teardown.py — the QVR analyses ("take it apart"). Library only; run.py calls these.

  Fourier exactness (Proposition 1–2, claim C2)
    fourier_report   per timepoint: R² of trig fits of z_t(x) = mean_q<Z_q> and of the
                     score term s_t(x), by degree, in every D mode (draws held fixed
                     across x); the fresh-draw residual (inference noise); the damping
                     check (Proposition 3). `verdict` turns it into pass/fail booleans:
                     z exact at degree K, score at 2K, not below K; identity = cos x.
  Readouts (Test A, claim C4)
    readouts         re-read the SAME circuit: native score, per-timepoint z² of <Z>,
                     Mahalanobis of mean <Z>, Mahalanobis of p̄_b (fit on training normals)
  Hamiltonian / σ (claim C3)
    hamiltonian      eigenvalue gaps, interaction-order weights S_r vs init, |σ| vs init
  Subtype probes (claim C5)
    probe            logistic / boosted-tree probe fit on Xprobe (anomalies from TRAINING
                     units), evaluated on test anomalies; unit-level permutation null

Self-test:  python -m qvr.teardown
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from qvr import circuit as C
from qvr.data import t_grid
from qvr.infer import infer

MODES = ('mean', 'identity', 'mc', 'averaged')


# ---------------------------------------------------------------------------
# Fourier exactness and damping (moved from scripts/fourier_check.py, E-F1)
# ---------------------------------------------------------------------------

def n_embed(cfg) -> int:
    """Trig degree of z in each input channel."""
    if cfg.per_qubit_input or cfg.embedding in ('ry_first', 'rx_first'):
        return 1
    return cfg.n_qubits                       # 'ry_all': x on every qubit


def x_points(cfg, n_grid: int, n_rand: int, rng) -> torch.Tensor:
    """(G,) evenly spaced for scalar embeddings; (G, Q) uniform random for per-qubit."""
    if cfg.per_qubit_input:
        return torch.as_tensor(rng.uniform(-math.pi, math.pi, (n_rand, cfg.n_qubits)), dtype=C.DTYPE)
    return torch.linspace(-math.pi, math.pi, n_grid + 1, dtype=C.DTYPE)[:-1]


def trig_basis(x: torch.Tensor, deg: int, per_channel: bool) -> np.ndarray:
    x = x.numpy()
    if not per_channel:
        cols = [np.ones_like(x)] + [f(k * x) for k in range(1, deg + 1) for f in (np.cos, np.sin)]
        return np.stack(cols, 1)
    Q = x.shape[1]
    per = [[np.ones(len(x))] + [f(k * x[:, c]) for k in range(1, deg + 1) for f in (np.cos, np.sin)]
           for c in range(Q)]
    return np.stack([np.prod(combo, axis=0) for combo in itertools.product(*per)], 1)


def r2_by_degree(x, Y: np.ndarray, max_deg: int, per_channel: bool) -> dict:
    """Y: (G, T). Returns {deg: (min R², median R²) over t}."""
    out = {}
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    for d in range(0, max_deg + 1):
        A = trig_basis(x, d, per_channel)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        ssr = ((Y - A @ coef) ** 2).sum(0)
        r2 = np.where(sst > 1e-24, 1 - ssr / np.maximum(sst, 1e-300), 1.0)
        out[d] = (float(r2.min()), float(np.median(r2)))
    return out


def as_input(x: torch.Tensor, T: int) -> torch.Tensor:
    """(G,) → (G, T);  (G, Q) → (G, T, Q): same x at every timepoint."""
    return x.unsqueeze(1).expand(x.shape[0], T, *x.shape[1:]).contiguous()


def z_and_s(params, cfg, X, t, D):
    """Draw-averaged z (G, T) and score term s (G, T) under cfg.cost."""
    zd = C.expvals_from_D(params, X, t, D, cfg, keep_draws=True).mean(-1)   # (G, S, T)
    z = zd.mean(1)
    eta = params['eta_0']
    s = (eta - z) ** 2 / 4 if cfg.cost == 'outside' else ((eta - zd) ** 2 / 4).mean(1)
    return z, s


# ---------------------------------------------------------------------------
# 1 + 2. Exactness and inference-draw noise
# ---------------------------------------------------------------------------

@torch.no_grad()
def exactness(params, cfg, t, x, seed: int) -> dict:
    T, G, K = len(t), x.shape[0], n_embed(cfg)
    X = as_input(x, T)
    g = torch.Generator().manual_seed(seed)
    res = {'K': K}
    for mode in MODES:
        S = cfg.N_E
        if mode == 'mc':                 # same draws for every x (one circuit's view)
            eps = C.sample_eps((1, S, T, cfg.n_terms), g)
        elif mode == 'averaged':
            eps = C.sample_eps((1, S, cfg.n_terms), g)
        else:
            eps = None
        D = C.build_D(params, mode, cfg, 1, T, S, eps=eps)
        z, s = z_and_s(params, cfg, X, t, D)
        res[mode] = {'z': r2_by_degree(x, z.numpy(), K + 1, cfg.per_qubit_input),
                     's': r2_by_degree(x, s.numpy(), 2 * K + 1, cfg.per_qubit_input)}
    # fresh draws per x (what inference does: draws are keyed on series index)
    eps = C.sample_eps((G, cfg.N_E, T, cfg.n_terms), g)
    D = C.build_D(params, 'mc', cfg, G, T, cfg.N_E, eps=eps)
    z, _ = z_and_s(params, cfg, X, t, D)
    A = trig_basis(x, K, cfg.per_qubit_input)
    Y = z.numpy()
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - (resid ** 2).sum(0) / np.maximum(sst, 1e-300)
    # draw noise expected: SD over draws of a single-draw z, / sqrt(N_E)
    zd = C.expvals_from_D(params, X, t, D, cfg, keep_draws=True).mean(-1)   # (G, S, T)
    draw_sd = float((zd.std(1) / math.sqrt(cfg.N_E)).mean())
    res['mc_fresh_draws'] = {'r2_degK_min': float(r2.min()), 'r2_degK_median': float(np.median(r2)),
                             'resid_sd': float(resid.std()), 'expected_draw_sd': draw_sd}
    return res


# ---------------------------------------------------------------------------
# 3. Damping
# ---------------------------------------------------------------------------

@torch.no_grad()
def analytic_expectation(params, cfg, x, t):
    """E_eps[z_t(x)] for mode 'mc' in closed form: (G, T); plus coherence damping (T, dim, dim)."""
    W = C.W_matrix(params['alpha'], cfg)
    phi = C.embed_state(x, cfg).to(C.CDTYPE) @ W.T                     # (G, dim)  W E(x)|0>
    O = C.z_signs(cfg).to(C.DTYPE).mean(-1).to(C.CDTYPE)                # (dim,)  mean_q Z_q
    M = W @ torch.diag(O) @ W.conj().T                                  # W O W†
    P = C.term_parities(cfg).to(C.DTYPE)                                # (n_terms, dim)
    dP = P.unsqueeze(1) - P.unsqueeze(2)                                # (n, b, b'): P_b' − P_b
    mu, sig = params['mu'].to(C.DTYPE), params['sigma'].abs().to(C.DTYPE)
    mean_rate = -0.5 * torch.einsum('j,jab->ab', mu, dP)                # Δθ = rate·t
    var_rate = 0.25 * torch.einsum('j,jab->ab', sig ** 2, dP ** 2)
    damp = torch.exp(-0.5 * var_rate.unsqueeze(0) * t.view(-1, 1, 1) ** 2)          # (T, b, b')
    ph = torch.exp(1j * mean_rate.unsqueeze(0) * t.view(-1, 1, 1))                   # (T, b, b')
    K = (ph * damp).to(C.CDTYPE) * M.unsqueeze(0)                                     # (T, b, b')
    z = torch.einsum('gb,tbc,gc->gt', phi.conj(), K, phi).real
    return z, damp


@torch.no_grad()
def damping(params, cfg, t, x, n_mc: int, seed: int) -> dict:
    T = len(t)
    X = as_input(x, T)
    z_an, damp = analytic_expectation(params, cfg, x, t)
    g = torch.Generator().manual_seed(seed)
    acc, done, chunk = torch.zeros_like(z_an), 0, 500
    while done < n_mc:                                  # same draws for all x, many draws
        s = min(chunk, n_mc - done)
        D = C.build_D(params, 'mc', cfg, 1, T, s, eps=C.sample_eps((1, s, T, cfg.n_terms), g))
        acc += C.expvals_from_D(params, X, t, D, cfg, keep_draws=True).mean(-1).sum(1)
        done += s
    z_mc = acc / n_mc
    z_mu = C.expvals_from_D(params, X, t, C.build_D(params, 'mean', cfg, 1, T), cfg).mean(-1)
    off = ~torch.eye(cfg.dim, dtype=torch.bool)
    med_damp = damp[:, off].median(dim=1).values
    return {'n_mc': n_mc,
            'max_abs_analytic_vs_mc': float((z_an - z_mc).abs().max()),
            'mc_se_scale': float(1 / math.sqrt(n_mc)),
            'max_abs_analytic_vs_mean_mode': float((z_an - z_mu).abs().max()),
            'mean_abs_analytic_vs_mean_mode': float((z_an - z_mu).abs().mean()),
            'median_offdiag_damping_by_t': [float(v) for v in med_damp],
            't': [float(v) for v in t]}


def verdict(res, tol=1e-9) -> dict:
    """The registered predictions, as booleans."""
    ex, K = res['exactness'], res['exactness']['K']
    v = {}
    for mode in MODES:
        z, s = ex[mode]['z'], ex[mode]['s']
        v[f'{mode}_z_exact_at_K'] = z[K][0] > 1 - tol
        v[f'{mode}_s_exact_at_2K'] = s[2 * K][0] > 1 - tol
        if mode == 'identity':
            # D = 0: W cancels, <Z_q> = cos(x_q) → degree 1 in every channel, whatever K is
            v['identity_z_exact_at_1'] = z[1][0] > 1 - tol
        else:
            # bound is tight: degree K-1 is not enough (identity is the exception above)
            v[f'{mode}_z_not_exact_below_K'] = z[K - 1][0] < 1 - 1e-6
    d = res['damping']
    v['damping_matches_mc'] = d['max_abs_analytic_vs_mc'] < 6 * d['mc_se_scale']
    return v




def fourier_report(params, cfg, T, n_grid=360, n_rand=2000, n_mc=4000, seed=0, max_t=64) -> dict:
    """Everything for claims C2/C3 on one trained model; JSON-ready. The claims are per
    timepoint, so long series are checked on max_t evenly spaced timepoints of the model's
    own grid t_grid(T) (memory stays bounded for T in the thousands)."""
    params = {k: v.detach().to(C.DTYPE) for k, v in params.items()}
    t = t_grid(T)
    if T > max_t:
        t = t[torch.as_tensor(np.unique(np.linspace(0, T - 1, max_t).round().astype(int)))]
    x = x_points(cfg, n_grid, n_rand, np.random.default_rng(seed))
    res = {'T': T, 'n_points': int(x.shape[0]), 'embedding': cfg.embedding,
           'exactness': exactness(params, cfg, t, x, seed),
           'damping': damping(params, cfg, t, x if not cfg.per_qubit_input else x[:500], n_mc, seed + 1)}
    res['verdict'] = {k: bool(v) for k, v in verdict(res).items()}
    res['all_hold'] = all(res['verdict'].values())
    return _jsonable(res)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ---------------------------------------------------------------------------
# Readouts (Test A)
# ---------------------------------------------------------------------------

READOUTS = ('native', 'tz2_z', 'mahal_zmean', 'mahal_pbar')


def _mahal_fit(F):
    from sklearn.covariance import LedoitWolf
    mu, P = F.mean(0), LedoitWolf().fit(F).precision_
    return lambda G: np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G - mu, P, G - mu), 0))


@torch.no_grad()
def readouts(params, cfg, Xtr, sets: dict, n_fit=5000, seed=0, mode='mc', batch_size=4096) -> dict:
    """{readout: {set: scores}}. Readouts are fit on a subsample of training normals."""
    t = t_grid(Xtr.shape[1])
    rng = np.random.default_rng(seed)
    idx = torch.as_tensor(np.sort(rng.choice(len(Xtr), min(n_fit, len(Xtr)), replace=False)))
    o_tr = infer(params, cfg, Xtr[idx], t, mode=mode, seed=seed, keep_time=True, batch_size=batch_size)
    mu_t, sd_t = o_tr['z_t'].mean(0), o_tr['z_t'].std(0) + 1e-9
    m_z, m_p = _mahal_fit(o_tr['zrepr']), _mahal_fit(o_tr['pbar'])
    out = {r: {} for r in READOUTS}
    for k, Xk in sets.items():
        o = infer(params, cfg, Xk, t, mode=mode, seed=seed, keep_time=True, batch_size=batch_size)
        out['native'][k] = o['score']
        out['tz2_z'][k] = (((o['z_t'] - mu_t) / sd_t) ** 2).mean((1, 2))
        out['mahal_zmean'][k] = m_z(o['zrepr'])
        out['mahal_pbar'][k] = m_p(o['pbar'])
    return out


# ---------------------------------------------------------------------------
# Hamiltonian / σ
# ---------------------------------------------------------------------------

def _S_r(mu, cfg):
    mu = np.abs(np.asarray(mu))
    orders = [len(c) for c in cfg.terms]
    return {str(r): float(mu[[i for i, o in enumerate(orders) if o == r]].sum() / mu.sum())
            for r in sorted(set(orders))}


def hamiltonian(params, cfg, init=None) -> dict:
    """Eigenvalue gaps and interaction-order weights, against the initialization: S_r of a
    random init is already ≈ (#terms of order r)/(#terms), so only the change is informative."""
    p = {k: v.detach() for k, v in params.items()}
    lam = C.eigenvalues(p['mu'], cfg).numpy()
    out = {'min_gap': float(np.diff(np.sort(lam)).min()), 'S_r': _S_r(p['mu'].numpy(), cfg),
           'abs_sigma': [float(v) for v in p['sigma'].abs()], 'eta_0': float(p['eta_0'])}
    if init is not None:
        i = {k: v.detach() for k, v in init.items()}
        out['S_r_init'] = _S_r(i['mu'].numpy(), cfg)
        out['mean_abs_dmu'] = float((p['mu'] - i['mu']).abs().mean())
        out['abs_sigma_init'] = [float(v) for v in i['sigma'].abs()]
    return out


# ---------------------------------------------------------------------------
# Subtype probes
# ---------------------------------------------------------------------------

def probe(F_fit, y_fit, u_fit, F_eval, y_eval, n_perm=0, seed=0, min_per_class=5) -> dict | None:
    """Multiclass probe of subtype from features. Classes kept only if they have
    ≥ min_per_class UNITS in the fit set and appear in the eval set. Unit-level null:
    unit labels are permuted (a unit's series keep their joint label mix)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y_fit, y_eval, u_fit = (np.asarray(a).astype(str) for a in (y_fit, y_eval, u_fit))
    unknown = {'NA', '', 'nan', 'None'}
    classes = [c for c in np.unique(y_fit) if c not in unknown
               and len(np.unique(u_fit[y_fit == c])) >= min_per_class and (y_eval == c).any()]
    if len(classes) < 2:
        return None
    mf, me = np.isin(y_fit, classes), np.isin(y_eval, classes)
    Ff, yf, uf, Fe, ye = F_fit[mf], y_fit[mf], u_fit[mf], F_eval[me], y_eval[me]

    def metrics(model, labels):
        model.fit(Ff, labels)
        P = model.predict_proba(Fe)
        cls = list(model.classes_)
        pred = np.array(cls)[P.argmax(1)]
        if len(cls) == 2:
            auc = roc_auc_score(ye == cls[1], P[:, 1])
        else:
            auc = float(np.mean([roc_auc_score(ye == c, P[:, i]) for i, c in enumerate(cls)
                                 if (ye == c).any() and (ye != c).any()]))
        return {'macro_auc': float(auc), 'bal_acc': float(balanced_accuracy_score(ye, pred)),
                'kappa': float(cohen_kappa_score(ye, pred))}

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))  # noqa: E731
    out = {'classes': [str(c) for c in classes], 'n_fit': int(len(yf)), 'n_eval': int(len(ye)),
           'logreg': metrics(lr(), yf),
           'hgb': metrics(HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0), yf)}
    if n_perm:
        rng = np.random.default_rng(seed)
        uu = np.unique(uf)
        idx = {k: np.flatnonzero(uf == k) for k in uu}
        null = []
        for _ in range(n_perm):
            perm = dict(zip(uu, rng.permutation(uu)))
            yp = np.empty_like(yf)
            for k in uu:
                yp[idx[k]] = rng.choice(yf[idx[perm[k]]], len(idx[k]), replace=True)
            if len(np.unique(yp)) >= 2:
                null.append(metrics(lr(), yp)['macro_auc'])
        null = np.asarray(null)
        out['logreg']['null_mean'] = float(null.mean()) if len(null) else float('nan')
        out['logreg']['null_p95'] = float(np.quantile(null, 0.95)) if len(null) else float('nan')
        out['logreg']['p_value'] = float((1 + (null >= out['logreg']['macro_auc']).sum()) / (1 + len(null)))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    print('=' * 70)
    print('qvr.teardown self-test (random params: predictions must hold exactly)')
    print('=' * 70)
    print('Fourier exactness + damping')
    for emb, Q in (('ry_all', 3), ('ry_first', 2), ('rx_first', 2), ('rx_per_qubit', 3)):
        for cost in ('outside', 'inside'):
            ans = 'chain_full' if emb == 'rx_first' else 'chain'
            cfg = C.QVRConfig(n_qubits=Q, n_layers=3, N_E=10, embedding=emb, cost=cost, ansatz=ans)
            p = {k: v.detach() for k, v in C.init_params(cfg, np.random.default_rng(3)).items()}
            p['sigma'] = p['sigma'] * 0.1
            r = fourier_report(p, cfg, T=9, n_grid=256, n_rand=1500, n_mc=20000, seed=1)
            bad = [k for k, v in r['verdict'].items() if not v]
            f = r['exactness']['mc_fresh_draws']
            check(f'{emb:12s} Q={Q} {cost:7s} K={r["exactness"]["K"]}: all predictions hold'
                  + (f'  failed {bad}' if bad else ''), not bad,
                  f"fresh-draw resid {f['resid_sd']:.3f} vs draw SD {f['expected_draw_sd']:.3f}")
    cfg = C.QVRConfig(n_qubits=3, n_layers=3)
    p = {k: v.detach() for k, v in C.init_params(cfg, np.random.default_rng(4)).items()}
    p0 = dict(p, sigma=torch.zeros_like(p['sigma']))
    d = damping(p0, cfg, t_grid(5), x_points(cfg, 64, 0, np.random.default_rng(0)), 200, 0)
    check('σ = 0 → analytic == MC == D=μ', d['max_abs_analytic_vs_mean_mode'] < 1e-12
          and d['max_abs_analytic_vs_mc'] < 1e-12)

    print('Readouts')
    g = torch.Generator().manual_seed(0)
    Xtr = (torch.rand(300, 9, generator=g, dtype=C.DTYPE) - 0.5) * 2
    Xa = (torch.rand(50, 9, generator=g, dtype=C.DTYPE) - 0.5) * 6
    ro = readouts(p, cfg, Xtr, {'n': Xtr[:60], 'a': Xa}, n_fit=200)
    check('four readouts × two sets, finite', all(np.isfinite(ro[r][k]).all() for r in READOUTS for k in ('n', 'a')))
    from qvr.data import roc_auc
    y = np.r_[np.zeros(60), np.ones(50)]
    a = roc_auc(y, np.r_[ro['tz2_z']['n'], ro['tz2_z']['a']])
    check('tz2_z detects a variance anomaly', a > 0.8, f'AUC {a:.3f}')

    print('Hamiltonian')
    h = hamiltonian(p, cfg, init=p)
    check('S_r sums to 1, Δμ vs itself = 0', abs(sum(h['S_r'].values()) - 1) < 1e-12 and h['mean_abs_dmu'] == 0)

    print('Probe')
    rng = np.random.default_rng(0)
    uf = np.repeat([f'u{i}' for i in range(30)], 10)
    yf = np.repeat(np.array(['a', 'b'])[rng.integers(0, 2, 30)], 10)
    Ff = rng.normal(size=(300, 4)) + (yf == 'b')[:, None] * 1.5
    ye = np.array(['a', 'b'] * 50)
    Fe = rng.normal(size=(100, 4)) + (ye == 'b')[:, None] * 1.5
    pr = probe(Ff, yf, uf, Fe, ye, n_perm=50)
    check('real signal: high AUC, beats the unit-level null', pr['logreg']['macro_auc'] > 0.85
          and pr['logreg']['p_value'] < 0.05, f"{pr['logreg']['macro_auc']:.3f}, p {pr['logreg']['p_value']:.3f}")
    pr0 = probe(rng.normal(size=(300, 4)), yf, uf, rng.normal(size=(100, 4)), ye, n_perm=50)
    check('no signal: AUC ≈ 0.5', abs(pr0['logreg']['macro_auc'] - 0.5) < 0.12, f"{pr0['logreg']['macro_auc']:.3f}")
    check('classes with < 5 units dropped', probe(Ff, np.where(uf == 'u0', 'c', yf), uf, Fe,
                                                  np.append(ye[:-1], 'c'))['classes'] == ['a', 'b'])
    yna = np.where(uf < 'u05', 'NA', yf)
    check("'NA' labels ignored", 'NA' not in probe(Ff, yna, uf, Fe, ye)['classes'])
    cfg2 = C.QVRConfig(n_qubits=2, n_layers=2, embedding='rx_first', ansatz='chain_full', N_E=4)
    p2 = {k: v.detach() for k, v in C.init_params(cfg2, np.random.default_rng(6)).items()}
    r2 = fourier_report(p2, cfg2, T=4096, n_grid=64, n_mc=200)
    check('long series (T = 4096) checked on 64 timepoints, predictions hold',
          len(r2['damping']['t']) == 64 and r2['all_hold'])
    print('=' * 70)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()

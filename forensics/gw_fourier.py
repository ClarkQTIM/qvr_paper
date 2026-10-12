"""gw_fourier.py — E-F1 on the QVR4GW authors' own trained circuits.
Embedding is RX(x) on one qubit, so <H> must be exactly degree 1 in x at every t
(a + b cos x + c sin x) and the score term exactly degree 2.
Usage: python forensics/gw_fourier.py /path/to/QVR4GW   (needs gw_sim.py alongside)"""
import pickle, sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import gw_sim as G
R = sys.argv[1].rstrip('/') + '/'
def r2(x, Y, d):
    A = np.stack([np.ones_like(x)] + [f(j * x) for j in range(1, d + 1) for f in (np.cos, np.sin)], 1)
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return (1 - ((Y - A @ c) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).min()
for kind in ('filtered', 'unfiltered'):
    for k in range(1, 6):
        op = pickle.load(open(R + f'results/{kind}_fold_{k}/training.pkl', 'rb'))['opt_params']
        p = {kk: np.asarray(v.detach(), float).ravel() for kk, v in op.items()}
        p['alphas'] = p['alphas'].reshape(3, 2, 3); eta = float(p['eta_0'][0])
        x = np.linspace(-np.pi, np.pi, 513)[:-1]; t = np.linspace(0.1, 2 * np.pi, 64)
        z = G.expvals(np.repeat(x[:, None], 64, 1), t, p['alphas'], p['mus'], p['sigmas'], 10,
                      np.random.default_rng(0))            # same draws for every x
        s = (eta - z) ** 2
        print(f'{kind:10s} fold {k}: z R2 deg0 {r2(x, z, 0):.4f} deg1 {r2(x, z, 1):.12f} | '
              f'score R2 deg1 {r2(x, s, 1):.4f} deg2 {r2(x, s, 2):.12f}')

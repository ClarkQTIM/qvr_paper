"""
gw_sim.py — exact 2-qubit statevector re-implementation of QVR4GW's rewinder.py
circuit (no shots), plus their utils.rescale, for forensic checks.

Circuit (rewinder.get_anomaly_expec):
  AngleEmbedding(x) with 1 feature  -> RX(x) on wire 0
  W(alpha): for each layer: RX,RY,RZ on each qubit, then CNOT(0,1)
  D: RZ(g0 t) q0, RZ(g1 t) q1, CNOT(0,1) RZ(g2 t) q1 CNOT(0,1)   (k=2)
  W^dagger, measure 0.5 (Z0 + Z1)
Score per series = mean_t (eta0 - mean_{N_E draws} <H>)^2,  t = linspace(0.1, 2pi, T)
Draws: g ~ Normal(mu, |sigma|).  Wire 0 = most significant bit (PennyLane).

Self-test: python gw_sim.py --self-test
"""
import argparse
import numpy as np

I2 = np.eye(2, dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)


def rx(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


def W_matrix(alphas):
    U = np.eye(4, dtype=complex)
    for layer in alphas:
        g = [rz(layer[q][2]) @ ry(layer[q][1]) @ rx(layer[q][0]) for q in range(2)]
        U = CNOT @ np.kron(g[0], g[1]) @ U
    return U


def D_eigs(g):
    """Eigenphases of D for bitstrings 00,01,10,11 (wire0 = MSB). g: (..., 3)."""
    z0 = np.array([1, 1, -1, -1]); z1 = np.array([1, -1, 1, -1])
    # RZ(a) = exp(-i a Z/2); CNOT·RZ(q1)·CNOT applies exp(-i a Z0Z1/2)
    ph = g[..., 0:1] * z0 + g[..., 1:2] * z1 + g[..., 2:3] * (z0 * z1)
    return np.exp(-0.5j * ph)


def expvals(x, t, alphas, mus, sigmas, n_draws, rng):
    """x: (N, T) rescaled; returns (N, T) mean over draws of 0.5<Z0+Z1>."""
    W = W_matrix(alphas)
    N, T = x.shape
    c, s = np.cos(x / 2), np.sin(x / 2)
    psi = np.zeros((N, T, 4), complex)
    psi[..., 0] = c; psi[..., 2] = -1j * s          # RX on wire 0 applied to |00>
    phi = psi @ W.T                                  # W psi
    H = np.array([1, 0, 0, -1.0])                    # 0.5(z0+z1) diag
    out = np.zeros((N, T))
    for _ in range(n_draws):
        g = rng.normal(mus, np.abs(sigmas), size=(T, 3))   # fresh draw per (t, draw), as in rewinder.py
        d = D_eigs(t[:, None] * g)                   # (T, 4)
        chi = phi * d[None]                          # D W psi
        fin = chi @ W.conj()                         # W^† (row-vector form)
        out += (np.abs(fin) ** 2 @ H)
    return out / n_draws


def scores(x, params, n_draws=10, seed=0, t_start=0.1):
    T = x.shape[1]
    t = np.linspace(t_start, 2 * np.pi, T)
    rng = np.random.default_rng(seed)
    z = expvals(x, t, params['alphas'], params['mus'], params['sigmas'], n_draws, rng)
    return ((params['eta_0'] - z) ** 2).mean(1)


def rescale(X, eps=0.01):
    """QVR4GW utils.rescale: per-timepoint min-max over THE SET PASSED IN."""
    lo, hi = X.min(0, keepdims=True), X.max(0, keepdims=True)
    return (X - lo) / (hi - lo) * (2 * np.pi - 2 * eps) + (-np.pi + eps)


def _self_test():
    import pennylane as qml
    import torch
    rng = np.random.default_rng(1)
    alphas = rng.uniform(0, 2 * np.pi, (3, 2, 3))
    g = rng.normal(size=3)
    dev = qml.device('default.qubit', wires=2)

    def tf(a):
        for layer in range(3):
            for q in range(2):
                qml.RX(a[layer][q][0], wires=q); qml.RY(a[layer][q][1], wires=q); qml.RZ(a[layer][q][2], wires=q)
            qml.CNOT(wires=[0, 1])

    @qml.qnode(dev)
    def circ(x, D):
        qml.AngleEmbedding(np.array([x]), wires=range(2))
        tf(alphas)
        qml.RZ(D[0], wires=0); qml.RZ(D[1], wires=1)
        qml.CNOT(wires=[0, 1]); qml.RZ(D[2], wires=1); qml.CNOT(wires=[0, 1])
        qml.adjoint(tf)(alphas)
        return qml.expval(qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(0), qml.PauliZ(1)]))

    err = 0
    for x, t in [(0.3, 1.1), (-2.0, 4.0), (2.9, 0.1)]:
        ref = float(circ(x, g * t))
        W = W_matrix(alphas)
        psi = np.array([np.cos(x / 2), 0, -1j * np.sin(x / 2), 0])
        fin = W.conj().T @ (D_eigs((g * t)[None])[0] * (W @ psi))
        mine = float(np.abs(fin) ** 2 @ np.array([1, 0, 0, -1.0]))
        err = max(err, abs(ref - mine))
    print(f'  [{"PASS" if err < 1e-10 else "FAIL"}] matches PennyLane rewinder circuit (max |Δ| {err:.1e})')
    X = rng.normal(size=(50, 7))
    r = rescale(X)
    ok = np.allclose(r.min(0), -np.pi + 0.01) and np.allclose(r.max(0), np.pi - 0.01)
    print(f'  [{"PASS" if ok else "FAIL"}] rescale spans [-pi+eps, pi-eps] per timepoint')
    p = dict(alphas=alphas, mus=g, sigmas=np.abs(g), eta_0=0.1)
    s = scores(rescale(X), p)
    ok2 = s.shape == (50,) and np.all(s >= 0)
    print(f'  [{"PASS" if ok2 else "FAIL"}] scores shape {s.shape}, non-negative')
    print('RESULT:', 'ALL PASS' if (err < 1e-10 and ok and ok2) else 'FAIL')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--self-test', action='store_true')
    if ap.parse_args().self_test:
        _self_test()

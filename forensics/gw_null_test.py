# Usage: python forensics/gw_null_test.py /path/to/QVR4GW
"""Null test: QVR4GW pipeline on signal-free surrogates. Only difference between
'noise' and 'events' is the size of the set passed to utils.rescale."""
import pickle, sys, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import gw_sim as G
R = (sys.argv[1] if len(sys.argv) > 1 else 'QVR4GW').rstrip('/') + '/'   # path to a QVR4GW clone
def ld(f): return pickle.load(open(R+f,'rb'))
def a(x): return np.asarray(x.detach() if hasattr(x,'detach') else x,float).ravel()
def surrogates(E, n, rng):
    F=np.fft.rfft(E,axis=1); idx=rng.integers(0,len(E),n)
    ph=np.exp(2j*np.pi*rng.random((n,F.shape[1]))); ph[:,0]=1; ph[:,-1]=1
    return np.fft.irfft(np.abs(F[idx])*ph, n=E.shape[1], axis=1)
rng=np.random.default_rng(0)
N_POOL=5000
for kind in ['filtered','unfiltered']:
    E=np.asarray(pd.read_pickle(R+('data/events.pkl' if kind=='filtered' else 'data/events_unfiltered.pkl')))
    pool=surrogates(E,N_POOL,rng)          # stands in for the noise file
    fake_ev=surrogates(E,78,rng)           # signal-free 'events'
    noise_test=G.rescale(pool)[2078:2156]  # paper: rescale whole noise file, take test slice
    for k in range(1,6):
        op=ld(f'results/{kind}_fold_{k}/training.pkl')['opt_params']
        p={kk:a(v) for kk,v in op.items()}; p['alphas']=p['alphas'].reshape(3,2,3); p['eta_0']=float(p['eta_0'][0])
        sn=G.scores(noise_test,p,seed=10+k)
        auc=lambda s1: roc_auc_score(np.r_[np.zeros(78),np.ones(78)],np.r_[sn,s1])
        s_real_sep=G.scores(G.rescale(E),p,seed=20+k)                 # paper pipeline, real events
        s_fake_sep=G.scores(G.rescale(fake_ev),p,seed=30+k)           # NO signal, scaled separately
        lo,hi=pool.min(0),pool.max(0)                                  # joint: events use noise scaling
        joint=lambda X: (X-lo)/(hi-lo)*(2*np.pi-0.02)-np.pi+0.01
        s_real_joint=G.scores(joint(E),p,seed=40+k)
        s_fake_joint=G.scores(joint(fake_ev),p,seed=50+k)
        print(f'{kind:10s} fold {k}: real events/separate {auc(s_real_sep):.3f} | SURROGATE/separate {auc(s_fake_sep):.3f} | real/joint {auc(s_real_joint):.3f} | surrogate/joint {auc(s_fake_joint):.3f}',flush=True)
# spread after scaling, as a function of set size
E=np.asarray(pd.read_pickle(R+'data/events.pkl')); pool=surrogates(E,N_POOL,rng)
for n in [78,500,N_POOL]:
    print(f'set size {n:5d}: std of rescaled values {G.rescale(pool[:n]).std():.3f}')

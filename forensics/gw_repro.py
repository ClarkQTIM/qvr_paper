# Usage: python forensics/gw_repro.py /path/to/QVR4GW
import pickle, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import gw_sim as G
R = (sys.argv[1] if len(sys.argv) > 1 else 'QVR4GW').rstrip('/') + '/'   # path to a QVR4GW clone
def ld(f): return pickle.load(open(R+f,'rb'))
def a(x): return np.asarray(x.detach() if hasattr(x,'detach') else x,float).ravel()
ev={'filtered':np.asarray(pd.read_pickle(R+'data/events.pkl')),'unfiltered':np.asarray(pd.read_pickle(R+'data/events_unfiltered.pkl'))}
for kind in ['filtered','unfiltered']:
  Xn=G.rescale(ev[kind])
  for k in range(1,6):
    d=f'results/{kind}_fold_{k}/'
    op=ld(d+'training.pkl')['opt_params']; p={kk:a(v) for kk,v in op.items()}; p['alphas']=p['alphas'].reshape(3,2,3); p['eta_0']=float(p['eta_0'][0])
    try: te=ld(d+'testing.pkl'); ta=a(te[5])
    except FileNotFoundError: ta=a(ld(d+'testing_anom.pkl')[5])
    s=G.scores(Xn,p,n_draws=10,seed=k)
    print(f'{kind:10s} fold {k}: mine mean {s.mean():.4f} sd {s.std():.4f} | theirs mean {ta.mean():.4f} sd {ta.std():.4f} | corr {np.corrcoef(s,ta)[0,1]:.3f}')

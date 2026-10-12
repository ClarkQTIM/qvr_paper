# Usage: python forensics/gw_spread_curve.py /path/to/QVR4GW
import pickle, sys, numpy as np, torch
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import gw_sim as G
R = (sys.argv[1] if len(sys.argv) > 1 else 'QVR4GW').rstrip('/') + '/'   # path to a QVR4GW clone
def a(x): return np.asarray(x.detach() if hasattr(x,'detach') else x,float).ravel()
rng=np.random.default_rng(0)
Z=rng.standard_normal((78,4096))
for kind in ['filtered','unfiltered']:
  for k in range(1,6):
    op=pickle.load(open(R+f'results/{kind}_fold_{k}/training.pkl','rb'))['opt_params']
    p={kk:a(v) for kk,v in op.items()}; p['alphas']=p['alphas'].reshape(3,2,3); p['eta_0']=float(p['eta_0'][0])
    te=None
    try: te=pickle.load(open(R+f'results/{kind}_fold_{k}/testing.pkl','rb'))
    except FileNotFoundError: te=pickle.load(open(R+f'results/{kind}_fold_{k}/testing_nonanom.pkl','rb'))
    theirs=a(te[4]).mean()
    row=[]
    for s in [0.05,0.2,0.4,0.6,0.84,1.0,1.34]:
        row.append(G.scores(np.clip(Z*s,-np.pi,np.pi),p,seed=1).mean())
    print(f'{kind:10s} f{k} their noise mean {theirs:.4f} | score at scaled std 0.05..1.34: '+' '.join(f'{v:.4f}' for v in row))

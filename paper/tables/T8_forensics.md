# T8_forensics

Published pipelines run as written (forensics/). BA = balanced accuracy with the threshold tuned as published. GW one-number baseline = per-series std after the rung's scaling (0 or 1 = perfect separation).

| claim | pipeline | published BA | reproduced BA | QVR, other | one-number baseline AUC |
|---|---|---|---|---|---|
| GW | paper, Table 3 (filtered; 5 runs) | 0.919 | — | AUC 0.917 (their saved scores) | — |
| GW | rung A: their protocol: separate scaling, validation events = test events | — | 0.903 | AUC 0.949 (folds 1.000, 0.988, 0.835, 1.000, 0.924) | 0.000 |
| GW | rung A_their: their protocol, the authors' own events | — | 0.877 | AUC 0.922 (folds 1.000, 0.976, 0.746, 1.000, 0.887) | 0.000 |
| GW | rung A0: their protocol, NO SIGNAL (noise windows as "events") | — | 0.898 | AUC 0.944 (folds 1.000, 0.984, 0.869, 0.994, 0.875) | 0.000 |
| GW | rung B: one scaler fit on training noise | — | 0.506 | AUC 0.489 (folds 0.526, 0.432, 0.459, 0.494, 0.534) | 0.714 |
| GW | rung B0: one scaler, NO SIGNAL | — | 0.535 | AUC 0.534 (folds 0.615, 0.519, 0.503, 0.502, 0.530) | 0.504 |
| GW | rung C: one scaler, disjoint validation / test events | — | 0.512 | AUC 0.464 (folds 0.582, 0.310, 0.507, 0.425, 0.496) | 0.715 |

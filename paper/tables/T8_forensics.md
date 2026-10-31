# T8_forensics

Published pipelines run as written (forensics/). BA = balanced accuracy with the threshold tuned as published. GW one-number baseline = per-series std after the rung's scaling (0 or 1 = perfect separation).

| claim | pipeline | published BA | reproduced BA | QVR, other | one-number baseline AUC |
|---|---|---|---|---|---|
| Baker | QVR reproduction, notebook variant (5 runs; baker) | 0.675 | 0.642 | median 0.592 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker) | 0.820 | 0.683 | median 0.621 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker_fu_bi_train100) | 0.820 | 0.683 | median 0.621 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker_fu_bi_train782) | 0.820 | 0.692 | median 0.621 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker_fu_bi_train782_normscale) | 0.820 | 0.692 | median 0.658 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker_fu_tri_train100) | 0.820 | 0.683 | median 0.592 | — |
| Baker | QVR reproduction, paper variant (50 runs; baker_fu_tri_train782) | 0.820 | 0.675 | median 0.588 | — |
| Baker | QVR reproduction, nb_tau5 variant (50 runs; baker_nb_tau5) | 0.675 | 0.725 | median 0.608 | — |
| Baker | QVR reproduction, paper_iter variant (10 runs; baker_paper_iter) | 0.820 | 0.675 | median 0.629 | — |
| Baker | QVR reproduction, paper_ry variant (50 runs; baker_paper_ry) | 0.820 | 0.700 | median 0.617 | — |
| Baker | one-number baseline vol_mean, no training (data of: baker, baker_fu_bi_train100, baker_fu_tri_train100, baker_nb_tau5, baker_paper_iter, baker_paper_ry) | — | 0.675 | — | 0.678 (mean over sets) |
| Baker | one-number baseline chan_mean_sd, no training (data of: baker, baker_fu_bi_train100, baker_nb_tau5, baker_paper_iter, baker_paper_ry) | — | 0.583 | — | 0.606 (mean over sets) |
| Baker | one-number baseline trig_moments, no training (data of: baker, baker_fu_bi_train100, baker_nb_tau5, baker_paper_iter, baker_paper_ry) | — | 0.675 | — | 0.667 (mean over sets) |
| Baker | one-number baseline vol_mean, no training (data of: baker_fu_bi_train782, baker_fu_bi_train782_normscale, baker_fu_tri_train782) | — | 0.667 | — | 0.644 (mean over sets) |
| Baker | one-number baseline chan_mean_sd, no training (data of: baker_fu_bi_train782) | — | 0.575 | — | 0.600 (mean over sets) |
| Baker | one-number baseline trig_moments, no training (data of: baker_fu_bi_train782) | — | 0.625 | — | 0.629 (mean over sets) |
| Baker | one-number baseline chan_mean_sd, no training (data of: baker_fu_bi_train782_normscale) | — | 0.575 | — | 0.602 (mean over sets) |
| Baker | one-number baseline trig_moments, no training (data of: baker_fu_bi_train782_normscale) | — | 0.617 | — | 0.635 (mean over sets) |
| Baker | one-number baseline chan_mean_sd, no training (data of: baker_fu_tri_train100) | — | 0.617 | — | 0.641 (mean over sets) |
| Baker | one-number baseline trig_moments, no training (data of: baker_fu_tri_train100) | — | 0.658 | — | 0.681 (mean over sets) |
| Baker | one-number baseline chan_mean_sd, no training (data of: baker_fu_tri_train782) | — | 0.600 | — | 0.630 (mean over sets) |
| Baker | one-number baseline trig_moments, no training (data of: baker_fu_tri_train782) | — | 0.625 | — | 0.654 (mean over sets) |
| Baker | **best of all 415 reproduction runs** (nb_tau5; baker_nb_tau5) | 0.820 (paper) / 0.675 (notebook) | 0.725 | — | — |
| GW | paper, Table 3 (filtered; 5 runs) | 0.919 | — | AUC 0.917 (their saved scores) | — |
| GW | rung A: their protocol: separate scaling, validation events = test events | — | 0.903 | AUC 0.949 (folds 1.000, 0.988, 0.835, 1.000, 0.924) | 0.000 |
| GW | rung A_their: their protocol, the authors' own events | — | 0.877 | AUC 0.922 (folds 1.000, 0.976, 0.746, 1.000, 0.887) | 0.000 |
| GW | rung A0: their protocol, NO SIGNAL (noise windows as "events") | — | 0.898 | AUC 0.944 (folds 1.000, 0.984, 0.869, 0.994, 0.875) | 0.000 |
| GW | rung B: one scaler fit on training noise | — | 0.506 | AUC 0.489 (folds 0.526, 0.432, 0.459, 0.494, 0.534) | 0.714 |
| GW | rung B0: one scaler, NO SIGNAL | — | 0.535 | AUC 0.534 (folds 0.615, 0.519, 0.503, 0.502, 0.530) | 0.504 |
| GW | rung C: one scaler, disjoint validation / test events | — | 0.512 | AUC 0.464 (folds 0.582, 0.310, 0.507, 0.425, 0.496) | 0.715 |

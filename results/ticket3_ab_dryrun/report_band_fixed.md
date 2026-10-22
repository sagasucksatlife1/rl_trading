# Walk-forward report: band_fixed

Grid `20261022-185547_band_fixed_039a845bde` | data hash `41567179a4` | source `synthetic_pit` | point-in-time: **True** | folds 4 x seeds 3 | 150000 steps/seed | reward `dsr` (kappa 0) | band 0.03

## Verdict

**DRY RUN (synthetic_pit data, not a real-data verdict): would read as -> FAIL - median-seed Sharpe at 2x costs -0.66 <= GBM+no-trade-band 0.51 (ensemble -0.37, worst seed -0.87). Ship the GBM+no-trade-band rule; archive the RL allocator as research.**

## Strategies, out of sample, 1x costs

```
                      sharpe   cagr  ann_vol  max_dd  turnover_ann  cost_drag_ann  avg_gross  beta_to_bench
rl_ensemble           -0.013 -0.008    0.115   0.366        55.262          0.042      0.760          0.824
equal_weight_monthly   0.213  0.018    0.112   0.327         2.481          0.001      0.773          0.822
buy_and_hold           0.282  0.018    0.074   0.205         0.612          0.000      0.492          0.560
gbm_linear_monthly     0.322  0.029    0.108   0.264         6.446          0.005      0.691          0.757
gbm_mean_variance      0.422  0.033    0.085   0.181         5.792          0.006      0.448          0.538
gbm_linear_band        0.644  0.069    0.114   0.310        20.276          0.016      0.756          0.776
```

## RL seeds (stitched OOS): median / worst / best

```
        sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
median     -0.152     -0.663     -1.052        65.733
worst      -0.356     -0.868     -1.514        63.536
best       -0.077     -0.619     -1.008        74.200
```

```
      sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
seed                                               
0        -0.077     -0.663     -1.052        74.200
1        -0.152     -0.619     -1.008        65.733
2        -0.356     -0.868     -1.514        63.536
```

## Cost stress (Sharpe, OOS)

```
                         1x     2x     3x
rl_ensemble          -0.013 -0.374 -0.783
equal_weight_monthly  0.213  0.202  0.191
buy_and_hold          0.282  0.277  0.272
gbm_linear_monthly    0.322  0.279  0.236
gbm_mean_variance     0.422  0.351  0.279
gbm_linear_band       0.644  0.505  0.367
rl_median_seed       -0.152 -0.663 -1.052
rl_worst_seed        -0.356 -0.868 -1.514
```

## Regimes (Hurst/entropy classifier, 2x costs)

```
                                days  sharpe_2x  ann_ret_2x
strategy             regime                                
rl_ensemble          calm        448      0.099       0.010
                     turbulent   458     -0.766      -0.095
gbm_linear_band      calm        448      1.387       0.152
                     turbulent   458     -0.292      -0.034
gbm_linear_monthly   calm        448      1.298       0.125
                     turbulent   458     -0.528      -0.062
equal_weight_monthly calm        448      0.574       0.059
                     turbulent   458     -0.112      -0.013
```

## Deflated Sharpe (RL ensemble, 1x)

- trials (grid runs) on this data hash (MLflow): 8 (88 seed runs)
- expected max Sharpe under the null (annualised): 0.911
- PSR(SR>0) 0.490 | **DSR 0.040**

## Behaviour audit

- avg_gross: 0.760
- std_gross: 0.149
- beta_to_ew: 0.824
- corr_to_ew: 0.926
- avg_herfindahl: 0.090
- eff_n_assets: 11.736
- flags: ['returns >0.9 correlated with equal weight: little cross-sectional skill']

![equity](equity.png)

![equity 2x](equity_2x.png)

![behaviour](behaviour.png)
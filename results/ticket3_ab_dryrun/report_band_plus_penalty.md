# Walk-forward report: band_plus_penalty

Grid `20261022-195605_band_plus_penalty_c308beb14b` | data hash `41567179a4` | source `synthetic_pit` | point-in-time: **True** | folds 4 x seeds 3 | 150000 steps/seed | reward `turnover` (kappa 0.002) | band 0.03

## Verdict

**DRY RUN (synthetic_pit data, not a real-data verdict): would read as -> FAIL - median-seed Sharpe at 2x costs -0.71 <= GBM+no-trade-band 0.51 (ensemble -0.32, worst seed -0.90). Ship the GBM+no-trade-band rule; archive the RL allocator as research.**

## Strategies, out of sample, 1x costs

```
                      sharpe   cagr  ann_vol  max_dd  turnover_ann  cost_drag_ann  avg_gross  beta_to_bench
rl_ensemble            0.043 -0.002    0.118   0.351        55.047          0.041      0.770          0.840
equal_weight_monthly   0.213  0.018    0.112   0.327         2.481          0.001      0.773          0.822
buy_and_hold           0.282  0.018    0.074   0.205         0.612          0.000      0.492          0.560
gbm_linear_monthly     0.322  0.029    0.108   0.264         6.446          0.005      0.691          0.757
gbm_mean_variance      0.422  0.033    0.085   0.181         5.792          0.006      0.448          0.538
gbm_linear_band        0.644  0.069    0.114   0.310        20.276          0.016      0.756          0.776
```

## RL seeds (stitched OOS): median / worst / best

```
        sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
median     -0.091     -0.712     -1.122        69.636
worst      -0.291     -0.896     -1.309        60.512
best       -0.057     -0.439     -1.095        75.139
```

```
      sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
seed                                               
0        -0.057     -0.712     -1.095        75.139
1        -0.091     -0.439     -1.309        69.636
2        -0.291     -0.896     -1.122        60.512
```

## Cost stress (Sharpe, OOS)

```
                         1x     2x     3x
rl_ensemble           0.043 -0.318 -0.690
equal_weight_monthly  0.213  0.202  0.191
buy_and_hold          0.282  0.277  0.272
gbm_linear_monthly    0.322  0.279  0.236
gbm_mean_variance     0.422  0.351  0.279
gbm_linear_band       0.644  0.505  0.367
rl_median_seed       -0.091 -0.712 -1.122
rl_worst_seed        -0.291 -0.896 -1.309
```

## Regimes (Hurst/entropy classifier, 2x costs)

```
                                days  sharpe_2x  ann_ret_2x
strategy             regime                                
rl_ensemble          calm        448      0.137       0.015
                     turbulent   458     -0.702      -0.087
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
- PSR(SR>0) 0.533 | **DSR 0.050**

## Behaviour audit

- avg_gross: 0.770
- std_gross: 0.151
- beta_to_ew: 0.840
- corr_to_ew: 0.927
- avg_herfindahl: 0.091
- eff_n_assets: 11.581
- flags: ['returns >0.9 correlated with equal weight: little cross-sectional skill']

![equity](equity.png)

![equity 2x](equity_2x.png)

![behaviour](behaviour.png)
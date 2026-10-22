# Walk-forward report: cvar_reward

Grid `20261022-193706_cvar_reward_f65beb3a0c` | data hash `41567179a4` | source `synthetic_pit` | point-in-time: **True** | folds 4 x seeds 3 | 150000 steps/seed | reward `cvar` (kappa 0) | band learned

## Verdict

**DRY RUN (synthetic_pit data, not a real-data verdict): would read as -> FAIL - median-seed Sharpe at 2x costs -1.47 <= GBM+no-trade-band 0.51 (ensemble -1.33, worst seed -1.50). Ship the GBM+no-trade-band rule; archive the RL allocator as research.**

## Strategies, out of sample, 1x costs

```
                      sharpe   cagr  ann_vol  max_dd  turnover_ann  cost_drag_ann  avg_gross  beta_to_bench
rl_ensemble           -0.433 -0.053    0.111   0.415        69.925          0.051      0.762          0.776
equal_weight_monthly   0.213  0.018    0.112   0.327         2.481          0.001      0.773          0.822
buy_and_hold           0.282  0.018    0.074   0.205         0.612          0.000      0.492          0.560
gbm_linear_monthly     0.322  0.029    0.108   0.264         6.446          0.005      0.691          0.757
gbm_mean_variance      0.422  0.033    0.085   0.181         5.792          0.006      0.448          0.538
gbm_linear_band        0.644  0.069    0.114   0.310        20.276          0.016      0.756          0.776
```

## RL seeds (stitched OOS): median / worst / best

```
        sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
median     -0.756     -1.470     -1.779        78.124
worst      -1.107     -1.497     -2.127        73.723
best       -0.543     -0.985     -1.520        84.856
```

```
      sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
seed                                               
0        -0.756     -1.470     -2.127        84.856
1        -1.107     -1.497     -1.779        78.124
2        -0.543     -0.985     -1.520        73.723
```

## Cost stress (Sharpe, OOS)

```
                         1x     2x     3x
rl_ensemble          -0.433 -1.328 -1.376
equal_weight_monthly  0.213  0.202  0.191
buy_and_hold          0.282  0.277  0.272
gbm_linear_monthly    0.322  0.279  0.236
gbm_mean_variance     0.422  0.351  0.279
gbm_linear_band       0.644  0.505  0.367
rl_median_seed       -0.756 -1.470 -1.779
rl_worst_seed        -1.107 -1.497 -2.127
```

## Regimes (Hurst/entropy classifier, 2x costs)

```
                                days  sharpe_2x  ann_ret_2x
strategy             regime                                
rl_ensemble          calm        448     -1.257      -0.125
                     turbulent   458     -1.392      -0.156
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
- PSR(SR>0) 0.206 | **DSR 0.005**

## Behaviour audit

- avg_gross: 0.762
- std_gross: 0.204
- beta_to_ew: 0.776
- corr_to_ew: 0.906
- avg_herfindahl: 0.095
- eff_n_assets: 10.867
- flags: ['returns >0.9 correlated with equal weight: little cross-sectional skill']

![equity](equity.png)

![equity 2x](equity_2x.png)

![behaviour](behaviour.png)
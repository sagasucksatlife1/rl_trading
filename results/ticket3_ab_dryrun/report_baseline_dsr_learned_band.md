# Walk-forward report: baseline_dsr_learned_band

Grid `20261022-183537_baseline_dsr_learned_band_c4ce345f60` | data hash `41567179a4` | source `synthetic_pit` | point-in-time: **True** | folds 4 x seeds 3 | 150000 steps/seed | reward `dsr` (kappa 0) | band learned

## Verdict

**DRY RUN (synthetic_pit data, not a real-data verdict): would read as -> FAIL - median-seed Sharpe at 2x costs -1.29 <= GBM+no-trade-band 0.51 (ensemble -0.92, worst seed -1.42). Ship the GBM+no-trade-band rule; archive the RL allocator as research.**

## Strategies, out of sample, 1x costs

```
                      sharpe   cagr  ann_vol  max_dd  turnover_ann  cost_drag_ann  avg_gross  beta_to_bench
rl_ensemble           -0.316 -0.041    0.114   0.402        70.153          0.051      0.791          0.805
equal_weight_monthly   0.213  0.018    0.112   0.327         2.481          0.001      0.773          0.822
buy_and_hold           0.282  0.018    0.074   0.205         0.612          0.000      0.492          0.560
gbm_linear_monthly     0.322  0.029    0.108   0.264         6.446          0.005      0.691          0.757
gbm_mean_variance      0.422  0.033    0.085   0.181         5.792          0.006      0.448          0.538
gbm_linear_band        0.644  0.069    0.114   0.310        20.276          0.016      0.756          0.776
```

## RL seeds (stitched OOS): median / worst / best

```
        sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
median     -0.561     -1.287     -1.839        78.572
worst      -1.108     -1.417     -1.897        77.810
best       -0.443     -1.028     -1.331        79.297
```

```
      sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
seed                                               
0        -0.561     -1.417     -1.839        79.297
1        -1.108     -1.287     -1.897        78.572
2        -0.443     -1.028     -1.331        77.810
```

## Cost stress (Sharpe, OOS)

```
                         1x     2x     3x
rl_ensemble          -0.316 -0.922 -1.457
equal_weight_monthly  0.213  0.202  0.191
buy_and_hold          0.282  0.277  0.272
gbm_linear_monthly    0.322  0.279  0.236
gbm_mean_variance     0.422  0.351  0.279
gbm_linear_band       0.644  0.505  0.367
rl_median_seed       -0.561 -1.287 -1.839
rl_worst_seed        -1.108 -1.417 -1.897
```

## Regimes (Hurst/entropy classifier, 2x costs)

```
                                days  sharpe_2x  ann_ret_2x
strategy             regime                                
rl_ensemble          calm        448     -0.765      -0.079
                     turbulent   458     -1.061      -0.124
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
- PSR(SR>0) 0.275 | **DSR 0.010**

## Behaviour audit

- avg_gross: 0.791
- std_gross: 0.183
- beta_to_ew: 0.805
- corr_to_ew: 0.919
- avg_herfindahl: 0.093
- eff_n_assets: 11.294
- flags: ['returns >0.9 correlated with equal weight: little cross-sectional skill']

![equity](equity.png)

![equity 2x](equity_2x.png)

![behaviour](behaviour.png)
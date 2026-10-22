# Walk-forward report: turnover_penalty

Grid `20261022-191545_turnover_penalty_2605ced0d5` | data hash `41567179a4` | source `synthetic_pit` | point-in-time: **True** | folds 4 x seeds 3 | 150000 steps/seed | reward `turnover` (kappa 0.002) | band learned

## Verdict

**DRY RUN (synthetic_pit data, not a real-data verdict): would read as -> FAIL - median-seed Sharpe at 2x costs -1.39 <= GBM+no-trade-band 0.51 (ensemble -1.19, worst seed -1.43). Ship the GBM+no-trade-band rule; archive the RL allocator as research.**

## Strategies, out of sample, 1x costs

```
                      sharpe   cagr  ann_vol  max_dd  turnover_ann  cost_drag_ann  avg_gross  beta_to_bench
rl_ensemble           -0.722 -0.082    0.111   0.491        64.438          0.046      0.760          0.755
equal_weight_monthly   0.213  0.018    0.112   0.327         2.481          0.001      0.773          0.822
buy_and_hold           0.282  0.018    0.074   0.205         0.612          0.000      0.492          0.560
gbm_linear_monthly     0.322  0.029    0.108   0.264         6.446          0.005      0.691          0.757
gbm_mean_variance      0.422  0.033    0.085   0.181         5.792          0.006      0.448          0.538
gbm_linear_band        0.644  0.069    0.114   0.310        20.276          0.016      0.756          0.776
```

## RL seeds (stitched OOS): median / worst / best

```
        sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
median     -0.748     -1.393     -1.857        77.242
worst      -1.002     -1.427     -1.882        68.699
best       -0.615     -1.238     -1.742        79.728
```

```
      sharpe_1x  sharpe_2x  sharpe_3x  turnover_ann
seed                                               
0        -0.615     -1.427     -1.882        79.728
1        -1.002     -1.393     -1.857        77.242
2        -0.748     -1.238     -1.742        68.699
```

## Cost stress (Sharpe, OOS)

```
                         1x     2x     3x
rl_ensemble          -0.722 -1.186 -1.252
equal_weight_monthly  0.213  0.202  0.191
buy_and_hold          0.282  0.277  0.272
gbm_linear_monthly    0.322  0.279  0.236
gbm_mean_variance     0.422  0.351  0.279
gbm_linear_band       0.644  0.505  0.367
rl_median_seed       -0.748 -1.393 -1.857
rl_worst_seed        -1.002 -1.427 -1.882
```

## Regimes (Hurst/entropy classifier, 2x costs)

```
                                days  sharpe_2x  ann_ret_2x
strategy             regime                                
rl_ensemble          calm        448     -0.907      -0.092
                     turbulent   458     -1.427      -0.167
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
- PSR(SR>0) 0.085 | **DSR 0.001**

## Behaviour audit

- avg_gross: 0.760
- std_gross: 0.234
- beta_to_ew: 0.755
- corr_to_ew: 0.885
- avg_herfindahl: 0.097
- eff_n_assets: 10.571
- flags: []

![equity](equity.png)

![equity 2x](equity_2x.png)

![behaviour](behaviour.png)
# Ticket 3: turnover fixes, A/B

> DRY RUN on synthetic / non-PIT data: this tests the fix mechanisms, it is NOT a verdict. Rerun on a vendor snapshot.

Sharpe ratios out of sample. Gate = median seed at 2x costs must beat `gbm_band_2x` (GBM + no-trade band).

```
                           rl_turnover_ann  rl_ens_1x  rl_ens_2x  rl_median_seed_2x  rl_worst_seed_2x  1/N_2x  gbm_linear_2x  gbm_mv_2x  gbm_band_2x    dsr            gate
variant                                                                                                                                                                    
baseline_dsr_learned_band           70.153     -0.316     -0.922             -1.287            -1.417   0.202          0.279      0.351        0.505  0.010  FAIL (dry run)
band_fixed                          55.262     -0.013     -0.374             -0.663            -0.868   0.202          0.279      0.351        0.505  0.040  FAIL (dry run)
turnover_penalty                    64.438     -0.722     -1.186             -1.393            -1.427   0.202          0.279      0.351        0.505  0.001  FAIL (dry run)
cvar_reward                         69.925     -0.433     -1.328             -1.470            -1.497   0.202          0.279      0.351        0.505  0.005  FAIL (dry run)
band_plus_penalty                   55.047      0.043     -0.318             -0.712            -0.896   0.202          0.279      0.351        0.505  0.050  FAIL (dry run)
```

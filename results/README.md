# Results

All runs are on the synthetic point-in-time market (data hash `41567179a4`), so they are dry
runs. They are still the evidence behind the kill decision; see `research/rl_allocator/README.md`.

| Folder | What it is | Outcome |
|---|---|---|
| `ticket3_ab_dryrun/` | 5 turnover-fix variants for the RL allocator, 3 seeds × 150k × last 4 folds | Only the no-trade band moves turnover. All fail the gate. |
| `ticket4_dryrun/` | RL allocator, band 0.08 (chosen on validation), 3 seeds × 150k × all 7 folds | Median seed 0.54 vs GBM + band 0.82 at 2× costs, same ~21×/yr turnover. DSR 0.004 after 9 trials. **Kill.** |
| `strategy_backtest/` | Shipped GBM + band rule vs alternatives, 9 walk-forward folds, 1×/2×/3× costs | Highest Sharpe of the five rules at 1× and 2×. Survives 3× costs (1.04). |

The two backtests cover different out-of-sample periods (7 vs 9 folds), so compare
strategies within a table, not across tables. The DSR of 1.0 in `strategy_backtest/` only
reflects the edge planted in the synthetic market; it says nothing about real markets.

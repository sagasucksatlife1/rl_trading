# Results (all DRY RUNS on synthetic point-in-time data)

None of these is a real-data verdict. They test the machinery and the turnover-fix
mechanisms. The Ticket 4 decision run (5 seeds x 600k x all folds) happens once, on a
vendor snapshot (`scripts/snapshot.py build --source sharadar ...`).

- `ticket3_ab_dryrun/`: 5 turnover-fix variants, 3 seeds x 150k x last 4 folds. All fail the gate.
  Only the no-trade band moves turnover; a validation-only sweep picked b = 0.08.
- `ticket4_dryrun/`: band 0.08, 3 seeds x 150k x all 7 folds. Turnover fixed (~21x/yr per seed,
  below GBM+band's 21x), but median-seed Sharpe at 2x costs 0.54 < GBM+no-trade-band 0.82.
  Dry-run answer: ship GBM+no-trade-band, archive the RL allocator as research. DSR 0.004 after 9 trials.

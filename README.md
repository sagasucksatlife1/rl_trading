# Project 24: GBM + no-trade band (RL allocator archived)

**Verdict.** The project set out to use RL as the decision engine on top of a GBM forecast.
It ships the **GBM cross-sectional score with a no-trade band and a hard risk overlay**, and
archives the RL allocator as research.

In the walk-forward evaluation, the RL allocator's median seed made a Sharpe of 0.54 at 2×
costs, against 0.82 for the GBM + no-trade-band rule, at the same ~21×/yr turnover. Its
deflated Sharpe was 0.004 after 9 trials, and the gap traces to decision quality in calm
regimes, not turnover. Full write-up: [`research/rl_allocator/README.md`](research/rl_allocator/README.md).
Evidence: [`results/`](results/).

> All evaluation so far is on a synthetic point-in-time market (a dry run). The decision is
> taken on that evidence because the gap and its mechanism are too clear for real data to
> plausibly reverse it. The data pipeline is ready for a vendor snapshot if a real-data
> confirmation is wanted.

## What ships (`rltrade/`)

| Layer | What runs | Where |
|---|---|---|
| Data | Point-in-time universe, corporate actions, delistings, hashed snapshots | `rltrade/data/loaders.py`, `scripts/snapshot.py` |
| Forecast | Walk-forward LightGBM cross-sectional score (purged, embargoed) | `rltrade/features/signal.py` |
| Decision | Long the top half by score, weights ∝ excess score, rebalance every 5 days, no-trade band 2% | `rltrade/strategy.py` |
| Execution | Overlay-projected target, then band; forced exits for names leaving the universe | `rltrade/envs/portfolio.py::execute_target` |
| Risk | Hard overlay: weight cap, gross/net, vol target, turnover cap, drawdown kill switch | `rltrade/risk/overlay.py` |
| Backtest | Same engine as the research (accounting reconciled to the cent), costs at 1×/2×/3×, regimes | `scripts/backtest_strategy.py` |
| Live | Shadow / paper / Alpaca daily loop, decision log, PSI/KS drift, tracking error, daily report | `rltrade/live/`, `scripts/shadow_run.py` |

## Quick start

```bash
pip install -r requirements.txt
make test        # 34 tests: look-ahead, accounting, PIT adjustment, overlay, band order, env (+ archived research)
make backtest    # walk-forward OOS backtest of the shipped rule vs alternatives -> results/strategy_backtest/
make fit         # fit the deployable state (final GBM, regime HMM, normaliser) -> deploy/current
make shadow      # daily loop in shadow mode: logs targets, sends no orders (run 4-8 weeks first)
make paper       # same loop against the local paper broker
```

On real data, build a snapshot first:
`python scripts/snapshot.py build --source sharadar --dir vendor/sharadar`. Then pass
`data.source=snapshot data.snapshot=<hash>` to `scripts/backtest_strategy.py` and
`scripts/fit_strategy.py`. Every report built on synthetic data is labelled DRY RUN.

## Repository layout

```
rltrade/                  shipped stack (data, features, strategy, engine, risk, eval, live)
scripts/                  snapshot.py, backtest_strategy.py, fit_strategy.py, shadow_run.py
configs/strategy.yaml     the shipped strategy's config
tests/                    tests for the shipped stack
results/                  evidence: Ticket 3 A/B, Ticket 4 dry run, strategy backtest
research/rl_allocator/    ARCHIVED: RL agents, experiment grid, frontier envs, offline RL, write-up
```

## Point-in-time data (Ticket 1)

`rltrade/data/loaders.py` turns three **unadjusted**, vendor-agnostic tables into an adjusted, point-in-time `Market`:

| Table | Columns | Meaning |
|---|---|---|
| prices | date, symbol, open, high, low, close, volume | every name that was ever in the universe, dead ones included |
| actions | date, symbol, type, value, ref_symbol | `split` (ratio), `dividend` (cash/share), `delist` (delisting return), `cash_merger` (cash/share), `stock_merger` (acquirer shares/share) |
| membership | symbol, start, end | dated universe membership |

- **Total-return adjustment:** each day's return is R = split × (close + dividend) / previous close − 1. On a delisting or merger day, R is the delisting return, and after that the position is held flat as cash.
- **Volume:** volume becomes dollar volume divided by the adjusted price, so the impact model's q/ADV stays correct.
- **Tradable at date t** means a member at t *and* listed at t. A name that delists tomorrow is still tradable today and takes the delisting return. The env force-sells non-members at the next open, and that overrides the overlay's turnover cap. Features, the GBM labels, the regime model and the baselines all use the same mask.
- **Snapshots:** `data/snapshots/<hash>/{prices,actions,membership}.parquet + manifest.json`. The hash is SHA-256 over the canonicalised tables plus the adjustment-code version. It is re-checked on load, and a tampered file refuses to load. Every run records the hash.
- **Tests** (`tests/test_pit.py`):
  - total returns match ground truth to 1e-15 across splits and dividends;
  - delisting and merger returns are exact, and the position goes flat afterwards;
  - membership is respected, and the env never holds a non-member;
  - snapshot hashes are stable and changes are detected;
  - a deleted split is flagged by `validate()`.

**Getting real survivorship-free data.** No free source covers this: it needs a vendor. For the US, use Sharadar (Nasdaq Data Link: SEP + ACTIONS + SP500, adapter included), Norgate or CRSP. For India, use NSE index-change circulars plus a delisted-securities price source. Then:
```bash
python scripts/snapshot.py build --source sharadar --dir vendor/sharadar      # or --source csv_pit --dir <generic csvs>
python scripts/backtest_strategy.py --config configs/strategy.yaml data.source=snapshot data.snapshot=<hash>
```
`configs/pit.yaml` defaults to `synthetic_pit`, a fixture with injected splits, dividends, delistings, mergers and index changes. The yfinance adapter still exists, but its snapshots are flagged `pit: false` and every report carries a warning.

## Production path

```
backtest on a vendor snapshot  ->  shadow mode 4-8 weeks  ->  paper (Alpaca / Kite)  ->  small capital, kill switch armed
```

`scripts/shadow_run.py --deploy deploy/current --broker shadow|paper|alpaca` runs the daily loop:
1. Rebuild features with the frozen feature state (final GBM, regime HMM, normaliser from `scripts/fit_strategy.py`).
2. Replay the last 126 days for a simulated track, used for live-vs-backtest tracking error.
3. Read **actual broker positions**. A new all-cash account enters on day 1; after that the rule rebalances every 5th trading day and holds in between.
4. Strategy target, then overlay projection, then no-trade band (the kill switch overrides the band; kill-switch state persists between days). The hard overlay applies every day, rebalance or not.
5. Send orders. Alpaca uses market-on-open, which matches the backtest's fill assumption.
6. Append to the decision log, and fill in yesterday's realised return.
7. Check PSI/KS feature drift and live-vs-backtest tracking error, then write a daily report.

`--replay-days N` runs the loop across history to test it end to end.

India: `KiteBroker` is a deliberate stub. Check SEBI's current framework for retail algorithmic orders before any live order.

## Data note

`data.source: synthetic_pit` is a test fixture, not a market. yfinance snapshots are flagged
`pit: false` (current tickers only, so not survivorship-free), and reports say so. Real
decisions need a vendor feed: Sharadar, Norgate or CRSP for the US, or NSE index-change
circulars plus delisted-securities prices for India.

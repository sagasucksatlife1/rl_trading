# Project 24: RL for Trading

**The thesis:** RL is a decision engine, not an alpha engine. Supervised models make the forecast, RL decides position size and trade timing, RL executes, and a hard risk overlay that the network cannot override constrains everything.

This repo implements the build at Level 4 ("Professional") and adds Level 5 research pieces:

| Layer | What runs | Where |
|---|---|---|
| Forecast | Walk-forward LightGBM cross-sectional score (purged, embargoed) | `rltrade/features/signal.py` |
| Regime | Gaussian HMM written from scratch, filtered (causal) P(turbulent) | `rltrade/features/regime.py` |
| Decision | **PPO with a permutation-equivariant per-asset policy** (main); RecurrentPPO, SAC (challenger), TD3 (check) | `rltrade/agents/policies.py`, `rltrade/agents/sb3_agents.py`, `rltrade/envs/portfolio.py` |
| Baselines (from scratch) | Tabular Q-learning; DQN with Double, Dueling and PER | `rltrade/agents/tabular_q.py`, `rltrade/agents/dqn.py` |
| Execution | RL order slicer vs TWAP and Almgren-Chriss | `rltrade/envs/execution.py` |
| Market making | RL quoter in a simulated LOB with adverse selection vs Avellaneda-Stoikov | `rltrade/envs/market_making.py` |
| Deep hedging | RL hedger under costs vs Black-Scholes delta, judged on CVaR | `rltrade/envs/hedging.py` |
| Risk | Hard overlay: weight cap, gross/net, vol target, turnover cap, drawdown kill switch | `rltrade/risk/overlay.py` |
| Proof | Purged walk-forward, ≥5 seeds, ensemble, deflated Sharpe, cost stress, regime split, behaviour audit | `rltrade/eval/`, `rltrade/experiment.py` |
| Production | Shadow, paper and Alpaca brokers; decision log; PSI/KS drift; tracking error; daily report | `rltrade/live/` |
| Offline RL | CQL/IQL (d3rlpy) on the decision log; behaviour cloning as the fallback | `rltrade/live/offline.py` |

## Quick start

```bash
pip install -r requirements.txt
make test        # 23 unit tests: look-ahead, accounting, costs, rewards, overlay, DSR, splits, env APIs
make sanity      # planted-signal / pure-noise / accounting / from-scratch agents
make quick       # end-to-end run in minutes: data -> GBM -> PPO seeds -> ensemble -> tear sheet
make train       # main experiment: shared-policy PPO, 5 seeds, walk-forward (overnight on a laptop)
make real        # same on the 50-name multi-asset universe via yfinance
make frontier    # execution, market making and deep hedging vs their classical baselines
```

Each run writes `runs/<timestamp>_<algo>_<cfghash>/` with the following files:
- `tearsheet.md`, `equity.png`, `behaviour.png`: the verdict, agent vs baselines, DSR, cost stress, regime breakdown and behaviour audit
- `oos_*.csv`: stitched out-of-sample daily records for every strategy
- `fold_k/seed_s/{model.zip, obs_rms.npz, meta.json, history.json}`: best checkpoint per seed, selected on validation
- `fold_k/feature_state.pkl`: frozen normaliser, GBM model and feature list for deployment
- `config.yaml`: the exact config plus the data hash. Every run can be reproduced from config, seed and data hash.
- `runs/trials.jsonl`: every trial ever run, so the deflated-Sharpe trial count stays honest

Overrides use dotted keys: `python scripts/train.py --config configs/default.yaml agent.algo=sac reward.kind=risk reward.risk=drawdown`

## Environment design (doc §4)

**Timing (the anti-leak rule).** The decision at row *t* sees features up to close *t*, with portfolio weights marked to close *t*. The trade executes at **open t+1** against the weights held just before that open. The reward is the net return from open t+1 to open t+2. `tests/test_lookahead.py` changes the future and asserts that features, the GBM score, HMM probabilities and env observations from the past do not change.

**State:** per-asset log returns over 1/5/21/63 days, 12-1 momentum, realised vol (21d and 63d), vol-of-vol, volume surprise, a variance-ratio Hurst estimate, distance from the 252-day high, and the GBM score, rank and confidence (trailing realised IC, lagged so it is causal). Global features: market return and vol, vol ratio, average pairwise correlation, dispersion, and HMM P(turbulent). Portfolio features: current weights plus cash, drawdown, P&L since start, and days since the last trade. Everything is z-scored with statistics fitted on the training window only. Markets are a POMDP, so the shared policy reads a stacked window of the last 5 observations (`env.window`). `configs/recurrent.yaml` swaps in an LSTM (RecurrentPPO).

**Action:** continuous target weights. Long-only uses a softmax over the assets plus cash. Long/short uses tanh outputs scaled so that gross exposure ≤ L. A no-trade band (|target − current| < b → no trade) can be fixed, or **learned** as an extra action dimension (`env.learned_band`).

**Reward** (`rltrade/envs/rewards.py`) has four options: `log` (log(1+R_net)); `risk` (log return minus λ times a risk term: rolling variance, downside deviation or drawdown increment); `dsr` (the differential Sharpe ratio of Moody & Saffell, 1998); and `cvar`.

**Costs** (`rltrade/envs/costs.py`): commission, plus half the spread, plus square-root impact Y·σ·√(q/V), where σ and ADV are causal estimates. The cost multiplier is drawn per training episode from U[0.5, 3].

**Episodes:** random one-year windows, optional random asset subsets, random initial weights, and randomised costs.

## Why the policy is permutation-equivariant

A plain MLP over the flattened observation must learn "how good is this asset" separately for every asset slot, which gives it the capacity to memorise the single history. On the planted-signal market (5 assets, test Sharpe of 1/N = 2.54, oracle = 7.0):

| Policy | Train Sharpe | Test Sharpe |
|---|---|---|
| MLP PPO (base, γ=0.9, low init std) | 5.7 – 6.9 | 1.8 – 2.3 (loses to 1/N) |
| MLP PPO + asset shuffling | 2.1 | 2.8 |
| **Shared per-asset policy** | 4.7 | **3.3** |

The MLP's train Sharpe exceeds the oracle's: it memorised the path. The shared policy (`rltrade/agents/policies.py`) scores each asset with **one** network applied to that asset's own features plus shared context, mean-pools for the critic, the cash logit and the learned band, and is tested for exact permutation equivariance (`tests/test_policy.py`). That is the structural prior "assets are exchangeable": fewer parameters, far less room to memorise, and a policy that transfers to a reordered or resized universe.

## Validation protocol (doc §6)

The pass bar: the RL ensemble must beat **the GBM score with naive sizing** (`gbm_linear_monthly`), net of costs, out of sample, at the median seed, and it must keep a positive Sharpe at 2× costs. If it fails, the tear sheet says *"use the simple rule"*. That is a valid result.

- **Baselines** (`rltrade/baselines/rules.py`), all run through the same env with the same timing, costs and overlay: 1/N, buy-and-hold, GBM → linear weights, GBM → mean-variance with an L1 turnover penalty (proximal-gradient solver), and GBM + no-trade band.
- **Sanity tests** (`scripts/sanity.py`): on the planted-signal market the agent must beat 1/N significantly. On the pure-noise market the agent's excess return must not be significant; a significant result means a leak. The accounting test checks that the env's NAV matches an independent share-and-cash backtest to within a cent on 10M of capital.
- **Splits** (`rltrade/eval/splits.py`): chronological train/val/test with a purge and embargo between blocks, plus expanding or rolling walk-forward folds whose test blocks tile into one stitched out-of-sample track.
- **Statistics:** the distribution across seeds (median and worst, not best), the deflated Sharpe ratio using the trial count from `runs/trials.jsonl`, cost stress at 2× and 3×, the regime breakdown, and a behaviour audit that flags "always long", beta-only and mostly-flat policies.

## Production path (doc §7)

```
backtest passes §6  ->  shadow mode 4-8 weeks  ->  paper (Alpaca / Kite)  ->  small capital, kill switch armed
```

`scripts/shadow_run.py --model runs/<run>/fold_0 --broker shadow|paper|alpaca` runs the daily loop:
1. Rebuild features with the frozen feature state.
2. Replay the last 126 days to warm up the LSTM state.
3. Build the observation from **actual broker positions**.
4. Apply the policy, then the band, then the overlay (kill-switch state persists between days).
5. Send orders. Alpaca uses market-on-open, which matches the backtest's fill assumption.
6. Append to the decision log, and fill in yesterday's reward.
7. Check PSI/KS feature drift and live-vs-backtest tracking error, then write a daily report.

`--replay-days N` runs the loop across history to test it end to end.

India: `KiteBroker` is a deliberate stub. Check SEBI's current framework for retail algorithmic orders before any live order.

## Frontier extensions (doc §8)

| Extension | Status |
|---|---|
| Market-making agent in a simulated LOB, with an Avellaneda-Stoikov baseline | implemented (`envs/market_making.py`) |
| Execution agent, with Almgren-Chriss and TWAP baselines | implemented (`envs/execution.py`) |
| Synthetic market generators (stationary block bootstrap, regime-switching GARCH) | implemented (`data/synthetic.py`) |
| Offline RL (CQL/IQL) from logged decisions | implemented (`live/offline.py`, needs `d3rlpy`) |
| Risk-sensitive RL (CVaR reward) | implemented (`reward.kind=cvar`) |
| Deep hedging, with a BS-delta baseline | implemented (`envs/hedging.py`) |
| Distributional RL (QR-DQN) | `sb3_contrib.QRDQN` works on `DiscretePositionEnv` |
| Hierarchical RL, meta-RL, LLM state features | not built; the regime feature and option-style hooks are the starting point |

## Results from the build run (synthetic data, CPU sandbox)

These numbers come from short training budgets. They validate the machinery. They do **not** say anything about real markets.

**Sanity suite** (`make sanity`)
- Accounting: env NAV vs independent share/cash backtest, max error **1.9e-8** on 10M capital. PASS.
- Planted signal: shared-policy PPO test Sharpe **3.41** vs 1/N 2.54, PSR(excess > 0) = 0.998. PASS.
- Pure noise: excess-return PSR = 0.02, so the agent cannot profit from a random walk and no leak is detected. PASS.
- From-scratch DQN: train return climbs while test Sharpe goes negative. That is overfitting to one path, in miniature.

**Quick end-to-end experiment** (`make quick`: 10 assets, 3 seeds × 60k steps, single split)
- Verdict: **FAIL, use the simple rule.** The RL ensemble's Sharpe was −0.07 vs the GBM baseline's −0.57, but the median seed was −0.38 and the edge did not survive 2× costs. RL turnover was about 76× per year (cost drag 4.6% per year), the same disease as the original long-only build. On this test block every strategy lost money, and the GBM's OOS IC was only 0.026.
- Next lever: the full budget (`make train`: 600k steps × 5 seeds, walk-forward), plus a turnover penalty or a wider learned band.

**Frontier agents** (`make frontier`, 150k–300k PPO steps)

| Task | Classical baseline | RL |
|---|---|---|
| Execution (IS, bps; lower is better) | Almgren-Chriss 72.6 / TWAP 72.1 | 77.5, with lower std (124 vs 132) and better p95 (294 vs 297) |
| Market making | Avellaneda-Stoikov: P&L 54.7, Sharpe-like 5.19, \|inv\| 1.07 | P&L 49.6, Sharpe-like 4.67, \|inv\| **0.82** |
| Deep hedging (CVaR 5%; higher is better) | BS delta −1.43 | −1.73 |

## Backtracking map (doc §9)

Read the build from the outside in.

| Peel | File | Fundamental |
|---|---|---|
| 1 | `scripts/train.py`, `configs/` | Gymnasium API, vectorised envs, SB3 workflow |
| 2 | `rltrade/eval/` | Backtest overfitting, purge/embargo, walk-forward, DSR |
| 3 | `rltrade/envs/costs.py`, `envs/execution.py` | Spread, square-root impact, Almgren-Chriss |
| 4 | `rltrade/envs/rewards.py` | Log utility, differential Sharpe, reward shaping |
| 5 | `agents/sb3_agents.py` hyperparameters | Clipped surrogate, GAE, entropy, max-entropy RL |
| 6 | SB3 actor/critic | Policy gradient theorem, actor-critic |
| 7 | `rltrade/agents/dqn.py` | Replay, target networks, Double DQN, dueling, PER |
| 8 | `rltrade/agents/tabular_q.py` | Bellman equations, TD learning, exploration |
| 9 | `rltrade/envs/portfolio.py` | MDP vs POMDP, and why markets break the Markov property |

## Data note

`data.source: yfinance` pulls the 50-name multi-asset universe (`rltrade/data/loader.py`) and caches it. yfinance is **not** point-in-time and **not** free of survivorship bias, so use a vendor feed for production and keep the `Market` interface. If the network is unavailable, the loader falls back to the regime-switching GARCH synthetic market, which has a weak planted cross-sectional edge. Results on synthetic data validate the pipeline. They say nothing about real markets.

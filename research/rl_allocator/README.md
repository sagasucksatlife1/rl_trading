# ARCHIVED RESEARCH: the RL allocator

**Status: archived. Not deployed.** The shipped strategy is the GBM + no-trade-band rule in
`rltrade/` (see the top-level README). The repo state before archiving is tagged
`rl-allocator-final`. Everything here still runs:
`python research/rl_allocator/scripts/<script>.py --config research/rl_allocator/configs/<cfg>.yaml`.

## The finding

**The RL allocator does not beat GBM + no-trade band at any cost level. The deflated Sharpe
says its best result is not distinguishable from luck. The failure is in decision quality,
not turnover.**

All numbers below are walk-forward, out of sample, on the synthetic point-in-time market
(`synthetic_pit`, data hash `41567179a4`): splits, dividends, delistings, mergers and index
changes, with a weak planted cross-sectional edge. There are 7 expanding folds with a 21-day
purge and 5-day embargo, 3 seeds × 150k PPO steps (permutation-equivariant policy), and a
no-trade band of 0.08 chosen on validation blocks only. Full reports are in
`results/ticket4_dryrun/` and `results/ticket3_ab_dryrun/`.

| | Sharpe 1× | Sharpe 2× | Sharpe 3× | Turnover/yr |
|---|---|---|---|---|
| RL median seed | 0.58 | **0.54** | 0.13 | 21.7× |
| RL worst seed | 0.32 | 0.27 | 0.04 | 18.7× |
| RL seed ensemble (best rule found) | 0.27 | 0.29 | 0.15 | 5.3× |
| **GBM + no-trade band** | 0.96 | **0.82** | 0.68 | 20.9× |
| GBM linear, monthly | 0.79 | 0.74 | 0.70 | 6.3× |
| 1/N, monthly | 0.63 | 0.62 | 0.61 | 2.2× |

1. **It is not significant.** The DSR is **0.004** after 9 logged trials. With that many
   looks, the expected best Sharpe from noise alone is 1.30 (annualised), and the RL median
   seed made 0.54.
2. **It is not a costs story.** Both turnover bugs were found and fixed:
   - the no-trade band was applied to raw targets before the risk overlay, so it never bit;
   - the band was too narrow.

   Turnover fell from 70–80× to about 21×/yr, the same as the baseline. At equal turnover
   the simple rule still wins by 0.28 Sharpe at 2× costs.
3. **The mechanism is decision quality.** By regime (Hurst/entropy classifier, 2× costs), RL
   scores 0.33 vs the rule's 1.16 in **calm** markets and 0.25 vs 0.33 in turbulent ones. The
   gap is largest exactly where a linear rule on a persistent cross-sectional signal is right
   by construction. That is a model that learned less than a linear map, which more training
   steps don't fix.
4. **The seeds disagree.** No rule for combining them beats the median seed:
   - mean-then-band: 0.29 at 2×;
   - per-asset median of banded targets: −0.10, at 32×/yr;
   - majority vote on executable targets: 0.09, at 2.4×/yr.

   The seeds don't agree on which assets to trade, so every combination either churns or
   freezes. A policy that had learned the signal would agree with itself across seeds.

The verdict is **ship GBM + no-trade band, archive the RL allocator.** We are not paying for a
full 5 × 600k run on vendor data. The gap is large, and its mechanism is clear enough that the
synthetic-vs-real difference is not a plausible way to flip it. The code to do that run exists
(`scripts/walkforward.py`, `configs/full.yaml`, `data.source=snapshot`) if a "confirmed on real
data" line is ever needed for credibility.

### What we learned along the way (reusable)
- **The band has to be applied to the executable (overlay-projected) target.** Applied to
  the raw target, every hard limit turns into a daily re-trade.
  `rltrade/envs/portfolio.py::execute_target` is the fix and is used by the shipped stack.
- **Ensembles must vote on executable targets too.** It's the same bug in a new place.
- **A plain MLP policy memorises the single history.** On the planted market it scored
  train Sharpe 5.7–6.9, above the oracle's 6.7, and test 1.8–2.3. The equivariant
  per-asset policy fixed the memorisation, but not the decision gap above.
- **Shaping turnover through the reward is weak at realistic κ.** At κ=0.002 the penalty is
  about 0.05 reward units/day against about 1 unit of return noise. CVaR shaping doesn't
  touch turnover at all.

## Timeline of the experiment
1. The smoke test (3 seeds × 60k) showed 76×/yr turnover, and the allocator failed 2× costs.
2. Ticket 3 A/B over 5 turnover fixes: only the band works.
3. A diagnostic showed the band never bit (band-before-overlay). Fixed, and the A/B was rerun.
4. A validation-only band sweep chose b = 0.08 (17.6×/yr on validation).
5. Ticket 4 dry run at b = 0.08 across all folds: the gate fails (0.54 vs 0.82 at 2×).
6. Ensemble repair attempts (3 rules): none beats the median seed.
7. Kill decision, and the code is archived here.

## Environment design (doc §4)

**Timing (the anti-leak rule).** The decision at row *t* sees features up to close *t*, with portfolio weights marked to close *t*. The trade executes at **open t+1** against the weights held just before that open. The reward is the net return from open t+1 to open t+2. `tests/test_lookahead.py` changes the future and asserts that features, the GBM score, HMM probabilities and env observations from the past do not change.

**State:** per-asset log returns over 1/5/21/63 days, 12-1 momentum, realised vol (21d and 63d), vol-of-vol, volume surprise, a variance-ratio Hurst estimate, distance from the 252-day high, and the GBM score, rank and confidence (trailing realised IC, lagged so it is causal). Global features: market return and vol, vol ratio, average pairwise correlation, dispersion, and HMM P(turbulent). Portfolio features: current weights plus cash, drawdown, P&L since start, and days since the last trade. Everything is z-scored with statistics fitted on the training window only. Markets are a POMDP, so the shared policy reads a stacked window of the last 5 observations (`env.window`). `research/rl_allocator/configs/recurrent.yaml` swaps in an LSTM (RecurrentPPO).

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

The MLP's train Sharpe exceeds the oracle's: it memorised the path. The shared policy (`research/rl_allocator/agents/policies.py`) scores each asset with **one** network applied to that asset's own features plus shared context, mean-pools for the critic, the cash logit and the learned band, and is tested for exact permutation equivariance (`research/rl_allocator/tests/test_policy.py`). That is the structural prior "assets are exchangeable": fewer parameters, far less room to memorise, and a policy that transfers to a reordered or resized universe.

## Validation protocol (doc §6)

The pass bar: the RL ensemble must beat **the GBM score with naive sizing** (`gbm_linear_monthly`), net of costs, out of sample, at the median seed, and it must keep a positive Sharpe at 2× costs. If it fails, the tear sheet says *"use the simple rule"*. That is a valid result.

- **Baselines** (`rltrade/baselines/rules.py`), all run through the same env with the same timing, costs and overlay: 1/N, buy-and-hold, GBM → linear weights, GBM → mean-variance with an L1 turnover penalty (proximal-gradient solver), and GBM + no-trade band.
- **Sanity tests** (`research/rl_allocator/scripts/sanity.py`): on the planted-signal market the agent must beat 1/N significantly. On the pure-noise market the agent's excess return must not be significant; a significant result means a leak. The accounting test checks that the env's NAV matches an independent share-and-cash backtest to within a cent on 10M of capital.
- **Splits** (`rltrade/eval/splits.py`): chronological train/val/test with a purge and embargo between blocks, plus expanding or rolling walk-forward folds whose test blocks tile into one stitched out-of-sample track.
- **Statistics:** the distribution across seeds (median and worst, not best), the deflated Sharpe ratio using the trial count from `runs/trials.jsonl`, cost stress at 2× and 3×, the regime breakdown, and a behaviour audit that flags "always long", beta-only and mostly-flat policies.

## Frontier extensions (doc §8)

| Extension | Status |
|---|---|
| Market-making agent in a simulated LOB, with an Avellaneda-Stoikov baseline | implemented (`envs/market_making.py` (research)) |
| Execution agent, with Almgren-Chriss and TWAP baselines | implemented (`envs/execution.py` (research)) |
| Synthetic market generators (stationary block bootstrap, regime-switching GARCH) | implemented (`data/synthetic.py`) |
| Offline RL (CQL/IQL) from logged decisions | implemented (`offline.py` (research), needs `d3rlpy`) |
| Risk-sensitive RL (CVaR reward) | implemented (`reward.kind=cvar`) |
| Deep hedging, with a BS-delta baseline | implemented (`envs/hedging.py` (research)) |
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

## The real experiment (Tickets 2–4)

```bash
python research/rl_allocator/scripts/ab_turnover.py --workers 4 agent.total_steps=150000 agent.seeds=[0,1,2] split.max_folds=4   # Ticket 3 screen
python research/rl_allocator/scripts/walkforward.py --config research/rl_allocator/configs/full.yaml --variant <winner> --workers 4                      # Ticket 4: 5 seeds x 600k x all folds
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```
- **Runs and logging** (`research/rl_allocator/eval/walkforward.py`): the (fold, seed) jobs run in a process pool. Every job is one MLflow run (`kind=seed_run`) carrying the full flattened config, seed, fold, data hash, config hash, validation curve and test metrics at 1×, 2× and 3× costs.
- **Report** (`research/rl_allocator/eval/report.py`): stitched out-of-sample results, the **median and worst seed** (the best seed is shown but never used), cost stress, a Hurst/entropy regime split (`rltrade/features/regime_he.py`, thresholds fitted on each fold's training rows), and a DSR whose N is **every distinct config ever logged on that data hash**.
- **Gate (Ticket 4):** the median seed's out-of-sample Sharpe at 2× costs must beat **GBM + no-trade band** at 2× costs. If it fails, the report says to ship that rule and archive the RL allocator.
- **Reward variants** (`reward.kind`): `dsr` (Sharpe-style), `turnover` (log return − κ·turnover), `cvar` (Rockafellar–Uryasev CVaR penalty). `reward.turnover_penalty` adds κ on top of any of them. The fixed no-trade band is `env.band` with `env.learned_band: false`.

## Backtracking map (doc §9)

Read the build from the outside in.

| Peel | File | Fundamental |
|---|---|---|
| 1 | `research/rl_allocator/scripts/train.py`, `configs/` | Gymnasium API, vectorised envs, SB3 workflow |
| 2 | `rltrade/eval/` | Backtest overfitting, purge/embargo, walk-forward, DSR |
| 3 | `rltrade/envs/costs.py`, `envs/execution.py` (research) | Spread, square-root impact, Almgren-Chriss |
| 4 | `rltrade/envs/rewards.py` | Log utility, differential Sharpe, reward shaping |
| 5 | `agents/sb3_agents.py` hyperparameters | Clipped surrogate, GAE, entropy, max-entropy RL |
| 6 | SB3 actor/critic | Policy gradient theorem, actor-critic |
| 7 | `research/rl_allocator/agents/dqn.py` | Replay, target networks, Double DQN, dueling, PER |
| 8 | `research/rl_allocator/agents/tabular_q.py` | Bellman equations, TD learning, exploration |
| 9 | `rltrade/envs/portfolio.py` | MDP vs POMDP, and why markets break the Markov property |

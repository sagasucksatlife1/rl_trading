"""Sanity tests to run before anything else.

  1. planted-signal env: the agent must find a known weak edge
  2. pure-noise env:     the agent must earn ~0 minus costs out of sample
  3. accounting:         env P&L == independent share-based backtest, to the cent
  4. from-scratch agents: tabular Q-learning and DQN on the single-asset planted env

    python scripts/sanity.py            # all
    python scripts/sanity.py --only accounting
"""
import numpy as np
from _common import ROOT, parse

from rltrade.config import cfg_get, load_config
from rltrade.eval.backtest import share_backtest
from rltrade.eval.dsr import psr
from rltrade.eval.metrics import sharpe
from rltrade.eval.rollout import rollout
from rltrade.eval.splits import chrono_split
from rltrade.pipeline import build_dataset, fold_envdata, make_env


def _prep(cfg):
    market, fs = build_dataset(cfg)
    fold = chrono_split(fs.first_valid(), len(market.dates), purge=21, embargo=5)
    data, _ = fold_envdata(market, fs, fold.train, cfg)
    return market, fold, data


def accounting():
    cfg = load_config(ROOT / "configs/quick.yaml")
    _, fold, data = _prep(cfg)
    env = make_env(data, fold.test, cfg, train=False, overlay=True)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    ex, navs, done = [], [], False
    while not done:
        obs, _, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        ex.append(info["weights"])
        navs.append(info["nav"])
        done = te or tr
    cap = env.c["nav0"]
    bt = share_backtest(env.d, env.assets, env.t0, np.array(ex), env.cost, capital=cap)
    err = float(np.max(np.abs(bt - np.array(navs) * cap)))
    ok = err < 0.01
    print(f"[accounting] max |env - share backtest| = {err:.2e} on capital {cap:,.0f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def rl_sanity(config: str, expect_edge: bool):
    from rltrade.agents.sb3_agents import SB3Policy, train_seed
    from rltrade.baselines.rules import EqualWeight

    cfg = load_config(ROOT / config)
    _, fold, data = _prep(cfg)
    out = ROOT / "runs" / "sanity" / config.split("/")[-1].replace(".yaml", "")
    r = train_seed(lambda s: make_env(data, fold.train, cfg, train=True, seed=s),
                   lambda: make_env(data, fold.val, cfg, train=False), cfg, 0, out)
    pol = SB3Policy.load(out)
    rec = rollout(make_env(data, fold.test, cfg, train=False), pol)
    ew = rollout(make_env(data, fold.test, cfg, train=False), EqualWeight(every=1))
    s_rl, s_ew = sharpe(rec["r_net"].to_numpy()), sharpe(ew["r_net"].to_numpy())
    excess = rec["r_net"].to_numpy() - ew["r_net"].to_numpy()
    p = psr(excess)
    if expect_edge:
        ok = s_rl > s_ew + 0.3 and p > 0.9
        print(f"[planted] RL Sharpe {s_rl:.2f} vs 1/N {s_ew:.2f}; PSR(excess>0)={p:.3f} -> {'PASS' if ok else 'FAIL'}")
    else:
        ok = p < 0.95
        print(f"[noise]   RL Sharpe {s_rl:.2f} vs 1/N {s_ew:.2f}; PSR(excess>0)={p:.3f} "
              f"(must not be significant) -> {'PASS' if ok else 'FAIL: LEAK?'}")
    return ok


def scratch_agents():
    from rltrade.agents.tabular_q import TabularQ
    from rltrade.envs.single_asset import DiscretePositionEnv, TabularDiscretizer

    cfg = load_config(ROOT / "configs/sanity_planted.yaml")
    _, fold, data = _prep(cfg)
    mk = lambda train, rows, seed=0: DiscretePositionEnv(data, rows, asset=0, train=train,
                                                         cfg={"episode_len": 252}, reward={"kind": "log"},
                                                         seed=seed)
    env = mk(True, fold.train)
    disc = TabularDiscretizer(env, features=("ret_21", "ret_63"))
    agent = TabularQ(disc.n_states, env.action_space.n, seed=0)
    agent.train(env, disc, episodes=400, log_every=100)
    test = mk(False, fold.test)

    def run(act):
        obs, _ = test.reset()
        rs, done = [], False
        while not done:
            obs, _, te, tr, info = test.step(act(obs))
            rs.append(info["r_net"])
            done = te or tr
        return sharpe(np.array(rs))

    s_q = run(lambda o: agent.act(disc(o)))
    s_long = run(lambda o: 2)
    print(f"[tabular-q] test Sharpe {s_q:.2f} vs always-long {s_long:.2f}")
    try:
        from rltrade.agents.dqn import DQNAgent
    except ImportError:
        print("[dqn] torch not installed; skipped")
        return True
    dqn = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=0)
    dqn.train(env, total_steps=40_000, log_every=10_000)
    s_d = run(lambda o: dqn.act(o))
    print(f"[dqn] test Sharpe {s_d:.2f} vs always-long {s_long:.2f}")
    print("[scratch] informational: these teach the mechanics; a DQN that wins in-sample and loses "
          "out of sample on one path is the overfitting failure mode this project is built to avoid")
    return True


def main():
    args, _ = parse(__doc__, extra=lambda ap: ap.add_argument("--only", default=None))
    results = {}
    if args.only in (None, "accounting"):
        results["accounting"] = accounting()
    if args.only in (None, "scratch"):
        results["scratch"] = scratch_agents()
    if args.only in (None, "planted"):
        results["planted"] = rl_sanity("configs/sanity_planted.yaml", True)
    if args.only in (None, "noise"):
        results["noise"] = rl_sanity("configs/sanity_noise.yaml", False)
    print("\nSANITY:", {k: ("ran" if k == "scratch" else "PASS" if v else "FAIL") for k, v in results.items()})


if __name__ == "__main__":
    main()

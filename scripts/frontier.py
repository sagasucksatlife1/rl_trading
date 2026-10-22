"""Frontier agents vs their classical baselines.

  execution      RL slicer vs TWAP and Almgren-Chriss (implementation shortfall, bps)
  market_making  RL quoter vs Avellaneda-Stoikov (P&L, inventory) in a simulated LOB
  hedging        deep hedging vs Black-Scholes delta under costs (CVaR of hedging error)

    python scripts/frontier.py --task execution --steps 200000
"""
import json

from _common import ROOT, parse


def _train(env_fn, steps, seed=0, algo="ppo"):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda i=i: env_fn(seed * 100 + i) for i in range(4 if algo == "ppo" else 1)])
    if algo == "ppo":
        m = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", n_steps=1024, batch_size=256,
                learning_rate=3e-4, gamma=1.0, ent_coef=0.0)
    else:
        m = SAC("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", gamma=1.0, learning_starts=2000)
    m.learn(steps)
    return lambda obs: m.predict(obs, deterministic=True)[0]


def main():
    args, _ = parse(__doc__, extra=lambda ap: (ap.add_argument("--task", default="all"),
                                              ap.add_argument("--steps", type=int, default=200_000)))
    res = {}
    if args.task in ("all", "execution"):
        from rltrade.envs.execution import TWAP, AlmgrenChriss, ExecutionEnv, evaluate_execution

        env = ExecutionEnv(seed=123)
        rl = _train(lambda s: ExecutionEnv(seed=s), args.steps)
        res["execution"] = {"twap": evaluate_execution(env, TWAP(env.p["N"])),
                            "almgren_chriss": evaluate_execution(env, AlmgrenChriss(env)),
                            "rl_ppo": evaluate_execution(env, rl)}
    if args.task in ("all", "market_making"):
        from rltrade.envs.market_making import AvellanedaStoikov, MarketMakingEnv, evaluate_mm

        env = MarketMakingEnv(seed=123)
        rl = _train(lambda s: MarketMakingEnv(seed=s), args.steps)
        res["market_making"] = {"avellaneda_stoikov": evaluate_mm(env, AvellanedaStoikov(env)),
                                "rl_ppo": evaluate_mm(env, rl)}
    if args.task in ("all", "hedging"):
        from rltrade.envs.hedging import BSDelta, HedgingEnv, evaluate_hedging

        env = HedgingEnv(seed=123)
        rl = _train(lambda s: HedgingEnv(seed=s), args.steps)
        res["hedging"] = {"bs_delta": evaluate_hedging(env, BSDelta()), "rl_ppo": evaluate_hedging(env, rl)}
    out = ROOT / "runs" / "frontier.json"
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **res}, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

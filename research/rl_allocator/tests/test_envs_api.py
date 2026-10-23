import numpy as np
from gymnasium.utils.env_checker import check_env

from research.rl_allocator.envs.execution import TWAP, AlmgrenChriss, ExecutionEnv, evaluate_execution
from research.rl_allocator.envs.hedging import BSDelta, HedgingEnv, evaluate_hedging
from research.rl_allocator.envs.market_making import AvellanedaStoikov, MarketMakingEnv, evaluate_mm
from research.rl_allocator.envs.single_asset import DiscretePositionEnv, TabularDiscretizer
from rltrade.pipeline import make_env


def test_portfolio_env_gym_api(prepared):
    cfg, _, _, fold, data = prepared
    check_env(make_env(data, fold.train, cfg, train=True, seed=0), skip_render_check=True)


def test_single_asset_and_discretizer(prepared):
    cfg, _, _, fold, data = prepared
    env = DiscretePositionEnv(data, fold.train, asset=0, seed=0)
    disc = TabularDiscretizer(env, ("ret_5", "gbm_score"))
    obs, _ = env.reset()
    for a in (0, 1, 2, 2, 1):
        obs, r, te, tr, info = env.step(a)
        s = disc(obs)
        assert 0 <= s < disc.n_states
    assert info["weights"][0] == 0.0


def test_execution_baselines():
    env = ExecutionEnv(seed=0, random_params=False)
    check_env(ExecutionEnv(seed=0), skip_render_check=True)
    twap = evaluate_execution(env, TWAP(env.p["N"]), n_episodes=200)
    ac = evaluate_execution(env, AlmgrenChriss(env), n_episodes=200)
    assert np.isfinite(twap["mean_is_bps"]) and np.isfinite(ac["mean_is_bps"])


def test_market_making_baseline():
    env = MarketMakingEnv(seed=0)
    check_env(MarketMakingEnv(seed=0), skip_render_check=True)
    res = evaluate_mm(env, AvellanedaStoikov(env), n_episodes=50)
    assert res["mean_pnl"] > 0  # AS earns the spread on average


def test_hedging_delta_beats_no_hedge():
    env = HedgingEnv(seed=0, cost=0.0, stoch_vol=False)
    delta = evaluate_hedging(env, BSDelta(), n_episodes=500)
    none = evaluate_hedging(env, lambda o: np.array([-1.0], dtype=np.float32), n_episodes=500)
    assert delta["std"] < 0.5 * none["std"]

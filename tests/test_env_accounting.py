import numpy as np

from rltrade.eval.backtest import share_backtest
from rltrade.pipeline import make_env


def _run(env, rng):
    obs, _ = env.reset()
    ex, navs, rets, done = [], [], [], False
    while not done:
        obs, r, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        ex.append(info["weights"])
        navs.append(info["nav"])
        rets.append(info["r_net"])
        done = te or tr
    return np.array(ex), np.array(navs), np.array(rets)


def test_reconciles_to_the_cent(prepared):
    cfg, _, _, fold, data = prepared
    for overlay in (False, True):
        env = make_env(data, fold.test, cfg, train=False, overlay=overlay)
        ex, navs, _ = _run(env, np.random.default_rng(1))
        cap = env.c["nav0"]
        bt = share_backtest(env.d, env.assets, env.t0, ex, env.cost, capital=cap)
        assert np.max(np.abs(bt - navs * cap)) < 0.01


def test_nav_equals_compounded_returns(prepared):
    cfg, _, _, fold, data = prepared
    env = make_env(data, fold.test, cfg, train=False, overlay=False)
    _, navs, rets = _run(env, np.random.default_rng(2))
    np.testing.assert_allclose(navs, np.cumprod(1 + rets), rtol=1e-10)


def test_hold_has_no_cost(prepared):
    cfg, _, _, fold, data = prepared
    env = make_env(data, fold.test, cfg, train=False, overlay=False)
    env.reset()
    _, _, _, _, info = env.step_weights(np.full(env.k, 1.0 / env.k))
    assert info["cost"] > 0
    _, _, _, _, info = env.step_weights(None)
    assert info["cost"] == 0 and info["turnover"] < 1e-12


def test_long_only_weights_valid(prepared):
    cfg, _, _, fold, data = prepared
    env = make_env(data, fold.train, cfg, train=True, seed=0)
    env.reset()
    for _ in range(50):
        _, _, te, tr, info = env.step(env.action_space.sample())
        w = info["weights"]
        assert (w >= -1e-12).all() and w.sum() <= 1 + 1e-9
        if te or tr:
            env.reset()

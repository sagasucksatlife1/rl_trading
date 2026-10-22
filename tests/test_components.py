import numpy as np
import pytest

from rltrade.envs.costs import CostModel, liquidity_inputs
from rltrade.envs.rewards import Reward
from rltrade.eval.dsr import deflated_sharpe, expected_max_sharpe, psr
from rltrade.eval.splits import chrono_split, walk_forward
from rltrade.risk.overlay import RiskOverlay


def test_cost_sqrt_impact_scaling():
    c = CostModel(0, 0, impact_coef=1.0)
    args = dict(nav=1e7, price=np.array([100.0]), adv=np.array([1e6]), sigma=np.array([0.02]))
    c1 = c.cost(np.array([0.01]), **args)
    c4 = c.cost(np.array([0.04]), **args)
    # cost = |dw| * Y sigma sqrt(q/V): 4x size -> 4 * 2 = 8x cost
    assert c4 / c1 == pytest.approx(8.0)
    assert c.scaled(2.0).cost(np.array([0.01]), **args) == pytest.approx(2 * c1)


def test_liquidity_inputs_causal():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), 0))
    vol = rng.uniform(1e5, 2e5, (300, 3))
    s1, a1 = liquidity_inputs(close, vol)
    close2, vol2 = close.copy(), vol.copy()
    close2[200:] *= 2
    vol2[200:] *= 3
    s2, a2 = liquidity_inputs(close2, vol2)
    np.testing.assert_allclose(s1[:200], s2[:200])
    np.testing.assert_allclose(a1[:200], a2[:200])


def test_dsr_reward_prefers_higher_sharpe():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, 2000)
    tot = {}
    for mu in (0.0, 0.001):
        rw = Reward(kind="dsr", eta=0.01)
        tot[mu] = sum(rw(mu + e) for e in noise)
    assert tot[0.001] > tot[0.0]


def test_log_reward_compounds():
    rw = Reward(kind="log", scale=1.0)
    rs = [0.1, -0.05, 0.02]
    assert sum(rw(r) for r in rs) == pytest.approx(np.log(np.prod([1 + r for r in rs])))


def test_overlay_limits():
    ov = RiskOverlay(max_weight=0.3, vol_target=None, max_turnover=None, dd_kill=None)
    w, flags = ov.apply(np.array([0.6, 0.3, 0.1]), np.zeros(3))
    assert w.max() <= 0.3 + 1e-12 and "weight_cap" in flags


def test_overlay_turnover_and_vol():
    ov = RiskOverlay(max_weight=1.0, vol_target=0.10, max_turnover=0.2, dd_kill=None)
    cov = np.eye(2) * (0.02**2)
    w, flags = ov.apply(np.array([0.5, 0.5]), np.zeros(2), cov)
    assert np.abs(w).sum() <= 0.2 + 1e-12
    assert "vol_target" in flags and "turnover_cap" in flags


def test_kill_switch_and_rearm():
    ov = RiskOverlay(dd_kill=0.1, cooldown=3, vol_target=None, max_turnover=None)
    ov.reset(1.0)
    ov.update_nav(1.2)
    ov.update_nav(1.0)  # 16.7% dd -> kill
    w, flags = ov.apply(np.array([0.2, 0.2]), np.array([0.2, 0.2]))
    assert flags.get("kill_switch") and np.allclose(w, 0)
    for _ in range(3):
        ov.update_nav(1.0)
    assert not ov.killed


def test_psr_dsr_behave():
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 1000)
    assert psr(good) > 0.95
    assert psr(rng.normal(0, 0.01, 1000)) < 0.95
    d1 = deflated_sharpe(good, 1)["dsr"]
    d100 = deflated_sharpe(good, 100)["dsr"]
    assert d100 < d1
    assert expected_max_sharpe(100, 1e-3) > expected_max_sharpe(10, 1e-3) > 0


def test_splits_purged_and_ordered():
    f = chrono_split(100, 3000, purge=21, embargo=5)
    assert f.train[1] + 26 <= f.val[0] and f.val[1] + 26 <= f.test[0] and f.test[1] == 3000
    folds = walk_forward(100, 3000, purge=21, embargo=5)
    assert len(folds) >= 3
    for a, b in zip(folds, folds[1:]):
        assert a.test[1] == b.test[0]  # OOS blocks tile without overlap
    for fo in folds:
        assert fo.train[1] + 26 <= fo.val[0] and fo.val[1] + 26 <= fo.test[0]


def test_turnover_penalty_variants():
    a = Reward(kind="turnover", turnover_penalty=0.002, scale=1.0)
    assert a(0.01, turnover=0.5) == pytest.approx(np.log1p(0.01) - 0.001)
    b = Reward(kind="log", turnover_penalty=0.002, scale=1.0)
    assert b(0.01, turnover=0.5) == pytest.approx(np.log1p(0.01) - 0.001)
    c0, c1 = Reward(kind="dsr"), Reward(kind="dsr", turnover_penalty=0.002)
    assert c1(0.01, turnover=1.0) < c0(0.01, turnover=1.0)


def test_cvar_reward_penalises_tail_only():
    rng = np.random.default_rng(0)
    r = Reward(kind="cvar", lam=1.0, scale=1.0, cvar_alpha=0.05)
    for x in rng.normal(0, 0.01, 100):
        r(x)
    small = r(-0.001)
    assert small == pytest.approx(np.log1p(-0.001))  # inside VaR: no penalty
    big = r(-0.08)
    assert big < np.log1p(-0.08) - 0.5  # tail loss: heavy penalty

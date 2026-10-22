"""The anti-leak rule: changing the future must not change any past feature."""
import numpy as np

from rltrade.data.synthetic import Market
from rltrade.features.library import build_features
from rltrade.features.regime import GaussianHMM, regime_observations
from rltrade.features.signal import walk_forward_signal


def _perturb_after(m: Market, t: int, seed: int = 9) -> Market:
    rng = np.random.default_rng(seed)
    o, c, v = m.open.copy(), m.close.copy(), m.volume.copy()
    shock = np.exp(rng.normal(0, 0.2, size=c.iloc[t + 1:].shape))
    c.iloc[t + 1:] *= shock
    o.iloc[t + 1:] *= shock
    v.iloc[t + 1:] *= np.exp(rng.normal(0, 0.5, size=v.iloc[t + 1:].shape))
    return Market(o, c, v)


def test_features_causal(small_market):
    t = 1000
    a = build_features(small_market)
    b = build_features(_perturb_after(small_market, t))
    np.testing.assert_allclose(a.asset[: t + 1], b.asset[: t + 1], equal_nan=True)
    np.testing.assert_allclose(a.glob[: t + 1], b.glob[: t + 1], equal_nan=True)
    assert not np.allclose(np.nan_to_num(a.asset[t + 5:]), np.nan_to_num(b.asset[t + 5:]))


def test_signal_causal(small_market):
    """GBM score at rows <= t uses labels known by t only (purged training)."""
    t = 1200
    pert = _perturb_after(small_market, t)
    out = []
    for m in (small_market, pert):
        fs = build_features(m)
        walk_forward_signal(fs, m.open, horizon=21, refit=126, min_train=504)
        out.append(fs.asset[:, :, fs.asset_names.index("gbm_score")])
    np.testing.assert_allclose(out[0][: t + 1], out[1][: t + 1], equal_nan=True, atol=1e-10)


def test_hmm_filter_causal(small_market):
    X = np.nan_to_num(regime_observations(small_market.close))[30:]
    hmm = GaussianHMM(2).fit(X[:600])
    t = 700
    p1 = hmm.filter(X)
    X2 = X.copy()
    X2[t + 1:] += 0.5
    p2 = hmm.filter(X2)
    np.testing.assert_allclose(p1[: t + 1], p2[: t + 1])


def test_env_observation_causal(prepared):
    """Obs at decision row t must not depend on prices after close t."""
    cfg, market, fs, fold, data = prepared
    from rltrade.pipeline import make_env

    env = make_env(data, fold.test, cfg, train=False)
    obs0, _ = env.reset()
    t = env.t
    d2 = type(data)(**{**data.__dict__})
    d2.open = data.open.copy()
    d2.close = data.close.copy()
    d2.open[t + 1:] *= 1.5
    d2.close[t + 1:] *= 1.5
    env2 = make_env(d2, fold.test, cfg, train=False)
    obs1, _ = env2.reset()
    np.testing.assert_allclose(obs0, obs1)

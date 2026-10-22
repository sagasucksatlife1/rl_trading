"""Baselines the RL agent must beat. All run through the SAME env (same timing,
costs, overlay), via `rollout`, so the comparison is apples to apples.

  EqualWeight     1/N, rebalanced every `every` days (1/N is hard to beat after costs)
  BuyAndHold      1/N once, never rebalanced
  SignalLinear    GBM score -> weights, monthly rebalance  (the real opponent)
  MeanVariance    GBM score -> mean-variance with an L1 turnover penalty
  SignalLinear(band=b) isolates whether RL's gain is just lower turnover
"""
from __future__ import annotations

import numpy as np


class _Periodic:
    def __init__(self, every: int = 21, band: float = 0.0):
        self.every, self.band, self.i = every, band, 0

    def reset(self):
        self.i = 0

    def __call__(self, obs, env):
        trade = self.i % self.every == 0
        self.i += 1
        if not trade:
            return None, 0.0
        w = np.asarray(self.target(env), dtype=np.float64)
        m = env.tradable_now()
        if not m.all():  # restrict to the point-in-time universe, keep the same gross exposure
            g = np.abs(w).sum()
            w = np.where(m, w, 0.0)
            if np.abs(w).sum() > 0 and env.c["action_mode"] == "long_only":
                w *= g / np.abs(w).sum()
        return w, self.band

    def target(self, env) -> np.ndarray:
        raise NotImplementedError


class EqualWeight(_Periodic):
    def target(self, env):
        return np.full(env.k, 1.0 / env.k) * (1.0 if env.c["action_mode"] == "long_only" else 1.0)


class BuyAndHold(EqualWeight):
    def __init__(self):
        super().__init__(every=10**9)


def _score(env, name: str = "gbm_score") -> np.ndarray:
    i = env.d.feat_names.index(name)
    return env.d.feats[env.t, env.assets, i].astype(np.float64)


class SignalLinear(_Periodic):
    """Long-only: weights proportional to positive scores (top-heavy). Long/short:
    weights proportional to demeaned scores, scaled to max_gross."""

    def __init__(self, every: int = 21, band: float = 0.0, top_frac: float = 0.5):
        super().__init__(every, band)
        self.top_frac = top_frac

    def target(self, env):
        s = _score(env)
        if env.c["action_mode"] == "long_only":
            cut = np.quantile(s, 1 - self.top_frac)
            w = np.maximum(s - cut, 0.0)
            return w / w.sum() if w.sum() > 0 else np.full(env.k, 1.0 / env.k)
        w = s - s.mean()
        return w / (np.abs(w).sum() + 1e-12) * env.c["max_gross"]


class MeanVariance(_Periodic):
    """max_w  alpha'w - (g/2) w'Sigma w - tau |w - w_cur|_1   s.t. w >= 0, sum w <= 1

    alpha_i = ic * sigma_i * score_i (Grinold's alpha = IC x vol x score).
    Solved by proximal gradient: gradient step, soft-threshold around w_cur, then
    projection onto {w >= 0, sum w <= 1}.
    """

    def __init__(self, every: int = 21, band: float = 0.0, ic: float = 0.05, gamma: float = 5.0,
                 tau: float = 0.002, iters: int = 300):
        super().__init__(every, band)
        self.ic, self.gamma, self.tau, self.iters = ic, gamma, tau, iters

    @staticmethod
    def _project(w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 0.0)
        if w.sum() <= 1.0:
            return w
        u = np.sort(w)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (css - 1))[0][-1]
        theta = (css[rho] - 1) / (rho + 1.0)
        return np.maximum(w - theta, 0.0)

    def target(self, env):
        s = _score(env)
        sig = env.d.sigma[env.t, env.assets]
        alpha = self.ic * sig * s * 21  # monthly horizon
        Sigma = env._cov(env.t) * 21
        cur = env.w_close
        L = self.gamma * np.linalg.eigvalsh(Sigma).max() + 1e-9
        step = 1.0 / L
        w = cur.copy()
        for _ in range(self.iters):
            g = alpha - self.gamma * Sigma @ w
            z = w + step * g
            d = z - cur
            z = cur + np.sign(d) * np.maximum(np.abs(d) - step * self.tau, 0.0)
            w = self._project(z)
        return w


def baseline_suite(band: float = 0.02) -> dict:
    return {
        "equal_weight_monthly": EqualWeight(every=21),
        "buy_and_hold": BuyAndHold(),
        "gbm_linear_monthly": SignalLinear(every=21),
        "gbm_mean_variance": MeanVariance(every=21),
        "gbm_linear_band": SignalLinear(every=5, band=band),
    }

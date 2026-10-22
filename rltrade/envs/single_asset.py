"""Single-asset env for the discrete baselines (tabular Q-learning, DQN).

It is the portfolio env with k=1 in long/short mode, plus a discrete action map:
  3 actions  {short, flat, long}       -> position {-1, 0, +1}
  3 actions  {reduce, hold, increase}  -> position +/- step (mode="delta")
Costs, timing and rewards are identical to the portfolio env, so results are
directly comparable.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rltrade.envs.portfolio import EnvData, PortfolioEnv


class DiscretePositionEnv(gym.Wrapper):
    def __init__(self, data: EnvData, rows: tuple[int, int], asset: int = 0, mode: str = "levels",
                 levels: tuple[float, ...] = (-1.0, 0.0, 1.0), step: float = 0.5, train: bool = True,
                 cfg: dict | None = None, **kw):
        c = {"n_assets": 1, "action_mode": "long_short", "random_subset": False,
             "random_init_weights": False, **(cfg or {})}
        env = PortfolioEnv(data, rows, cfg=c, train=train, assets=[asset], **kw)
        super().__init__(env)
        self.mode, self.levels, self.step_size = mode, np.asarray(levels), step
        self.action_space = spaces.Discrete(3 if mode == "delta" else len(levels))
        self.position = 0.0

    def reset(self, **kw):
        self.position = 0.0
        return self.env.reset(**kw)

    def step(self, action):
        if self.mode == "delta":
            self.position = float(np.clip(self.position + (int(action) - 1) * self.step_size, -1, 1))
        else:
            self.position = float(self.levels[int(action)])
        return self.env.step_weights(np.array([self.position]))


class TabularDiscretizer:
    """State for tabular Q-learning: (bucket of feature A) x (bucket of feature B) x position.

    Bucket edges are fitted on training observations (terciles by default).
    """

    def __init__(self, env: DiscretePositionEnv, features: tuple[str, ...] = ("ret_5", "gbm_score"),
                 n_bins: int = 3):
        base: PortfolioEnv = env.env
        self.idx = [base.d.feat_names.index(f) for f in features if f in base.d.feat_names]
        self.n_bins = n_bins
        rows = slice(base.lo, base.hi)
        a = int(base.fixed_assets[0])
        self.edges = [np.quantile(base.d.feats[rows, a, i], np.linspace(0, 1, n_bins + 1)[1:-1]) for i in self.idx]
        self.n_pos = env.action_space.n if env.mode == "levels" else 5
        self.n_states = n_bins ** len(self.idx) * self.n_pos
        self.env = env

    def __call__(self, obs: np.ndarray) -> int:
        F = self.env.env.F
        s = 0
        for e, i in zip(self.edges, self.idx):
            s = s * self.n_bins + int(np.searchsorted(e, obs[i]))
        pos = self.env.position
        p = int(np.argmin(np.abs(self.env.levels - pos))) if self.env.mode == "levels" else int(round((pos + 1) * 2))
        return s * self.n_pos + p

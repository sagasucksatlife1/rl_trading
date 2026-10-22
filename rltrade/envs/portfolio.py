"""Multi-asset portfolio allocation environment (the main env).

Timing (the anti-leak rule), for a decision at row t:
  * observation  = features up to close t, portfolio weights marked to close t
  * action       = target weights, executed at OPEN t+1 against the weights held
                   just before that open (previous target drifted open t -> open t+1)
  * reward       = net return from open t+1 to open t+2, after costs of the trade

Action spaces (`action_mode`):
  long_only   a in [-1,1]^(k+1)  -> softmax(a * temperature) over k assets + cash
  long_short  a in [-1,1]^k      -> weights a, scaled so gross <= max_gross
Optional `learned_band`: one extra action dim sets the no-trade band in [0, band_max].

Training-mode tricks to get "fresh" episodes from one history:
  random 1-year windows, random asset subsets (and order), random initial weights,
  random cost multiplier per episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rltrade.data.synthetic import Market
from rltrade.envs.costs import CostModel, liquidity_inputs
from rltrade.envs.rewards import Reward
from rltrade.features.library import FeatureSet
from rltrade.risk.overlay import RiskOverlay


@dataclass
class EnvData:
    """Aligned numpy arrays for one market. Rows = dates."""
    dates: np.ndarray
    assets: list[str]
    open: np.ndarray  # [T, N]
    close: np.ndarray  # [T, N]
    sigma: np.ndarray  # [T, N] daily vol, causal
    adv: np.ndarray  # [T, N] average daily volume, causal
    feats: np.ndarray  # [T, N, F] normalised
    glob: np.ndarray  # [T, G] normalised
    feat_names: list[str]
    glob_names: list[str]
    first_row: int  # first row with valid features & signal

    @classmethod
    def build(cls, market: Market, fs_norm: FeatureSet, first_row: int,
              feature_names: list[str] | None = None) -> "EnvData":
        names = feature_names or fs_norm.asset_names
        idx = [fs_norm.asset_names.index(n) for n in names]
        close = market.close.to_numpy(dtype=np.float64)
        sigma, adv = liquidity_inputs(close, market.volume.to_numpy(dtype=np.float64))
        return cls(np.asarray(market.dates), list(market.assets), market.open.to_numpy(dtype=np.float64),
                   close, sigma, adv, fs_norm.asset[:, :, idx].astype(np.float32),
                   fs_norm.glob.astype(np.float32), list(names), list(fs_norm.glob_names), first_row)

    @property
    def n_rows(self) -> int:
        return len(self.dates)

    def log_close_returns(self) -> np.ndarray:
        lr = np.diff(np.log(self.close), axis=0)
        return np.vstack([np.zeros((1, lr.shape[1])), lr])


DEFAULTS = dict(
    n_assets=None,  # k; None = all assets in the universe
    action_mode="long_only",
    temperature=3.0,
    max_gross=1.0,
    band=0.0,  # fixed no-trade band
    learned_band=False,
    band_max=0.05,
    episode_len=252,
    random_start=True,
    random_subset=True,
    random_init_weights=True,
    cost_mult_range=(0.5, 3.0),
    nav0=1e7,  # currency units; drives market impact
    window=1,  # stack the last `window` observations (POMDP help for MLP policies)
    cov_window=63,
)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: EnvData, rows: tuple[int, int], cfg: dict | None = None,
                 cost: CostModel | None = None, reward: dict | None = None,
                 overlay: RiskOverlay | None = None, train: bool = True, assets: list[int] | None = None,
                 seed: int | None = None):
        super().__init__()
        self.d = data
        self.c = {**DEFAULTS, **(cfg or {})}
        self.train = train
        self.base_cost = cost or CostModel()
        self.reward_fn = Reward(**(reward or {}))
        self.overlay = overlay
        lo, hi = max(rows[0], data.first_row + self.c["window"] - 1), min(rows[1], data.n_rows - 2)
        if hi - lo < 10:
            raise ValueError(f"row range too short: {rows}")
        self.lo, self.hi = lo, hi  # decision rows t in [lo, hi)
        N = len(data.assets)
        self.k = self.c["n_assets"] or N
        self.fixed_assets = np.asarray(assets if assets is not None else np.arange(self.k))
        self.F = data.feats.shape[2]
        self.G = data.glob.shape[1]
        self.lr = data.log_close_returns()
        n_act = self.k + (1 if self.c["action_mode"] == "long_only" else 0) + int(self.c["learned_band"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.obs_dim_single = self.k * self.F + self.G + (self.k + 1) + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim_single * self.c["window"],),
                                            dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.t = lo

    # ---------- helpers ----------
    def _cov(self, t: int) -> np.ndarray:
        w = self.c["cov_window"]
        r = self.lr[max(1, t - w + 1): t + 1][:, self.assets]
        if len(r) < 5:
            return np.diag(self.d.sigma[t, self.assets] ** 2)
        return np.cov(r.T) + 1e-10 * np.eye(len(self.assets))

    def action_to_weights(self, a: np.ndarray) -> tuple[np.ndarray, float]:
        """Map a raw action to (risky weights [k], band)."""
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        band = self.c["band"]
        if self.c["learned_band"]:
            band = (a[-1] + 1) / 2 * self.c["band_max"]
            a = a[:-1]
        if self.c["action_mode"] == "long_only":
            z = a * self.c["temperature"]
            e = np.exp(z - z.max())
            w = (e / e.sum())[: self.k]
        else:
            w = a.copy()
            g = np.abs(w).sum()
            if g > self.c["max_gross"]:
                w *= self.c["max_gross"] / g
        return w, band

    @staticmethod
    def apply_band(target: np.ndarray, current: np.ndarray, band: float, long_only: bool) -> np.ndarray:
        if band <= 0:
            return target
        w = np.where(np.abs(target - current) < band, current, target)
        if long_only and w.sum() > 1.0:
            w = w / w.sum()
        return w

    def _obs_single(self, t: int) -> np.ndarray:
        a = self.assets
        port = np.concatenate([self.w_close, [1.0 - self.w_close.sum()]])
        dd = 1.0 - self.nav_close / self.peak
        state = np.array([dd * 10, np.log(self.nav_close / self.nav_start) * 10,
                          min(self.since_trade, 63) / 63.0])
        return np.concatenate([self.d.feats[t, a].reshape(-1), self.d.glob[t], port, state]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        frames = list(self._frames)
        return np.concatenate(frames).astype(np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        L = self.c["episode_len"]
        N = len(self.d.assets)
        if self.train and self.c["random_start"] and self.hi - self.lo > L:
            self.t0 = int(self._rng.integers(self.lo, self.hi - L))
            self.t_end = self.t0 + L
        else:
            self.t0, self.t_end = self.lo, self.hi
        if self.train and self.c["random_subset"]:
            self.assets = self._rng.choice(N, size=self.k, replace=False)
        else:
            self.assets = self.fixed_assets
        if "init_weights" in opts:
            w0 = np.asarray(opts["init_weights"], dtype=np.float64)
        elif self.train and self.c["random_init_weights"]:
            w0 = self._rng.dirichlet(np.ones(self.k + 1))[: self.k]
            if self.c["action_mode"] == "long_short":
                w0 = w0 * self._rng.choice([-1, 1], size=self.k)
        else:
            w0 = np.zeros(self.k)
        mult = (self._rng.uniform(*self.c["cost_mult_range"]) if self.train else 1.0) * opts.get("cost_mult", 1.0)
        self.cost = self.base_cost.scaled(mult)
        self.t = self.t0
        self.nav = 1.0
        self.nav_start = 1.0
        self.nav_close = 1.0
        self.peak = 1.0
        self.since_trade = 0
        self.w_open = w0.copy()  # held from open t (to be drifted to open t+1)
        self.w_close = w0.copy()
        self.reward_fn.reset()
        if self.overlay is not None:
            self.overlay.reset(1.0)
        from collections import deque

        self._frames = deque(maxlen=self.c["window"])
        for back in range(self.c["window"] - 1, -1, -1):
            self._frames.append(self._obs_single(self.t - back))
        return self._obs(), {"date": self.d.dates[self.t]}

    def step(self, action):
        target, band = self.action_to_weights(action)
        return self.step_weights(target, band)

    def step_weights(self, target: np.ndarray, band: float | None = None):
        """Execute target risky weights (pre-band, pre-overlay) at open t+1."""
        t, a = self.t, self.assets
        band = self.c["band"] if band is None else band
        # 1. drift held weights from open t to open t+1 (pre-trade weights)
        R_now = self.d.open[t + 1, a] / self.d.open[t, a] - 1.0
        g = 1.0 + self.w_open @ R_now
        current = self.w_open * (1.0 + R_now) / g
        self.nav *= g
        # 2. band, overlay, trade at open t+1 (target None = hold, no trade)
        if target is None:
            target = current.copy()
        target = self.apply_band(np.asarray(target, dtype=np.float64), current, band,
                                 self.c["action_mode"] == "long_only")
        flags: dict = {}
        if self.overlay is not None:
            self.overlay.update_nav(self.nav)
            target, flags = self.overlay.apply(target, current, self._cov(t))
        dw = target - current
        turnover = float(np.abs(dw).sum())
        cost = self.cost.cost(dw, self.nav * self.c["nav0"], self.d.open[t + 1, a], self.d.adv[t, a],
                              self.d.sigma[t, a])
        self.nav *= 1.0 - cost
        self.since_trade = 0 if turnover > 1e-9 else self.since_trade + 1
        # 3. hold target from open t+1 to open t+2: that is this step's realised return
        R_next = self.d.open[t + 2, a] / self.d.open[t + 1, a] - 1.0
        r_gross = float(target @ R_next)
        r_net = (1.0 - cost) * (1.0 + r_gross) - 1.0
        # bookkeeping for the next observation (marked at close t+1, no peek at open t+2)
        R_close = self.d.close[t + 1, a] / self.d.open[t + 1, a] - 1.0
        gc = 1.0 + target @ R_close
        self.w_close = target * (1.0 + R_close) / gc
        self.nav_close = self.nav * gc
        self.peak = max(self.peak, self.nav_close)
        self.w_open = target
        reward = self.reward_fn(r_net)
        # nav after the hold period is realised at open t+2 (the next step drifts with R_now)
        self.t += 1
        terminated = False
        truncated = self.t >= self.t_end or self.t + 2 >= self.d.n_rows
        self._frames.append(self._obs_single(self.t))
        info = {"date": self.d.dates[t + 1], "r_net": r_net, "r_gross": r_gross, "cost": cost,
                "turnover": turnover, "weights": target.copy(), "cash": 1.0 - float(target.sum()),
                "nav": self.nav * (1.0 + r_gross), "overlay": flags, "band": band}
        return self._obs(), float(reward), terminated, truncated, info

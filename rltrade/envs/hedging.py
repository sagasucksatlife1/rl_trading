"""Deep hedging env (Buehler et al., 2019): hedge a short call under costs.

The agent is short one European call and holds `h` shares of the underlying.
Each step it picks a new hedge ratio in [0, 1]. Proportional costs on every
rebalance. Reward = change in hedged P&L minus a risk penalty; the final
evaluation metric is the CVaR of terminal hedging error, which is how real books
are judged. Baseline: Black-Scholes delta hedging (optimal only with zero costs).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import norm


def bs_call(S, K, T, sigma, r=0.0):
    T = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)


class HedgingEnv(gym.Env):
    def __init__(self, S0: float = 100.0, K: float = 100.0, T: float = 30 / 252, n_steps: int = 30,
                 sigma: float = 0.2, cost: float = 0.001, risk_lambda: float = 1.0, stoch_vol: bool = True,
                 seed: int | None = None):
        super().__init__()
        self.p = dict(S0=S0, K=K, T=T, N=n_steps, sigma=sigma, cost=cost, lam=risk_lambda, sv=stoch_vol)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self.p
        self.dt = p["T"] / p["N"]
        self.i = 0
        self.S = p["S0"]
        self.vol = p["sigma"]
        self.h = 0.0
        # Sold the call at model value, so mark-to-model P&L starts at 0; at expiry the
        # option is marked at its payoff and the cumulative P&L is the hedging error.
        self.opt_val, _ = bs_call(self.S, p["K"], p["T"], p["sigma"])
        self.pnl = 0.0
        return self._obs(), {}

    def _obs(self):
        p = self.p
        tau = p["T"] - self.i * self.dt
        _, delta = bs_call(self.S, p["K"], tau, p["sigma"])
        return np.array([np.log(self.S / p["K"]) * 10, tau / p["T"], self.h, delta], dtype=np.float32)

    def step(self, action):
        p = self.p
        new_h = float((np.clip(np.asarray(action).reshape(-1)[0], -1, 1) + 1) / 2)
        c = p["cost"] * abs(new_h - self.h) * self.S
        self.h = new_h
        if p["sv"]:  # simple stochastic vol: log-vol AR(1)
            self.vol = float(np.clip(self.vol * np.exp(0.3 * np.sqrt(self.dt) * self.rng.normal()), 0.05, 1.0))
        S_new = self.S * np.exp(-0.5 * self.vol**2 * self.dt + self.vol * np.sqrt(self.dt) * self.rng.normal())
        self.i += 1
        tau = p["T"] - self.i * self.dt
        done = self.i >= p["N"]
        new_opt = max(S_new - p["K"], 0.0) if done else bs_call(S_new, p["K"], tau, p["sigma"])[0]
        dpnl = self.h * (S_new - self.S) - (new_opt - self.opt_val) - c
        self.S, self.opt_val = S_new, new_opt
        self.pnl += dpnl
        reward = dpnl - p["lam"] * dpnl**2
        info = {"terminal_pnl": self.pnl if done else None}
        return self._obs(), float(reward), done, False, info


class BSDelta:
    def __call__(self, obs):
        return np.array([obs[3] * 2 - 1], dtype=np.float32)


def evaluate_hedging(env: HedgingEnv, policy, n_episodes: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    pnl = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
        pnl.append(info["terminal_pnl"])
    a = np.asarray(pnl)
    q = np.quantile(a, alpha)
    return {"mean": float(a.mean()), "std": float(a.std()), f"cvar_{int(alpha*100)}": float(a[a <= q].mean())}

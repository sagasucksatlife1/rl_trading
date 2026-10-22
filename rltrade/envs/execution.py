"""Optimal execution env: liquidate (or buy) a parent order over N intervals.

Almgren-Chriss (2000) dynamics, arithmetic Brownian mid price:
    S_{k+1} = S_k + sigma * sqrt(tau) * z_k - gamma * v_k          (permanent impact)
    execution price  S~_k = S_k - eps * sign(v) - eta * v_k / tau   (temporary impact)
where v_k shares are sold in interval k. Implementation shortfall (IS) =
X * S_0 - sum v_k S~_k. The agent picks the fraction of REMAINING shares to
trade each interval (action a in [-1,1] -> fraction (a+1)/2); anything left at the end is dumped at the last step.

Reward per step = -(IS increment) / (X * S_0) in percent, minus risk aversion times
the variance of the remaining inventory's value (the AC mean-variance objective).

Baselines: TWAP and the closed-form Almgren-Chriss trajectory
    x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T),  kappa ~ sqrt(lam sigma^2 / eta)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExecutionEnv(gym.Env):
    def __init__(self, X: float = 1e5, S0: float = 50.0, sigma: float = 0.95, n_steps: int = 20,
                 horizon_days: float = 1.0, eta: float = 2.5e-6, gamma: float = 2.5e-7, eps: float = 0.0625,
                 risk_aversion: float = 1e-6, random_params: bool = True, seed: int | None = None):
        super().__init__()
        self.p = dict(X=X, S0=S0, sigma=sigma, N=n_steps, T=horizon_days, eta=eta, gamma=gamma, eps=eps,
                      lam=risk_aversion)
        self.random_params = random_params
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # frac = (a+1)/2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = dict(self.p)
        if self.random_params:  # domain randomisation over vol and impact
            p["sigma"] *= self.rng.uniform(0.5, 2.0)
            p["eta"] *= self.rng.uniform(0.5, 2.0)
        self.q = p
        self.tau = p["T"] / p["N"]
        self.k = 0
        self.x = p["X"]
        self.S = p["S0"]
        self.cash = 0.0
        self.last_dS = 0.0
        return self._obs(), {}

    def _obs(self):
        p = self.q
        return np.array([self.x / p["X"], 1 - self.k / p["N"], self.last_dS / (p["sigma"] * np.sqrt(self.tau)),
                         p["sigma"] / self.p["sigma"], p["eta"] / self.p["eta"]], dtype=np.float32)

    def _trade(self, v: float) -> float:
        p = self.q
        px = self.S - p["eps"] * np.sign(v) - p["eta"] * v / self.tau
        return v * px

    def step(self, action):
        p = self.q
        frac = float((np.clip(np.asarray(action).reshape(-1)[0], -1, 1) + 1) / 2)
        last = self.k == p["N"] - 1
        v = self.x if last else frac * self.x
        proceeds = self._trade(v)
        ideal = v * p["S0"]
        self.cash += proceeds
        self.x -= v
        dS = p["sigma"] * np.sqrt(self.tau) * self.rng.normal() - p["gamma"] * v
        self.S += dS
        self.last_dS = dS
        self.k += 1
        is_inc = (ideal - proceeds) / (p["X"] * p["S0"]) * 1e4  # bps
        risk = p["lam"] * (self.x**2) * (p["sigma"] ** 2) * self.tau / (p["X"] * p["S0"]) * 1e4
        done = self.k >= p["N"]
        info = {"shortfall_bps": (p["X"] * p["S0"] - self.cash) / (p["X"] * p["S0"]) * 1e4 if done else None,
                "traded": v}
        return self._obs(), float(-(is_inc + risk)) / 100.0, done, False, info  # reward in %-points


class TWAP:
    """Trade remaining / steps_left each interval."""

    def __init__(self, n_steps: int):
        self.N = n_steps

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        left = max(int(round(obs[1] * self.N)), 1)
        return np.array([2.0 / left - 1.0], dtype=np.float32)


class AlmgrenChriss:
    """Closed-form AC trajectory as a fraction-of-remaining policy."""

    def __init__(self, env: ExecutionEnv):
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        p, tau = self.env.q, self.env.tau
        N = p["N"]
        k = int(round((1 - obs[1]) * N))
        eta_t = p["eta"] - 0.5 * p["gamma"] * tau
        kt2 = p["lam"] * p["sigma"] ** 2 / max(eta_t, 1e-12)
        kappa = np.arccosh(1 + 0.5 * kt2 * tau**2) / tau if kt2 > 0 else 1e-9
        T = p["T"]
        hold = lambda j: np.sinh(kappa * (T - j * tau)) / np.sinh(kappa * T)
        now, nxt = hold(k), hold(k + 1)
        frac = 0.0 if now <= 0 else 1 - nxt / now
        return np.array([2.0 * frac - 1.0], dtype=np.float32)


def evaluate_execution(env: ExecutionEnv, policy, n_episodes: int = 500, seed: int = 0) -> dict:
    out = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
        out.append(info["shortfall_bps"])
    a = np.asarray(out)
    return {"mean_is_bps": float(a.mean()), "std_is_bps": float(a.std()), "p95_is_bps": float(np.quantile(a, 0.95))}

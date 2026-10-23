"""Market-making env in a simulated limit order book (Avellaneda-Stoikov dynamics).

No tick data needed: the simulator generates the order flow.
  * mid price: arithmetic BM with vol sigma, plus informed-flow jumps that move
    AGAINST the market maker right after a fill (adverse selection)
  * the agent quotes bid/ask offsets (delta_b, delta_a) from the mid
  * fills are Poisson with intensity  lambda(delta) = A * exp(-k * delta)

Reward = change in mark-to-market wealth - phi * inventory^2 (running inventory
penalty); a terminal penalty liquidates leftover inventory at a cost.

Baseline: Avellaneda-Stoikov (2008) closed form
    reservation r = s - q * g * sigma^2 * (T - t)
    spread        = g * sigma^2 * (T - t) + (2/g) * ln(1 + g/k)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarketMakingEnv(gym.Env):
    def __init__(self, S0: float = 100.0, sigma: float = 2.0, T: float = 1.0, n_steps: int = 200,
                 A: float = 140.0, k: float = 1.5, max_inventory: int = 10, phi: float = 0.01,
                 adverse_prob: float = 0.15, adverse_jump: float = 0.05, max_offset: float = 2.0,
                 terminal_cost: float = 0.5, random_params: bool = True, seed: int | None = None):
        super().__init__()
        self.p = dict(S0=S0, sigma=sigma, T=T, N=n_steps, A=A, k=k, Q=max_inventory, phi=phi,
                      adv_p=adverse_prob, adv_j=adverse_jump, max_off=max_offset, term=terminal_cost)
        self.random_params = random_params
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        q = dict(self.p)
        if self.random_params:
            q["sigma"] *= self.rng.uniform(0.5, 1.5)
            q["A"] *= self.rng.uniform(0.7, 1.3)
        self.q = q
        self.dt = q["T"] / q["N"]
        self.i = 0
        self.S = q["S0"]
        self.inv = 0
        self.cash = 0.0
        self.wealth = 0.0
        self.last_ret = 0.0
        return self._obs(), {}

    def _obs(self):
        q = self.q
        return np.array([self.inv / q["Q"], 1 - self.i / q["N"], self.last_ret / (q["sigma"] * np.sqrt(self.dt)),
                         q["sigma"] / self.p["sigma"]], dtype=np.float32)

    def offsets(self, action) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        return tuple(((a + 1) / 2 * self.q["max_off"]).tolist())  # type: ignore[return-value]

    def step(self, action):
        db, da = self.offsets(action)
        return self.step_offsets(db, da)

    def step_offsets(self, db: float, da: float):
        q, dt = self.q, self.dt
        lam_b = q["A"] * np.exp(-q["k"] * db) * dt
        lam_a = q["A"] * np.exp(-q["k"] * da) * dt
        bid_fill = self.inv < q["Q"] and self.rng.random() < 1 - np.exp(-lam_b)
        ask_fill = self.inv > -q["Q"] and self.rng.random() < 1 - np.exp(-lam_a)
        if bid_fill:
            self.inv += 1
            self.cash -= self.S - db
        if ask_fill:
            self.inv -= 1
            self.cash += self.S + da
        dS = q["sigma"] * np.sqrt(dt) * self.rng.normal()
        # informed flow: after a one-sided fill the price tends to move against the MM
        if bid_fill != ask_fill and self.rng.random() < q["adv_p"]:
            dS += -q["adv_j"] * q["sigma"] if bid_fill else q["adv_j"] * q["sigma"]
        self.S += dS
        self.last_ret = dS
        self.i += 1
        new_wealth = self.cash + self.inv * self.S
        reward = new_wealth - self.wealth - q["phi"] * self.inv**2
        self.wealth = new_wealth
        done = self.i >= q["N"]
        if done:
            reward -= q["term"] * abs(self.inv)
        info = {"pnl": self.wealth - (q["term"] * abs(self.inv) if done else 0.0), "inventory": self.inv,
                "fills": int(bid_fill) + int(ask_fill)}
        return self._obs(), float(reward), done, False, info


class AvellanedaStoikov:
    """Closed-form AS quotes, returned as raw offsets via `offsets(obs)`."""

    def __init__(self, env: MarketMakingEnv, gamma: float = 0.1):
        self.env, self.g = env, gamma

    def offsets(self, obs: np.ndarray) -> tuple[float, float]:
        q = self.env.q
        tleft = obs[1] * q["T"]
        inv = obs[0] * q["Q"]
        s2 = q["sigma"] ** 2
        r_off = -inv * self.g * s2 * tleft  # reservation minus mid
        half = 0.5 * (self.g * s2 * tleft + (2 / self.g) * np.log(1 + self.g / q["k"]))
        db = float(np.clip(half - r_off, 0, q["max_off"]))
        da = float(np.clip(half + r_off, 0, q["max_off"]))
        return db, da

    def __call__(self, obs):
        db, da = self.offsets(obs)
        m = self.env.q["max_off"]
        return np.array([db / m * 2 - 1, da / m * 2 - 1], dtype=np.float32)


def evaluate_mm(env: MarketMakingEnv, policy, n_episodes: int = 200, seed: int = 0) -> dict:
    pnl, inv = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        invs = []
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
            invs.append(abs(info["inventory"]))
        pnl.append(info["pnl"])
        inv.append(np.mean(invs))
    p = np.asarray(pnl)
    return {"mean_pnl": float(p.mean()), "std_pnl": float(p.std()),
            "sharpe_like": float(p.mean() / (p.std() + 1e-12)), "mean_abs_inventory": float(np.mean(inv))}

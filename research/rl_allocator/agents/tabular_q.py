"""Tabular Q-learning from scratch (Sutton & Barto ch. 6).

    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Epsilon-greedy exploration with linear decay. Its job: prove the env works and
teach the Bellman update with nothing hidden.
"""
from __future__ import annotations

import numpy as np


class TabularQ:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.05, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.rng = np.random.default_rng(seed)

    def act(self, s: int, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def train(self, env, discretize, episodes: int = 300, log_every: int = 50) -> list[float]:
        returns = []
        for ep in range(episodes):
            eps = self.eps_start + (self.eps_end - self.eps_start) * ep / max(episodes - 1, 1)
            obs, _ = env.reset()
            s = discretize(obs)
            done, total = False, 0.0
            while not done:
                a = self.act(s, eps)
                obs, r, term, trunc, _ = env.step(a)
                s2 = discretize(obs)
                target = r + (0.0 if term else self.gamma * self.Q[s2].max())
                self.Q[s, a] += self.alpha * (target - self.Q[s, a])
                s, total, done = s2, total + r, term or trunc
            returns.append(total)
            if log_every and (ep + 1) % log_every == 0:
                print(f"[tabular-q] ep {ep+1}/{episodes} eps={eps:.2f} mean return(last {log_every})="
                      f"{np.mean(returns[-log_every:]):.3f}")
        return returns

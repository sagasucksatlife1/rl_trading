"""DQN from scratch in PyTorch: Double DQN + Dueling heads + Prioritised Replay.

Pieces (Mnih et al. 2015; van Hasselt 2016; Wang 2016; Schaul 2016):
  * experience replay       - break temporal correlation of updates
  * target network          - a slowly updated copy for stable TD targets
  * Double DQN target       - y = r + g * Q_target(s', argmax_a Q_online(s', a))
                              fixes max-operator overestimation
  * Dueling head            - Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
  * PER                     - sample transitions with prob ~ |TD error|^alpha,
                              correct the bias with importance weights ^beta
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SumTree:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.tree = np.zeros(2 * capacity)
        self.ptr = 0
        self.size = 0

    def total(self) -> float:
        return self.tree[1]

    def update(self, idx: int, p: float) -> None:
        i = idx + self.cap
        self.tree[i] = p
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def find(self, v: float) -> int:
        i = 1
        while i < self.cap:
            left = 2 * i
            if v <= self.tree[left]:
                i = left
            else:
                v -= self.tree[left]
                i = left + 1
        return i - self.cap


class PrioritizedReplay:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, seed: int = 0):
        cap = 1
        while cap < capacity:
            cap *= 2
        self.tree = SumTree(cap)
        self.cap, self.alpha = cap, alpha
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.obs2 = np.zeros((cap, obs_dim), np.float32)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.max_p = 1.0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, d) -> None:
        i = self.tree.ptr
        self.obs[i], self.act[i], self.rew[i], self.obs2[i], self.done[i] = o, a, r, o2, d
        self.tree.update(i, self.max_p**self.alpha)
        self.tree.ptr = (i + 1) % self.cap
        self.tree.size = min(self.tree.size + 1, self.cap)

    def sample(self, n: int, beta: float):
        seg = self.tree.total() / n
        idx = np.array([self.tree.find(self.rng.uniform(seg * k, seg * (k + 1))) for k in range(n)])
        idx = np.minimum(idx, self.tree.size - 1)
        p = self.tree.tree[idx + self.cap] / self.tree.total()
        w = (self.tree.size * np.maximum(p, 1e-12)) ** (-beta)
        w /= w.max()
        return idx, self.obs[idx], self.act[idx], self.rew[idx], self.obs2[idx], self.done[idx], w.astype(np.float32)

    def update(self, idx, td) -> None:
        for i, e in zip(idx, np.abs(td) + 1e-6):
            self.max_p = max(self.max_p, e)
            self.tree.update(int(i), e**self.alpha)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.adv = nn.Linear(hidden, n_actions)
        self.val = nn.Linear(hidden, 1) if dueling else None

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        if not self.dueling:
            return a
        return self.val(h) + a - a.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 3e-4, gamma: float = 0.95,
                 buffer: int = 100_000, batch: int = 128, double: bool = True, dueling: bool = True,
                 per: bool = True, target_tau: float = 0.005, eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_frac: float = 0.3, learn_start: int = 1000, seed: int = 0):
        torch.manual_seed(seed)
        self.q = DuelingQNet(obs_dim, n_actions, dueling=dueling)
        self.tq = DuelingQNet(obs_dim, n_actions, dueling=dueling)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.rb = PrioritizedReplay(buffer, obs_dim, alpha=0.6 if per else 0.0, seed=seed)
        self.n_actions, self.gamma, self.batch = n_actions, gamma, batch
        self.double, self.tau = double, target_tau
        self.eps_start, self.eps_end, self.eps_frac = eps_start, eps_end, eps_frac
        self.learn_start = learn_start
        self.rng = np.random.default_rng(seed)

    def act(self, obs, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs[None], dtype=torch.float32)).argmax(1).item())

    def _learn(self, beta: float) -> float:
        idx, o, a, r, o2, d, w = self.rb.sample(self.batch, beta)
        o, o2 = torch.as_tensor(o), torch.as_tensor(o2)
        a, r, d, w = torch.as_tensor(a), torch.as_tensor(r), torch.as_tensor(d), torch.as_tensor(w)
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(o2).argmax(1, keepdim=True)
                q2 = self.tq(o2).gather(1, a2).squeeze(1)
            else:
                q2 = self.tq(o2).max(1).values
            y = r + self.gamma * (1 - d) * q2
        td = y - q
        loss = (w * nn.functional.smooth_l1_loss(q, y, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.rb.update(idx, td.detach().numpy())
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.tq.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())

    def train(self, env, total_steps: int = 50_000, log_every: int = 10_000) -> list[float]:
        obs, _ = env.reset()
        ep_ret, rets = 0.0, []
        for step in range(total_steps):
            frac = min(step / (self.eps_frac * total_steps), 1.0)
            eps = self.eps_start + frac * (self.eps_end - self.eps_start)
            a = self.act(obs, eps)
            obs2, r, term, trunc, _ = env.step(a)
            self.rb.add(obs, a, r, obs2, float(term))
            obs, ep_ret = obs2, ep_ret + r
            if term or trunc:
                rets.append(ep_ret)
                obs, _ = env.reset()
                ep_ret = 0.0
            if step >= self.learn_start:
                self._learn(beta=0.4 + 0.6 * step / total_steps)
            if log_every and (step + 1) % log_every == 0 and rets:
                print(f"[dqn] step {step+1}/{total_steps} eps={eps:.2f} mean ep return={np.mean(rets[-10:]):.3f}")
        return rets

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

"""Offline RL from logged decisions (CQL / IQL), with no live exploration.

Data comes from the shadow/paper decision log (`DecisionLog.transitions()`), or,
for development, from `collect_transitions`, which replays a behaviour policy
(plus exploration noise) through the backtest env.

Uses d3rlpy when installed. Without it, `bc_fit` trains a behaviour-cloning
policy as the conservative fallback (the policy CQL/IQL regularise towards).
"""
from __future__ import annotations

import numpy as np


def collect_transitions(env, policy, episodes: int = 20, noise: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    S, A, R, S2, D = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            w, band = policy(obs, env)
            if w is None:
                w = env.w_close.copy()
            w = np.clip(np.asarray(w) + rng.normal(0, noise / env.k, size=env.k), 0, None)
            w = w / max(w.sum(), 1.0)
            obs2, r, term, trunc, info = env.step_weights(w, band)
            S.append(obs)
            A.append(info["weights"])
            R.append(r)
            S2.append(obs2)
            D.append(float(term or trunc))
            obs, done = obs2, term or trunc
    f = lambda x: np.asarray(x, dtype=np.float32)
    return {"obs": f(S), "actions": f(A), "rewards": f(R), "next_obs": f(S2), "terminals": f(D)}


def train_offline(data: dict, algo: str = "cql", steps: int = 20_000, seed: int = 0):
    import d3rlpy

    ds = d3rlpy.dataset.MDPDataset(data["obs"], data["actions"], data["rewards"], data["terminals"])
    if algo == "cql":
        model = d3rlpy.algos.CQLConfig().create(device="cpu:0")
    elif algo == "iql":
        model = d3rlpy.algos.IQLConfig().create(device="cpu:0")
    else:
        raise ValueError(algo)
    model.fit(ds, n_steps=steps, n_steps_per_epoch=max(steps // 10, 1), show_progress=False)
    return model


def bc_fit(data: dict, hidden: int = 128, epochs: int = 200, lr: float = 1e-3, seed: int = 0):
    """Behaviour cloning: obs -> executed weights (softmax head incl. cash)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X = torch.as_tensor(data["obs"])
    W = torch.as_tensor(data["actions"])
    Y = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=0)], 1)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        logp = torch.log_softmax(net(X), 1)
        loss = -(Y * logp).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    class BCPolicy:
        def __call__(self, obs, env):
            with torch.no_grad():
                p = torch.softmax(net(torch.as_tensor(obs[None])), 1)[0].numpy()
            return p[: env.k], None

    return BCPolicy(), float(loss.item())

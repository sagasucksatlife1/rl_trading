"""Run a policy through an env deterministically and record everything.

A policy is any callable `policy(obs, env) -> (weights | None, band | None)` for
weight-space policies, or an SB3-style model wrapped by `SB3Policy`/`Ensemble`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rollout(env, policy, options: dict | None = None) -> pd.DataFrame:
    obs, info = env.reset(options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    rows = []
    done = False
    while not done:
        w, band = policy(obs, env)
        obs, reward, term, trunc, info = env.step_weights(w, band)
        done = term or trunc
        rows.append({"date": info["date"], "r_net": info["r_net"], "r_gross": info["r_gross"],
                     "cost": info["cost"], "turnover": info["turnover"], "gross": float(np.abs(info["weights"]).sum()),
                     "cash": info["cash"], "reward": reward, "nav": info["nav"],
                     "weights": info["weights"], "overlay": info["overlay"]})
    rec = pd.DataFrame(rows).set_index("date")
    return rec


def equal_weight_bench(env) -> np.ndarray:
    """Daily open-to-open equal-weight returns over the same rows, no costs (for beta)."""
    d, a = env.d, env.fixed_assets
    rows = range(env.lo, env.hi)
    out = [float(np.mean(d.open[t + 2, a] / d.open[t + 1, a] - 1)) for t in rows if t + 2 < d.n_rows]
    return np.asarray(out)

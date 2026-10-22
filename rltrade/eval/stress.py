"""Robustness checks run on the out-of-sample block.

  cost_stress       rerun at 2x and 3x costs: an edge that dies at 2x is not an edge
  regime_breakdown  performance in calm vs turbulent periods (HMM filtered prob)
  behaviour_audit   is the policy secretly "always 100% long"? (beta, not skill)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rltrade.eval.metrics import summarize
from rltrade.eval.rollout import rollout


def cost_stress(make_env, policy, mults=(1.0, 2.0, 3.0)) -> pd.DataFrame:
    rows = []
    for m in mults:
        rec = rollout(make_env(), policy, options={"cost_mult": m})
        s = summarize(rec)
        rows.append({"cost_mult": m, "sharpe": s["sharpe"], "cagr": s["cagr"], "max_dd": s["max_dd"],
                     "cost_drag_ann": s["cost_drag_ann"]})
    return pd.DataFrame(rows).set_index("cost_mult")


def regime_breakdown(rec: pd.DataFrame, env, threshold: float = 0.5) -> pd.DataFrame:
    gi = env.d.glob_names.index("p_turbulent") if "p_turbulent" in env.d.glob_names else None
    if gi is None:
        return pd.DataFrame()
    # rec rows are decisions t = t0 .. ; the regime known at decision time is row t
    rows = np.arange(env.t0, env.t0 + len(rec))
    # glob is z-scored (monotone in the raw probability): split the block at its median,
    # so "turbulent" = the more turbulent half of the test period.
    p = env.d.glob[rows, gi]
    turbulent = p > np.median(p) if np.std(p) > 0 else np.zeros(len(p), bool)
    out = []
    for name, mask in (("calm", ~turbulent), ("turbulent", turbulent)):
        if mask.sum() > 20:
            s = summarize(rec.iloc[mask])
            out.append({"regime": name, "days": int(mask.sum()), "sharpe": s["sharpe"],
                        "ann_ret": float(rec["r_net"].iloc[mask].mean() * 252), "max_dd": s["max_dd"],
                        "avg_gross": s.get("avg_gross")})
    return pd.DataFrame(out).set_index("regime") if out else pd.DataFrame()


def behaviour_audit(rec: pd.DataFrame, bench: np.ndarray) -> dict:
    W = np.stack(rec["weights"].to_numpy())
    gross = np.abs(W).sum(1)
    r = rec["r_net"].to_numpy()
    n = min(len(r), len(bench))
    beta = float(np.cov(r[:n], bench[:n])[0, 1] / (np.var(bench[:n], ddof=1) + 1e-18))
    corr = float(np.corrcoef(r[:n], bench[:n])[0, 1])
    Wn = np.abs(W) / np.maximum(gross, 1e-12)[:, None]
    conc = (Wn**2).sum(1)  # Herfindahl of the invested book
    flags = []
    if gross.mean() > 0.95 and gross.std() < 0.03:
        flags.append("near-constant full exposure: likely learned beta, not timing")
    if corr > 0.9:
        flags.append("returns >0.9 correlated with equal weight: little cross-sectional skill")
    if (gross < 0.02).mean() > 0.5:
        flags.append("flat more than half the time")
    return {"avg_gross": float(gross.mean()), "std_gross": float(gross.std()), "beta_to_ew": beta,
            "corr_to_ew": corr, "avg_herfindahl": float(conc.mean()), "eff_n_assets": float((1 / conc[gross > 0]).mean())
            if (gross > 0).any() else 0.0, "flags": flags}

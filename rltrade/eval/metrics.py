"""Performance metrics from a daily net-return series (+ optional weights/turnover)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + np.asarray(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((1 - nav / peak).max()) if len(nav) else 0.0


def summarize(rec: pd.DataFrame, bench: np.ndarray | None = None) -> dict:
    """`rec` is a rollout record with columns r_net, cost, turnover, gross, cash."""
    r = rec["r_net"].to_numpy()
    n = len(r)
    years = n / ANN
    nav = float(np.prod(1 + r))
    downside = r[r < 0]
    dd = max_drawdown(r)
    cagr = nav ** (1 / years) - 1 if years > 0 and nav > 0 else -1.0
    out = {
        "days": n,
        "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)) if n > 1 else 0.0,
        "sharpe": sharpe(r),
        "sortino": float(r.mean() / (np.sqrt((downside**2).mean()) + 1e-12) * np.sqrt(ANN)) if len(downside) else 0.0,
        "max_dd": dd,
        "calmar": cagr / dd if dd > 0 else 0.0,
        "hit_rate": float((r > 0).mean()),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt() + 3),
        "turnover_ann": float(rec["turnover"].sum() / years) if "turnover" in rec and years else 0.0,
        "cost_drag_ann": float(rec["cost"].sum() / years) if "cost" in rec and years else 0.0,
    }
    if "gross" in rec:
        out["avg_gross"] = float(rec["gross"].mean())
    if "cash" in rec:
        out["avg_cash"] = float(rec["cash"].mean())
    if bench is not None and len(bench) == n and np.var(bench) > 0:
        b = np.asarray(bench)
        beta = float(np.cov(r, b)[0, 1] / np.var(b, ddof=1))
        out["beta_to_bench"] = beta
        resid = r - beta * b
        out["alpha_ann"] = float(resid.mean() * ANN)
        out["info_ratio"] = sharpe(resid)
    return out

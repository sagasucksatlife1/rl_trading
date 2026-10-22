"""Independent share-based backtest used to reconcile the env's accounting.

The env tracks weights and multiplies NAV by returns. This module tracks SHARES
and CASH in currency, the way a broker statement does. Fed the same sequence of
executed weights, both must agree to the cent (tests/test_env_accounting.py).
"""
from __future__ import annotations

import numpy as np

from rltrade.envs.costs import CostModel
from rltrade.envs.portfolio import EnvData


def share_backtest(d: EnvData, assets: np.ndarray, t0: int, executed: np.ndarray, cost: CostModel,
                   capital: float = 1e7, init_weights: np.ndarray | None = None) -> np.ndarray:
    """Replay executed weights; return NAV (currency) at open t0+2, t0+3, ...

    executed[i] = weights traded at open (t0 + i + 1) (post band/overlay), as the
    env reports in info["weights"].
    """
    a = np.asarray(assets)
    w0 = np.zeros(len(a)) if init_weights is None else np.asarray(init_weights)
    P0 = d.open[t0, a]
    shares = w0 * capital / P0
    cash = capital * (1 - w0.sum())
    navs = []
    for i, w in enumerate(executed):
        t = t0 + i
        P = d.open[t + 1, a]
        V = cash + shares @ P
        cur_w = shares * P / V
        dw = w - cur_w
        c = cost.cost(dw, V, P, d.adv[t, a], d.sigma[t, a]) * V
        V_after = V - c
        shares = w * V_after / P
        cash = V_after * (1 - w.sum())
        navs.append(cash + shares @ d.open[t + 2, a])
    return np.asarray(navs)

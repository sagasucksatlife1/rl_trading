"""Reward functions. All are per-step and stateful (call `reset()` per episode).

  log         r_t = log(1 + R_net)
  risk        r_t = log(1 + R_net) - lam * risk_t, risk in {variance, downside, drawdown}
  dsr         differential Sharpe ratio (Moody & Saffell, 1998)
  cvar        r_t = log(1 + R_net) - lam * expected shortfall excess of recent returns

Sharpe at episode end is a bad reward because it is sparse. The DSR is the
first-order term of the change in an exponentially weighted Sharpe when R_t
arrives, so summing it approximately maximises Sharpe:

    dA = R_t - A_{t-1},   dB = R_t^2 - B_{t-1}
    D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
    A_t = A_{t-1} + eta dA,   B_t = B_{t-1} + eta dB
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Reward:
    def __init__(self, kind: str = "log", lam: float = 0.0, risk: str = "variance", window: int = 63,
                 eta: float = 1 / 63, scale: float = 100.0, cvar_alpha: float = 0.05, **_):
        self.kind, self.lam, self.risk, self.window = kind, lam, risk, window
        self.eta, self.scale, self.cvar_alpha = eta, scale, cvar_alpha
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque(maxlen=self.window)
        self.A = 0.0
        self.B = 1e-4  # ~1% daily vol prior so the DSR denominator starts sane
        self.peak = 1.0
        self.nav = 1.0
        self.prev_dd = 0.0

    def __call__(self, r_net: float) -> float:
        lr = float(np.log1p(r_net))
        self.hist.append(r_net)
        self.nav *= 1.0 + r_net
        self.peak = max(self.peak, self.nav)
        dd = 1.0 - self.nav / self.peak
        if self.kind == "log":
            out = lr
        elif self.kind == "risk":
            if self.risk == "variance":
                pen = float(np.var(self.hist)) if len(self.hist) > 1 else 0.0
            elif self.risk == "downside":
                pen = min(r_net, 0.0) ** 2
            elif self.risk == "drawdown":
                pen = max(0.0, dd - self.prev_dd)
            else:
                raise ValueError(self.risk)
            out = lr - self.lam * pen
        elif self.kind == "dsr":
            dA = r_net - self.A
            dB = r_net**2 - self.B
            denom = max(self.B - self.A**2, 1e-8) ** 1.5
            out = (self.B * dA - 0.5 * self.A * dB) / denom
            self.A += self.eta * dA
            self.B += self.eta * dB
            self.prev_dd = dd
            return float(np.clip(out, -10, 10)) * 0.1  # already scale-free; O(0.1) per step
        elif self.kind == "cvar":
            h = np.asarray(self.hist)
            if len(h) >= 10:
                q = np.quantile(h, self.cvar_alpha)
                es = -h[h <= q].mean()
            else:
                es = 0.0
            out = lr - self.lam * max(es, 0.0) * (r_net < 0)
        else:
            raise ValueError(f"unknown reward kind {self.kind!r}")
        self.prev_dd = dd
        return out * self.scale

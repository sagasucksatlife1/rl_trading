"""Reward functions. All are per-step and stateful (call `reset()` per episode).

Selected by `reward.kind` (Ticket 3 A/B lives behind this flag):

  log         r_t = log(1 + R_net)
  risk        r_t = log(1 + R_net) - lam * risk_t, risk in {variance, downside, drawdown}
  dsr         differential Sharpe ratio (Moody & Saffell, 1998)            [Sharpe-style]
  turnover    r_t = log(1 + R_net) - kappa * turnover_t                     [explicit turnover penalty]
  cvar        r_t = log(1 + R_net) - lam * (L_t - VaR_t)^+ / alpha          [risk-sensitive, CVaR]

`turnover_penalty` (kappa) can also be added on top of ANY kind. It is a shadow
cost per unit of NAV traded, charged in return units on top of the real costs
already inside R_net; it shapes the policy towards trading less without
changing what the env actually pays.

DSR: the first-order term of the change in an exponentially weighted Sharpe
when R_t arrives, so summing it approximately maximises Sharpe:
    dA = R_t - A_{t-1},   dB = R_t^2 - B_{t-1}
    D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}

CVaR: Rockafellar-Uryasev. CVaR_a(L) = min_v v + E[(L - v)^+] / a, with L = -R the
loss. With v fixed at the rolling a-quantile of PAST losses (VaR_t), the per-step
term (L_t - VaR_t)^+ / a is the part of CVaR the current action controls.
"""
from __future__ import annotations

from collections import deque

import numpy as np

KINDS = ("log", "risk", "dsr", "turnover", "cvar")


class Reward:
    def __init__(self, kind: str = "log", lam: float = 0.0, risk: str = "variance", window: int = 63,
                 eta: float = 1 / 63, scale: float = 100.0, cvar_alpha: float = 0.05,
                 turnover_penalty: float = 0.0, **_):
        if kind not in KINDS:
            raise ValueError(f"unknown reward kind {kind!r}; choose from {KINDS}")
        self.kind, self.lam, self.risk, self.window = kind, lam, risk, window
        self.eta, self.scale, self.cvar_alpha = eta, scale, cvar_alpha
        self.kappa = turnover_penalty
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque(maxlen=self.window)
        self.A = 0.0
        self.B = 1e-4  # ~1% daily vol prior so the DSR denominator starts sane
        self.peak = 1.0
        self.nav = 1.0
        self.prev_dd = 0.0

    def __call__(self, r_net: float, turnover: float = 0.0) -> float:
        lr = float(np.log1p(r_net))
        past = np.asarray(self.hist)
        self.hist.append(r_net)
        self.nav *= 1.0 + r_net
        self.peak = max(self.peak, self.nav)
        dd = 1.0 - self.nav / self.peak
        shadow = self.kappa * turnover
        if self.kind == "dsr":
            dA = r_net - self.A
            dB = r_net**2 - self.B
            denom = max(self.B - self.A**2, 1e-8) ** 1.5
            out = (self.B * dA - 0.5 * self.A * dB) / denom
            self.A += self.eta * dA
            self.B += self.eta * dB
            self.prev_dd = dd
            # DSR is scale-free, O(0.1) per step; the shadow cost is converted at the same
            # rate a log-return would be (x scale), then brought to DSR units (x 0.1)
            return float(np.clip(out, -10, 10)) * 0.1 - shadow * self.scale * 0.1
        if self.kind == "log":
            out = lr
        elif self.kind == "turnover":
            out = lr - (self.kappa or 0.002) * turnover
            shadow = 0.0
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
        else:  # cvar
            var_loss = -float(np.quantile(past, self.cvar_alpha)) if len(past) >= 20 else 0.02
            excess = max(-r_net - var_loss, 0.0)
            out = lr - self.lam * excess / self.cvar_alpha
        self.prev_dd = dd
        return (out - shadow) * self.scale

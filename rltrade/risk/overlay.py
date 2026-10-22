"""Deterministic risk overlay: the network is never the last line of defence.

Sits between the policy and the broker. Applied in order:
  1. drawdown kill switch  - if NAV drawdown from peak exceeds `dd_kill`, go to cash
                             for `cooldown` days, then re-arm from the current NAV
  2. per-asset weight cap  - |w_i| <= max_weight
  3. gross / net limits    - sum|w| <= max_gross, min_net <= sum w <= max_net
  4. vol targeting         - scale exposure so ex-ante annual vol <= vol_target
  5. turnover cap          - sum|w - w_current| <= max_turnover per rebalance

Every intervention is reported in the returned flags so it can be logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskOverlay:
    max_weight: float = 0.25
    max_gross: float = 1.0
    max_net: float = 1.0
    min_net: float = -1.0
    vol_target: float | None = 0.15
    max_turnover: float | None = 0.5
    dd_kill: float | None = 0.20
    cooldown: int = 21
    long_only: bool = True
    enabled: bool = True
    _peak: float = field(default=1.0, init=False)
    _killed_for: int = field(default=0, init=False)

    def reset(self, nav: float = 1.0) -> None:
        self._peak = nav
        self._killed_for = 0

    @property
    def killed(self) -> bool:
        return self._killed_for > 0

    def update_nav(self, nav: float) -> None:
        """Call once per step with the latest NAV, before `apply`."""
        if self._killed_for > 0:
            self._killed_for -= 1
            if self._killed_for == 0:
                self._peak = nav  # re-arm from here
            return
        self._peak = max(self._peak, nav)
        if self.dd_kill is not None and 1.0 - nav / self._peak > self.dd_kill:
            self._killed_for = self.cooldown

    def apply(self, target: np.ndarray, current: np.ndarray, cov: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        flags: dict = {}
        w = np.asarray(target, dtype=np.float64).copy()
        if not self.enabled:
            return w, flags
        if self.killed:
            w[:] = 0.0
            flags["kill_switch"] = True
        if self.long_only:
            w = np.maximum(w, 0.0)
        cap = np.clip(w, -self.max_weight, self.max_weight)
        if not np.allclose(cap, w):
            flags["weight_cap"] = True
        w = cap
        gross = np.abs(w).sum()
        if gross > self.max_gross:
            w *= self.max_gross / gross
            flags["gross_limit"] = True
        net = w.sum()
        if net > self.max_net or net < self.min_net:
            bound = self.max_net if net > self.max_net else self.min_net
            # shift via the side that breaches: scale that side's weights down
            side = w > 0 if net > self.max_net else w < 0
            excess = net - bound
            side_sum = w[side].sum()
            if abs(side_sum) > 1e-12:
                w[side] *= 1 - excess / side_sum
            flags["net_limit"] = True
        if self.vol_target is not None and cov is not None and np.abs(w).sum() > 0:
            vol = float(np.sqrt(max(w @ cov @ w, 0.0) * 252))
            if vol > self.vol_target:
                w *= self.vol_target / vol
                flags["vol_target"] = round(vol, 4)
        if self.max_turnover is not None and not self.killed:
            dw = w - current
            to = np.abs(dw).sum()
            if to > self.max_turnover:
                w = current + dw * (self.max_turnover / to)
                flags["turnover_cap"] = round(float(to), 4)
        return w, flags

    @classmethod
    def from_cfg(cls, d: dict | None) -> "RiskOverlay":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and not k.startswith("_")})

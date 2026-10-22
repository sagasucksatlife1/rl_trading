"""Transaction cost model: commission + half spread + square-root market impact.

For a trade of |dw_i| (fraction of NAV) in asset i:

    q_i      = |dw_i| * NAV / price_i               (shares)
    impact_i = Y * sigma_i * sqrt(q_i / V_i)         (fraction of price)
    cost     = sum_i |dw_i| * (commission + half_spread_i + impact_i) * multiplier

sigma_i is daily vol, V_i average daily volume (shares). Both are estimated from
data up to the decision date. `multiplier` is randomised per training episode
(e.g. U[0.5, 3]) so the agent learns a policy that survives cost uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    impact_coef: float = 0.7  # Y in the square-root law; 0.5-1.0 is the usual range
    multiplier: float = 1.0

    def rates(self, dw: np.ndarray, nav: float, price: np.ndarray, adv: np.ndarray,
              sigma: np.ndarray) -> np.ndarray:
        """Per-asset cost per unit of NAV traded."""
        q = np.abs(dw) * nav / np.maximum(price, 1e-12)
        impact = self.impact_coef * sigma * np.sqrt(q / np.maximum(adv, 1.0))
        return ((self.commission_bps + self.half_spread_bps) * 1e-4 + impact) * self.multiplier

    def cost(self, dw: np.ndarray, nav: float, price: np.ndarray, adv: np.ndarray,
             sigma: np.ndarray) -> float:
        """Total cost as a fraction of NAV."""
        return float(np.sum(np.abs(dw) * self.rates(dw, nav, price, adv, sigma)))

    def scaled(self, mult: float) -> "CostModel":
        return CostModel(self.commission_bps, self.half_spread_bps, self.impact_coef, self.multiplier * mult)


def liquidity_inputs(close: np.ndarray, volume: np.ndarray, window: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Causal daily vol and ADV [T, N]: row t uses data up to close t."""
    lr = np.vstack([np.zeros((1, close.shape[1])), np.diff(np.log(close), axis=0)])
    T = close.shape[0]
    sigma = np.empty_like(close)
    adv = np.empty_like(close)
    cs = np.cumsum(np.vstack([np.zeros((1, close.shape[1])), lr]), axis=0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, close.shape[1])), lr**2]), axis=0)
    cv = np.cumsum(np.vstack([np.zeros((1, close.shape[1])), volume]), axis=0)
    for t in range(T):
        lo = max(0, t + 1 - window)
        n = t + 1 - lo
        m = (cs[t + 1] - cs[lo]) / n
        var = (cs2[t + 1] - cs2[lo]) / n - m**2
        sigma[t] = np.sqrt(np.maximum(var, 0.0)) if n > 1 else 0.02
        adv[t] = (cv[t + 1] - cv[lo]) / n
    sigma = np.where(sigma > 0, sigma, 0.02)
    return sigma, adv

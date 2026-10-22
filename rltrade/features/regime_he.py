"""Hurst / entropy regime classifier (calm vs turbulent), causal.

Inputs, all on the equal-weight point-in-time market return, rolling, data to t only:
  hurst_t    variance-ratio Hurst over 126d       (< 0.5 = choppy / mean-reverting)
  pe_t       normalised permutation entropy (order 3) over 63d   (-> 1 = disorder)
  vol_t      21d realised vol

turbulence_t = z(vol) + z(pe) - z(hurst), with z-scores fitted on TRAINING rows only;
turbulent = turbulence above its training `q`-quantile. Thresholds never see test data.
"""
from __future__ import annotations

from itertools import permutations
from math import factorial, log

import numpy as np
import pandas as pd


def permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < order + 5:
        return np.nan
    pats = {p: i for i, p in enumerate(permutations(range(order)))}
    counts = np.zeros(len(pats))
    for i in range(len(x) - order + 1):
        counts[pats[tuple(np.argsort(x[i:i + order], kind="stable"))]] += 1
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / log(factorial(order)))


def regime_inputs(close: pd.DataFrame, tradable: pd.DataFrame | None = None) -> pd.DataFrame:
    r = np.log(close).diff()
    if tradable is not None:
        r = r.where(tradable)
    m = r.mean(axis=1)
    v1 = m.rolling(126).var()
    v10 = m.rolling(10).sum().rolling(126).var()
    hurst = 0.5 + 0.5 * np.log((v10 / (10 * v1)).clip(lower=1e-3)) / np.log(10)
    pe = m.rolling(63).apply(permutation_entropy, raw=True)
    vol = m.rolling(21).std() * np.sqrt(252)
    return pd.DataFrame({"hurst": hurst, "pe": pe, "vol": vol})


class HurstEntropyRegime:
    def __init__(self, q: float = 0.6):
        self.q = q

    def fit(self, X: pd.DataFrame, train_rows: slice) -> "HurstEntropyRegime":
        tr = X.iloc[train_rows].dropna()
        self.mu, self.sd = tr.mean(), tr.std() + 1e-12
        self.thr = float(np.quantile(self._score(tr), self.q))
        return self

    def _score(self, X: pd.DataFrame) -> pd.Series:
        z = (X - self.mu) / self.sd
        return z["vol"] + z["pe"] - z["hurst"]

    def turbulent(self, X: pd.DataFrame) -> np.ndarray:
        s = self._score(X)
        return (s > self.thr).to_numpy() & s.notna().to_numpy()


def regime_labels(close: pd.DataFrame, tradable: pd.DataFrame | None, train_rows: tuple[int, int],
                  q: float = 0.6) -> np.ndarray:
    """Boolean [T] turbulent flag, thresholds fitted on train rows."""
    X = regime_inputs(close, tradable)
    return HurstEntropyRegime(q).fit(X, slice(*train_rows)).turbulent(X)

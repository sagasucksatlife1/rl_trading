"""The feature library shared by training and live.

The anti-leak rule: a feature row at date t uses data up to and including the close
of t, nothing later. Every function here is a causal rolling computation; the
look-ahead test (tests/test_lookahead.py) perturbs the future and asserts the past
features are unchanged.

Output: `FeatureSet` with per-asset features [T, N, F] and global features [T, G].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rltrade.data.synthetic import Market

ANN = 252


@dataclass
class FeatureSet:
    dates: pd.DatetimeIndex
    assets: list[str]
    asset: np.ndarray  # [T, N, F]
    asset_names: list[str]
    glob: np.ndarray  # [T, G]
    glob_names: list[str]
    extra: dict = field(default_factory=dict)

    def add_asset_feature(self, name: str, values: np.ndarray) -> None:
        self.asset = np.concatenate([self.asset, values[:, :, None]], axis=2)
        self.asset_names.append(name)

    def add_global_feature(self, name: str, values: np.ndarray) -> None:
        self.glob = np.concatenate([self.glob, values[:, None]], axis=1)
        self.glob_names.append(name)

    def first_valid(self) -> int:
        """First row where every feature is finite."""
        ok = np.isfinite(self.asset).all(axis=(1, 2)) & np.isfinite(self.glob).all(axis=1)
        return int(np.argmax(ok)) if ok.any() else len(ok)


def _hurst_vr(r: pd.DataFrame, window: int = 126, lag: int = 10) -> pd.DataFrame:
    """Rolling Hurst exponent from the variance ratio of lag-sums to 1-day returns.

    H = 0.5 + 0.5 * log(VR) / log(lag); 0.5 = random walk, >0.5 trending,
    <0.5 mean-reverting.
    """
    v1 = r.rolling(window).var()
    vk = r.rolling(lag).sum().rolling(window).var()
    vr = (vk / (lag * v1)).clip(lower=1e-3)
    return 0.5 + 0.5 * np.log(vr) / np.log(lag)


def asset_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lc = np.log(close)
    r = lc.diff()
    rv21 = r.rolling(21).std() * np.sqrt(ANN)
    feats = {
        "ret_1": r,
        "ret_5": lc.diff(5),
        "ret_21": lc.diff(21),
        "ret_63": lc.diff(63),
        "mom_12_1": lc.shift(21) - lc.shift(252),
        "rvol_21": rv21,
        "rvol_63": r.rolling(63).std() * np.sqrt(ANN),
        "volvol_63": rv21.rolling(63).std() / (rv21.rolling(63).mean() + 1e-12),
        "vol_surprise": np.log(volume / volume.rolling(63).mean()),
        "hurst_126": _hurst_vr(r),
        "dist_hi_252": lc - lc.rolling(252).max(),
        "ret_21_volnorm": lc.diff(21) / (rv21 * np.sqrt(21 / ANN) + 1e-12),
    }
    return feats


def global_features(close: pd.DataFrame) -> dict[str, pd.Series]:
    r = np.log(close).diff()
    mkt = r.mean(axis=1)
    sd = r.rolling(63).std()
    port_var = mkt.rolling(63).var()
    n = r.shape[1]
    sum_var = (sd**2).sum(axis=1)
    sum_sd = sd.sum(axis=1)
    avg_corr = (port_var * n**2 - sum_var) / (sum_sd**2 - sum_var + 1e-18)
    mvol = mkt.rolling(21).std() * np.sqrt(ANN)
    return {
        "mkt_ret_21": mkt.rolling(21).sum(),
        "mkt_ret_63": mkt.rolling(63).sum(),
        "mkt_vol_21": mvol,
        "mkt_vol_ratio": mvol / (mkt.rolling(252).std() * np.sqrt(ANN) + 1e-12),
        "avg_corr_63": avg_corr.clip(-1, 1),
        "dispersion_21": r.rolling(21).sum().std(axis=1),
    }


def build_features(market: Market) -> FeatureSet:
    af = asset_features(market.close, market.volume)
    gf = global_features(market.close)
    names = list(af)
    A = np.stack([af[k].to_numpy(dtype=np.float64) for k in names], axis=2)
    gnames = list(gf)
    G = np.column_stack([gf[k].to_numpy(dtype=np.float64) for k in gnames])
    return FeatureSet(market.dates, market.assets, A, names, G, gnames)


class Normalizer:
    """Z-score with statistics fitted on the training window only, then clipped."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.mu_a = self.sd_a = self.mu_g = self.sd_g = None

    def fit(self, fs: FeatureSet, rows: slice | np.ndarray) -> "Normalizer":
        A = fs.asset[rows]
        G = fs.glob[rows]
        self.mu_a = np.nanmean(A, axis=(0, 1))
        self.sd_a = np.nanstd(A, axis=(0, 1)) + 1e-8
        self.mu_g = np.nanmean(G, axis=0)
        self.sd_g = np.nanstd(G, axis=0) + 1e-8
        return self

    def transform(self, fs: FeatureSet) -> FeatureSet:
        za = np.clip((fs.asset - self.mu_a) / self.sd_a, -self.clip, self.clip)
        zg = np.clip((fs.glob - self.mu_g) / self.sd_g, -self.clip, self.clip)
        return FeatureSet(fs.dates, fs.assets, np.nan_to_num(za), list(fs.asset_names),
                          np.nan_to_num(zg), list(fs.glob_names), dict(fs.extra))

    def state_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mu_a", "sd_a", "mu_g", "sd_g")} | {"clip": self.clip}

    @classmethod
    def from_state(cls, d: dict) -> "Normalizer":
        n = cls(d["clip"])
        for k in ("mu_a", "sd_a", "mu_g", "sd_g"):
            setattr(n, k, np.asarray(d[k]))
        return n

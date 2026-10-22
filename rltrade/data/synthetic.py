"""Synthetic markets.

Four generators, each returning a `Market` (open/close/volume panels):

* `planted_signal_market` - weak, known predictable component. The agent MUST find it.
* `noise_market`          - pure random walk. The agent MUST earn ~0 minus costs.
* `regime_garch_market`   - factor structure + 2-state regime switching GARCH(1,1).
                            Default stand-in for real data when offline.
* `block_bootstrap`       - stationary block bootstrap of a real market's joint daily
                            rows: thousands of plausible histories instead of one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Market:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    name: str = "market"
    # Only set for synthetic markets: the true expected next-day return (for diagnostics).
    true_alpha: pd.DataFrame | None = None
    # Point-in-time tradability [dates x assets]: index member AND listed on that date.
    # None = every asset tradable every day (legacy / synthetic markets).
    tradable: pd.DataFrame | None = None
    # Snapshot manifest (hash, source, pit flag, ...) when loaded from data/snapshots/.
    meta: dict | None = None

    @property
    def assets(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def subset(self, assets: list[str] | None = None, start=None, end=None) -> "Market":
        sl = slice(start, end)
        cols = assets or self.assets
        pick = lambda df: None if df is None else df.loc[sl, cols]
        return Market(pick(self.open), pick(self.close), pick(self.volume), self.name, pick(self.true_alpha),
                      pick(self.tradable), self.meta)

    def tradable_mask(self) -> np.ndarray:
        if self.tradable is None:
            return np.ones(self.close.shape, dtype=bool)
        return self.tradable.to_numpy(dtype=bool)

    def data_hash(self) -> str:
        if self.meta and self.meta.get("hash"):
            return self.meta["hash"][:10]
        import hashlib

        h = hashlib.sha1()
        for df in (self.open, self.close, self.volume):
            h.update(np.ascontiguousarray(df.to_numpy(dtype=np.float64)).tobytes())
        return h.hexdigest()[:10]


def _dates(n_days: int, start: str = "2005-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_days)


def _assemble(log_ret: np.ndarray, rng: np.random.Generator, name: str, start_price: float = 100.0,
              alpha: np.ndarray | None = None, base_volume: float = 2e6) -> Market:
    """Close-to-close log returns [T, N] -> open/close/volume panels.

    The open is the previous close times a small overnight move, so open-to-open and
    close-to-close returns differ slightly, as in real data.
    """
    T, N = log_ret.shape
    cols = [f"A{i:02d}" for i in range(N)]
    idx = _dates(T)
    close = start_price * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([np.full((1, N), start_price), close[:-1]])
    # Overnight gap carries a share of the day's move.
    gap = 0.3 * log_ret + rng.normal(0, 0.002, size=(T, N))
    open_ = prev_close * np.exp(gap)
    vol_scale = np.abs(log_ret) / (np.abs(log_ret).mean(axis=0, keepdims=True) + 1e-12)
    adv = base_volume * np.exp(rng.normal(0, 0.5, size=(1, N)))
    volume = adv * (0.6 + 0.4 * vol_scale) * np.exp(rng.normal(0, 0.25, size=(T, N)))
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return Market(mk(open_), mk(close), mk(volume), name, None if alpha is None else mk(alpha))


def noise_market(n_assets: int = 10, n_days: int = 2500, daily_vol: float = 0.015, seed: int = 0) -> Market:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, daily_vol, size=(n_days, n_assets))
    return _assemble(r, rng, "noise")


def planted_signal_market(n_assets: int = 10, n_days: int = 2500, daily_vol: float = 0.015,
                          ic: float = 0.08, persistence: float = 0.97, seed: int = 0) -> Market:
    """Each asset has a slow AR(1) alpha that leaks into its own past returns.

    The predictable share of next-day variance is ic**2 (~0.6% at ic=0.08): weak,
    like real markets, but findable. The alpha is observable only through price
    history (momentum-like), so a correct pipeline can learn it from features.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((n_days, n_assets))
    eps = rng.normal(size=(n_days, n_assets))
    for t in range(1, n_days):
        a[t] = persistence * a[t - 1] + np.sqrt(1 - persistence**2) * eps[t]
    a_prev = np.vstack([np.zeros((1, n_assets)), a[:-1]])
    alpha = ic * daily_vol * a_prev  # expected return for day t, fixed at t-1
    r = alpha + rng.normal(0, daily_vol * np.sqrt(1 - ic**2), size=(n_days, n_assets))
    return _assemble(r, rng, "planted", alpha=alpha)


def regime_garch_market(n_assets: int = 20, n_days: int = 4000, n_factors: int = 3, seed: int = 0,
                        signal_ic: float = 0.05) -> Market:
    """Factor model with regime-switching GARCH(1,1) factor vols and idiosyncratic GARCH.

    Regimes: calm (low vol, positive drift) and turbulent (high vol, negative drift,
    higher correlation). A weak cross-sectional momentum alpha is planted so that a
    supervised model has something real (and small) to learn.
    """
    rng = np.random.default_rng(seed)
    P = np.array([[0.995, 0.005], [0.02, 0.98]])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = rng.choice(2, p=P[regime[t - 1]])
    drift = np.where(regime == 0, 0.0004, -0.0008)
    corr_mult = np.where(regime == 0, 1.0, 1.8)
    loadings = rng.normal(0.0, 0.5, size=(n_factors, n_assets))
    loadings[0] = rng.uniform(0.6, 1.4, size=n_assets)  # market factor

    def garch(n, omega, a, b, scale):
        # GARCH recursion runs on the unscaled shock; the regime multiplier scales the
        # output only, so the process stays stationary (a + b < 1) in both regimes.
        h = np.full(n, omega / (1 - a - b))
        z = rng.normal(size=n)
        e = np.zeros(n)
        for t in range(n):
            if t:
                h[t] = omega + a * e[t - 1] ** 2 + b * h[t - 1]
            e[t] = np.sqrt(h[t]) * z[t]
        return e * scale

    f = np.column_stack([garch(n_days, 1.5e-6, 0.08, 0.9, corr_mult) for _ in range(n_factors)])
    idio = np.column_stack([garch(n_days, 3e-6, 0.06, 0.92, np.ones(n_days)) for _ in range(n_assets)])
    base = drift[:, None] * loadings[0][None, :] + f @ loadings + idio
    # Planted cross-sectional momentum: 21d past return rank predicts next day weakly.
    r = base.copy()
    alpha = np.zeros_like(r)
    for t in range(22, n_days):
        past = r[t - 21:t].sum(axis=0)
        z = (past - past.mean()) / (past.std() + 1e-12)
        alpha[t] = signal_ic * z * base[t - 63 if t >= 63 else 0:t].std(axis=0).mean()
        r[t] += alpha[t]
    return _assemble(r, rng, "regime_garch", alpha=alpha)


def block_bootstrap(market: Market, n_days: int | None = None, mean_block: int = 21, seed: int = 0) -> Market:
    """Stationary bootstrap (Politis-Romano) of joint daily rows.

    Resamples (close-to-close log return, overnight gap, log volume) jointly across
    all assets, so cross-asset correlation and vol-volume links survive. Block lengths
    are geometric with mean `mean_block`, keeping short-range autocorrelation.
    """
    rng = np.random.default_rng(seed)
    c = market.close.to_numpy()
    o = market.open.to_numpy()
    v = market.volume.to_numpy()
    lr = np.diff(np.log(c), axis=0)
    gap = np.log(o[1:] / c[:-1])
    lv = np.log(v[1:])
    T = lr.shape[0]
    n_days = n_days or T
    idx = np.empty(n_days, dtype=int)
    i = rng.integers(T)
    for t in range(n_days):
        idx[t] = i
        i = rng.integers(T) if rng.random() < 1.0 / mean_block else (i + 1) % T
    lr_s, gap_s, lv_s = lr[idx], gap[idx], lv[idx]
    close = c[0] * np.exp(np.cumsum(lr_s, axis=0))
    prev = np.vstack([c[:1], close[:-1]])
    open_ = prev * np.exp(gap_s)
    dates = _dates(n_days, str(market.dates[0].date()))
    mk = lambda a: pd.DataFrame(a, index=dates, columns=market.assets)
    return Market(mk(open_), mk(close), mk(np.exp(lv_s)), f"{market.name}_boot{seed}")

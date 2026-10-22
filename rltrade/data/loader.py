"""Market data loading.

Sources (config `data.source`):
  yfinance  - download adjusted OHLCV, cached to data_cache/ as versioned snapshots
  csv       - three wide CSVs (open/close/volume, dates x tickers) in `data.csv_dir`
  synthetic - `regime_garch` / `planted` / `noise` generators (offline default)

Production note: yfinance is NOT point-in-time or survivorship-free. For live use,
swap in a vendor feed that is, keeping the `Market` interface unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rltrade.config import ROOT, cfg_get
from rltrade.data.synthetic import Market, noise_market, planted_signal_market, regime_garch_market

log = logging.getLogger(__name__)

# 50-name multi-asset universe: equity indices, sectors, rates, credit, commodities,
# FX proxies, and liquid single names. Swap in your own list via `data.tickers`.
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VGK", "EWJ", "FXI", "INDA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "VNQ",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB", "BND", "MUB", "AGG",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "CPER", "UUP", "FXE", "FXY",
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "NVDA", "BRK-B",
]


def _clean(df: pd.DataFrame, min_coverage: float = 0.98) -> pd.DataFrame:
    keep = df.columns[df.notna().mean() >= min_coverage]
    return df[keep].ffill().dropna()


def load_yfinance(tickers: list[str], start: str, end: str | None, cache_dir: Path) -> Market:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = "yf_" + hashlib.sha1(json.dumps([sorted(tickers), start, end]).encode()).hexdigest()[:10]
    f_close = cache_dir / f"{key}_close.csv"
    if f_close.exists():
        rd = lambda n: pd.read_csv(cache_dir / f"{key}_{n}.csv", index_col=0, parse_dates=True)
        return Market(rd("open"), rd("close"), rd("volume"), "yfinance")
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    close = _clean(raw["Close"])
    cols = close.columns
    open_ = raw["Open"][cols].reindex(close.index).ffill()
    volume = raw["Volume"][cols].reindex(close.index).ffill().replace(0, np.nan).ffill().bfill()
    for name, df in (("open", open_), ("close", close), ("volume", volume)):
        df.to_csv(cache_dir / f"{key}_{name}.csv")
    (cache_dir / f"{key}_meta.json").write_text(json.dumps({"tickers": list(cols), "start": start, "end": end}))
    return Market(open_, close, volume, "yfinance")


def load_csv(csv_dir: Path) -> Market:
    rd = lambda n: pd.read_csv(csv_dir / f"{n}.csv", index_col=0, parse_dates=True)
    close = _clean(rd("close"))
    return Market(rd("open")[close.columns].loc[close.index], close,
                  rd("volume")[close.columns].loc[close.index], "csv")


def load_market(cfg: dict) -> Market:
    src = cfg_get(cfg, "data.source", "synthetic")
    seed = cfg_get(cfg, "data.seed", 0)
    n_assets = cfg_get(cfg, "data.n_assets", 20)
    n_days = cfg_get(cfg, "data.n_days", 4000)
    if src == "yfinance":
        try:
            m = load_yfinance(cfg_get(cfg, "data.tickers") or DEFAULT_UNIVERSE,
                              cfg_get(cfg, "data.start", "2007-01-01"), cfg_get(cfg, "data.end"),
                              ROOT / cfg_get(cfg, "data.cache_dir", "data_cache"))
            log.info("loaded %d assets x %d days from yfinance", len(m.assets), len(m.dates))
            return m
        except Exception as e:  # network blocked, rate limited, etc.
            if not cfg_get(cfg, "data.fallback_to_synthetic", True):
                raise
            log.warning("yfinance failed (%s); falling back to synthetic regime_garch market", e)
            src = "regime_garch"
    if src == "csv":
        return load_csv(ROOT / cfg_get(cfg, "data.csv_dir", "data_cache/csv"))
    if src in ("synthetic", "regime_garch"):
        return regime_garch_market(n_assets=n_assets, n_days=n_days, seed=seed,
                                   signal_ic=cfg_get(cfg, "data.signal_ic", 0.05))
    if src == "planted":
        return planted_signal_market(n_assets=n_assets, n_days=n_days, seed=seed,
                                     ic=cfg_get(cfg, "data.signal_ic", 0.15))
    if src == "noise":
        return noise_market(n_assets=n_assets, n_days=n_days, seed=seed)
    raise ValueError(f"unknown data.source {src!r}")

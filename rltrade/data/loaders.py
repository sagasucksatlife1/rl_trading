"""Point-in-time data pipeline: raw vendor tables -> hashed snapshot -> adjusted Market.

Three raw tables, all UNADJUSTED, all long format (vendor-agnostic schema):

  prices      date, symbol, open, high, low, close, volume
  actions     date, symbol, type, value, ref_symbol
                type = split      value = new shares per old share (2-for-1 -> 2.0), on the ex-date
                type = dividend   value = cash per share, on the ex-date
                type = delist     value = delisting return (CRSP DLRET style) realised on that date
                type = cash_merger  value = cash per share paid; the target delists that day
                type = stock_merger value = acquirer shares per target share; ref_symbol = acquirer
  membership  symbol, start, end      dated index/universe membership intervals; end empty = still in

Survivorship-bias-free means: the price and action tables contain every name that
was EVER in the universe (dead ones included), and membership says WHEN.

Adjustment (`build_market`): a total-return index per symbol, built forward:
    R_t   = split_t * (close_t + div_t) / close_{t-1} - 1          ordinary day
    R_d   = delisting return                                      delist / merger day
    A_t   = A_{t-1} * (1 + R_t);  open_adj_t = A_t * open_t / close_t
    vol_adj_t = dollar_volume_t / close_adj_t   (keeps q / ADV = notional / dollar volume)
After delisting the adjusted price is held flat (the position became cash).

Tradability at decision date t = member at t AND listed at t. A name that delists at
t+1 is still tradable at t: the agent does not know the future, it eats the delisting
return, exactly as a real book would.

Snapshots (`write_snapshot` / `load_snapshot`): the three raw tables are canonicalised
(sorted, typed) and hashed together with ADJUSTMENT_VERSION. Files land in
data/snapshots/<hash>/{prices,actions,membership}.parquet + manifest.json. A run
records the hash; `load_snapshot(hash)` re-hashes on load and refuses on mismatch.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rltrade.config import ROOT, cfg_get
from rltrade.data.synthetic import Market, noise_market, planted_signal_market, regime_garch_market

log = logging.getLogger(__name__)

ADJUSTMENT_VERSION = "pit-adj-v1"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"

PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]
ACTION_COLS = ["date", "symbol", "type", "value", "ref_symbol"]
MEMBER_COLS = ["symbol", "start", "end"]
ACTION_TYPES = {"split", "dividend", "delist", "cash_merger", "stock_merger"}

# 50-name multi-asset universe (yfinance adapter only - NOT survivorship-free).
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VGK", "EWJ", "FXI", "INDA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "VNQ",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB", "BND", "MUB", "AGG",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "CPER", "UUP", "FXE", "FXY",
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "NVDA", "BRK-B",
]


# ----------------------------------------------------------------------------- raw tables
@dataclass
class RawData:
    prices: pd.DataFrame
    actions: pd.DataFrame
    membership: pd.DataFrame
    source: str = "unknown"
    pit: bool = True  # False = NOT survivorship-free (e.g. yfinance); flagged in every report

    def canonical(self) -> "RawData":
        p = self.prices[PRICE_COLS].copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        p["symbol"] = p["symbol"].astype(str)
        for c in PRICE_COLS[2:]:
            p[c] = p[c].astype("float64")
        p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
        a = self.actions.reindex(columns=ACTION_COLS).copy()
        a["date"] = pd.to_datetime(a["date"]).dt.normalize()
        a["symbol"] = a["symbol"].astype(str)
        a["type"] = a["type"].astype(str)
        a["value"] = a["value"].astype("float64")
        a["ref_symbol"] = a["ref_symbol"].fillna("").astype(str)
        bad = set(a["type"]) - ACTION_TYPES
        if bad:
            raise ValueError(f"unknown action types {bad}")
        a = a.sort_values(["symbol", "date", "type"]).reset_index(drop=True)
        m = self.membership[MEMBER_COLS].copy()
        m["symbol"] = m["symbol"].astype(str)
        m["start"] = pd.to_datetime(m["start"]).dt.normalize()
        m["end"] = pd.to_datetime(m["end"]).dt.normalize()
        m = m.sort_values(["symbol", "start"]).reset_index(drop=True)
        return RawData(p, a, m, self.source, self.pit)

    def content_hash(self) -> str:
        c = self.canonical()
        h = hashlib.sha256(ADJUSTMENT_VERSION.encode())
        for df in (c.prices, c.actions, c.membership):
            buf = io.StringIO()
            df.to_csv(buf, index=False, float_format="%.10g", date_format="%Y-%m-%d")
            h.update(buf.getvalue().encode())
        return h.hexdigest()


def validate(raw: RawData) -> list[str]:
    """Data-quality checks. Returns a list of problems (empty = clean)."""
    c = raw.canonical()
    issues = []
    p = c.prices
    if p.duplicated(["symbol", "date"]).any():
        issues.append("duplicate (symbol, date) price rows")
    if (p[["open", "close"]] <= 0).any().any():
        issues.append("non-positive open/close prices")
    if ((p["high"] < p[["open", "close"]].max(1) - 1e-9) | (p["low"] > p[["open", "close"]].min(1) + 1e-9)).any():
        issues.append("high/low inconsistent with open/close")
    lr = p.groupby("symbol")["close"].transform(lambda s: np.log(s).diff().abs())
    act_days = set(zip(c.actions["symbol"], c.actions["date"]))
    is_act = np.array([(s, d) in act_days for s, d in zip(p["symbol"], p["date"])], dtype=bool)
    jumps = p[(lr > np.log(1.8)).to_numpy() & ~is_act]
    if len(jumps):
        issues.append(f"{len(jumps)} unexplained >80% close-to-close moves (missing split/action?)")
    unknown = set(c.membership["symbol"]) - set(p["symbol"])
    if unknown:
        issues.append(f"members with no prices: {sorted(unknown)[:5]}")
    return issues


# ----------------------------------------------------------------------------- adjustment
def build_market(raw: RawData, start=None, end=None) -> Market:
    c = raw.canonical()
    dates = pd.DatetimeIndex(sorted(c.prices["date"].unique()))
    if start is not None or end is not None:
        dates = dates[(dates >= pd.Timestamp(start or dates[0])) & (dates <= pd.Timestamp(end or dates[-1]))]
    symbols = sorted(c.membership["symbol"].unique())
    T, N = len(dates), len(symbols)
    wide = {k: c.prices.pivot(index="date", columns="symbol", values=k).reindex(index=dates, columns=symbols)
            for k in ("open", "close", "volume")}
    O, C, V = (wide[k].to_numpy() for k in ("open", "close", "volume"))
    split = np.ones((T, N))
    div = np.zeros((T, N))
    delist_ret = np.full((T, N), np.nan)
    pos = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(symbols)}
    prices_by_sym = {s: g.set_index("date")["close"] for s, g in c.prices.groupby("symbol")}
    for r in c.actions.itertuples(index=False):
        if r.symbol not in col or r.date not in pos:
            continue
        i, j = pos[r.date], col[r.symbol]
        if r.type == "split":
            split[i, j] *= r.value
        elif r.type == "dividend":
            div[i, j] += r.value
        elif r.type in ("delist", "cash_merger", "stock_merger"):
            prev = prices_by_sym[r.symbol]
            prev = prev[prev.index < r.date]
            last = float(prev.iloc[-1]) if len(prev) else np.nan
            if r.type == "delist":
                delist_ret[i, j] = r.value
            elif r.type == "cash_merger":
                delist_ret[i, j] = r.value / last - 1.0
            else:
                acq = prices_by_sym.get(r.ref_symbol)
                acq_px = float(acq.get(r.date, np.nan)) if acq is not None else np.nan
                delist_ret[i, j] = r.value * acq_px / last - 1.0
    A = np.full((T, N), np.nan)
    Ao = np.full((T, N), np.nan)
    listed = np.zeros((T, N), dtype=bool)
    for j in range(N):
        valid = np.isfinite(C[:, j])
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        dl = np.where(np.isfinite(delist_ret[:, j]))[0]
        last = int(dl[0]) if len(dl) else int(np.max(np.where(valid)[0]))
        a = C[first, j]
        A[first, j] = a
        Ao[first, j] = O[first, j] if np.isfinite(O[first, j]) else a
        prev_c = C[first, j]
        for t in range(first + 1, last + 1):
            if np.isfinite(delist_ret[t, j]):
                a *= 1.0 + delist_ret[t, j]
                A[t, j] = a
                Ao[t, j] = a  # proceeds (cash) from the delisting date on
                break
            if not np.isfinite(C[t, j]):  # missing day: carry flat
                A[t, j], Ao[t, j] = a, a
                continue
            r = split[t, j] * (C[t, j] + div[t, j]) / prev_c - 1.0
            a *= 1.0 + r
            A[t, j] = a
            Ao[t, j] = a * (O[t, j] / C[t, j] if np.isfinite(O[t, j]) else 1.0)
            prev_c = C[t, j]
        listed[first:last + 1, j] = True
        if np.isfinite(delist_ret[last, j]):
            listed[last, j] = False  # not tradable on the day it ceases to exist
    member = np.zeros((T, N), dtype=bool)
    for r in c.membership.itertuples(index=False):
        j = col[r.symbol]
        end_ = r.end if pd.notna(r.end) else dates[-1] + pd.Timedelta(days=1)
        member[:, j] |= (dates >= r.start) & (dates < end_)
    tradable = member & listed
    dollar_vol = V * C
    # fill before listing (back-fill) and after delisting (flat) so env arithmetic is finite
    Adf = pd.DataFrame(A, index=dates, columns=symbols).ffill().bfill()
    Aodf = pd.DataFrame(Ao, index=dates, columns=symbols).ffill().bfill()
    vadj = pd.DataFrame(dollar_vol / np.where(np.isfinite(A), A, np.nan), index=dates, columns=symbols)
    vadj = vadj.where(np.isfinite(vadj) & (vadj > 0)).ffill().bfill().fillna(1.0)
    keep = tradable.any(axis=0)
    cols = [s for s, k in zip(symbols, keep) if k]
    meta = {"source": raw.source, "pit": raw.pit, "adjustment": ADJUSTMENT_VERSION,
            "n_symbols_ever": N, "n_tradable_ever": int(keep.sum())}
    return Market(Aodf[cols], Adf[cols], vadj[cols], raw.source,
                  tradable=pd.DataFrame(tradable[:, keep], index=dates, columns=cols), meta=meta)


# ----------------------------------------------------------------------------- snapshots
def write_snapshot(raw: RawData, root: Path = SNAPSHOT_DIR) -> str:
    c = raw.canonical()
    h = raw.content_hash()
    d = Path(root) / h[:16]
    if (d / "manifest.json").exists():
        return h
    d.mkdir(parents=True, exist_ok=True)
    c.prices.to_parquet(d / "prices.parquet", index=False)
    c.actions.to_parquet(d / "actions.parquet", index=False)
    c.membership.to_parquet(d / "membership.parquet", index=False)
    issues = validate(c)
    manifest = {"hash": h, "source": raw.source, "pit": raw.pit, "adjustment": ADJUSTMENT_VERSION,
                "rows": {"prices": len(c.prices), "actions": len(c.actions), "membership": len(c.membership)},
                "symbols": int(c.prices["symbol"].nunique()),
                "date_range": [str(c.prices["date"].min().date()), str(c.prices["date"].max().date())],
                "quality_issues": issues}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("snapshot %s written (%s, pit=%s, %d issues)", h[:16], raw.source, raw.pit, len(issues))
    return h


def load_snapshot(h: str, root: Path = SNAPSHOT_DIR, verify: bool = True, start=None, end=None) -> Market:
    cands = [p for p in Path(root).glob("*") if p.name.startswith(h[:16])]
    if not cands:
        raise FileNotFoundError(f"no snapshot {h} under {root}")
    d = cands[0]
    manifest = json.loads((d / "manifest.json").read_text())
    raw = RawData(pd.read_parquet(d / "prices.parquet"), pd.read_parquet(d / "actions.parquet"),
                  pd.read_parquet(d / "membership.parquet"), manifest["source"], manifest["pit"])
    if verify and raw.content_hash() != manifest["hash"]:
        raise ValueError(f"snapshot {d.name} content does not match its hash: refusing to load")
    m = build_market(raw, start, end)
    m.meta = {**m.meta, **manifest}
    if not manifest["pit"]:
        log.warning("snapshot %s is NOT point-in-time / survivorship-free (%s)", d.name, manifest["source"])
    return m


def list_snapshots(root: Path = SNAPSHOT_DIR) -> list[dict]:
    return [json.loads((p / "manifest.json").read_text()) for p in sorted(Path(root).glob("*"))
            if (p / "manifest.json").exists()]


# ----------------------------------------------------------------------------- adapters
def from_csv_dir(d: Path, source: str = "csv", pit: bool = True) -> RawData:
    """Generic schema: prices.csv, actions.csv, membership.csv with the columns above."""
    d = Path(d)
    rd = lambda n: pd.read_csv(d / f"{n}.csv")
    acts = rd("actions") if (d / "actions.csv").exists() else pd.DataFrame(columns=ACTION_COLS)
    return RawData(rd("prices"), acts, rd("membership"), source, pit)


def from_sharadar(sep: Path, actions: Path, membership: Path) -> RawData:
    """Sharadar (Nasdaq Data Link) exports: SEP, ACTIONS, and SP500 (or your own universe table).

    SEP has `closeunadj` (raw) and split-adjusted open/close; we rebuild raw opens as
    open * closeunadj / close. ACTIONS rows map: split -> split, dividend -> dividend,
    delisted -> delist (value 0 unless a DLRET-style return is supplied),
    acquisitionby (cash) -> cash_merger. SP500 add/remove rows become membership
    intervals. Written to Sharadar's published schema; check against your export.
    """
    p = pd.read_csv(sep, parse_dates=["date"])
    raw_close = p["closeunadj"]
    f = raw_close / p["close"]
    prices = pd.DataFrame({"date": p["date"], "symbol": p["ticker"], "open": p["open"] * f,
                           "high": p["high"] * f, "low": p["low"] * f, "close": raw_close,
                           "volume": p["volume"] / f})
    a = pd.read_csv(actions, parse_dates=["date"])
    m = {"split": "split", "dividend": "dividend", "delisted": "delist", "acquisitionby": "cash_merger"}
    a = a[a["action"].isin(m)]
    acts = pd.DataFrame({"date": a["date"], "symbol": a["ticker"], "type": a["action"].map(m),
                         "value": np.where(a["action"] == "delisted", 0.0, a["value"]),
                         "ref_symbol": a.get("contraticker", "")})
    sp = pd.read_csv(membership, parse_dates=["date"]).sort_values("date")
    rows, open_ = [], {}
    for r in sp.itertuples(index=False):
        if r.action in ("added", "historical") and r.ticker not in open_:
            open_[r.ticker] = r.date
        elif r.action == "removed" and r.ticker in open_:
            rows.append((r.ticker, open_.pop(r.ticker), r.date))
    rows += [(t, s, pd.NaT) for t, s in open_.items()]
    return RawData(prices, acts, pd.DataFrame(rows, columns=MEMBER_COLS), "sharadar", True)


def from_yfinance(tickers: list[str], start: str, end: str | None) -> RawData:
    """Current tickers only -> NOT survivorship-free. Raw OHLCV + split/dividend actions."""
    import yfinance as yf

    frames, acts = [], []
    for t in tickers:
        tk = yf.Ticker(t)
        h = tk.history(start=start, end=end, auto_adjust=False, actions=True)
        if h.empty:
            continue
        h.index = h.index.tz_localize(None)
        frames.append(pd.DataFrame({"date": h.index, "symbol": t, "open": h["Open"], "high": h["High"],
                                    "low": h["Low"], "close": h["Close"], "volume": h["Volume"]}))
        for d, v in h["Stock Splits"][h["Stock Splits"] > 0].items():
            acts.append((d, t, "split", v, ""))
        for d, v in h["Dividends"][h["Dividends"] > 0].items():
            acts.append((d, t, "dividend", v, ""))
    prices = pd.concat(frames, ignore_index=True)
    first = prices.groupby("symbol")["date"].min()
    mem = pd.DataFrame({"symbol": first.index, "start": first.values, "end": pd.NaT})
    return RawData(prices, pd.DataFrame(acts, columns=ACTION_COLS), mem, "yfinance", pit=False)


def synthetic_pit(n_symbols: int = 30, universe_size: int = 20, n_days: int = 3000, seed: int = 0,
                  signal_ic: float = 0.05) -> tuple[RawData, dict]:
    """A raw, UNADJUSTED universe with known corporate actions, for testing the pipeline.

    Starts from the regime-GARCH market (true total returns), then injects splits,
    dividends (price drops by the dividend on the ex-date), delistings with a known
    return, a cash merger, a stock merger, late listings, and index adds/removes.
    Returns (raw, truth) where truth['tr'] is the true total-return close panel.
    """
    rng = np.random.default_rng(seed)
    m = regime_garch_market(n_assets=n_symbols, n_days=n_days, seed=seed, signal_ic=signal_ic)
    dates = m.dates
    syms = [f"S{i:03d}" for i in range(n_symbols)]
    tr_close = m.close.to_numpy().copy()
    tr_open = m.open.to_numpy().copy()
    vol = m.volume.to_numpy().copy()
    T = len(dates)
    raw_c = np.full((T, n_symbols), np.nan)
    raw_o = np.full((T, n_symbols), np.nan)
    acts, truth = [], {"splits": [], "delists": [], "mergers": []}
    first = np.zeros(n_symbols, dtype=int)
    last = np.full(n_symbols, T - 1)
    late = rng.choice(n_symbols, size=n_symbols // 6, replace=False)
    first[late] = rng.integers(300, T // 2, size=len(late))
    dead = rng.choice(np.setdiff1d(np.arange(n_symbols), late), size=n_symbols // 6, replace=False)
    last[dead] = rng.integers(T // 2, T - 100, size=len(dead))
    for j in range(n_symbols):
        f_split = 1.0  # cumulative shares per original share
        cash_div = 0.0
        ex_div = set(range(first[j] + 63, last[j], 63))
        split_days = set(rng.choice(np.arange(first[j] + 100, max(last[j] - 50, first[j] + 101)),
                                    size=rng.integers(0, 3), replace=False).tolist())
        for t in range(first[j], last[j] + 1):
            if t in split_days and t > first[j]:
                ratio = float(rng.choice([2.0, 3.0, 0.5]))
                f_split *= ratio
                acts.append((dates[t], syms[j], "split", ratio, ""))
                truth["splits"].append((syms[j], str(dates[t].date()), ratio))
            if t in ex_div and t > first[j]:
                # dividend = 0.5% of the raw price; raw price falls by it on the ex-date
                px = tr_close[t, j] / f_split * np.exp(-cash_div)
                d = 0.005 * px
                acts.append((dates[t], syms[j], "dividend", d, ""))
                cash_div += np.log(1 + d / (px - d))
            scale = np.exp(-cash_div) / f_split
            raw_c[t, j] = tr_close[t, j] * scale
            raw_o[t, j] = tr_open[t, j] * scale
    # delistings: with a known delisting return; two of them as mergers
    for k, j in enumerate(dead):
        t = last[j]
        if k == 0:
            cash = raw_c[t - 1, j] * 1.25
            acts.append((dates[t], syms[j], "cash_merger", cash, ""))
            truth["mergers"].append((syms[j], "cash", 0.25))
        elif k == 1:
            acq = int(np.setdiff1d(np.arange(n_symbols), np.concatenate([dead, late]))[0])
            ratio = raw_c[t - 1, j] * 1.10 / raw_c[t, acq]
            acts.append((dates[t], syms[j], "stock_merger", ratio, syms[acq]))
            truth["mergers"].append((syms[j], "stock", 0.10))
        else:
            r = float(rng.choice([-0.3, -0.9, -1.0]))
            acts.append((dates[t], syms[j], "delist", r, ""))
            truth["delists"].append((syms[j], r))
        raw_c[t, j] = np.nan  # no trade on the delist day itself
        raw_o[t, j] = np.nan
    rows = []
    for j in range(n_symbols):
        for t in range(first[j], last[j] + 1):
            if np.isfinite(raw_c[t, j]):
                hi = max(raw_o[t, j], raw_c[t, j]) * 1.005
                lo = min(raw_o[t, j], raw_c[t, j]) * 0.995
                rows.append((dates[t], syms[j], raw_o[t, j], hi, lo, raw_c[t, j],
                             vol[t, j] * tr_close[t, j] / raw_c[t, j]))
    prices = pd.DataFrame(rows, columns=PRICE_COLS)
    # membership: start with `universe_size` names; swap names in/out every ~250 days
    mem, current = [], {}
    order = [j for j in range(n_symbols) if first[j] == 0]
    for j in order[:universe_size]:
        current[j] = dates[0]
    reserve = [j for j in range(n_symbols) if j not in current]
    for t in range(250, T, 250):
        for j in list(current):
            if last[j] < T - 1 and last[j] <= t:  # dead names leave at their delisting
                mem.append((syms[j], current.pop(j), dates[last[j]]))
        while len(current) < universe_size and reserve:
            cand = [j for j in reserve if first[j] <= t and last[j] > t]
            if not cand:
                break
            j = cand[0]
            reserve.remove(j)
            current[j] = dates[t]
        if len(current) == universe_size and reserve and rng.random() < 0.5:
            out = sorted(current)[int(rng.integers(len(current)))]
            cand = [j for j in reserve if first[j] <= t and last[j] > t]
            if cand:
                mem.append((syms[out], current.pop(out), dates[t]))
                reserve.append(out)
                j = cand[0]
                reserve.remove(j)
                current[j] = dates[t]
    mem += [(syms[j], s, pd.NaT) for j, s in current.items()]
    truth["tr_close"] = pd.DataFrame(tr_close, index=dates, columns=syms)
    truth["first"] = dict(zip(syms, first.tolist()))
    truth["last"] = dict(zip(syms, last.tolist()))
    raw = RawData(prices, pd.DataFrame(acts, columns=ACTION_COLS), pd.DataFrame(mem, columns=MEMBER_COLS),
                  "synthetic_pit", True)
    return raw, truth


# ----------------------------------------------------------------------------- config entry
def load_market(cfg: dict) -> Market:
    """Every non-legacy source goes through a hashed snapshot, so a run is reproducible
    from its data hash. Legacy synthetic generators (regime_garch/planted/noise) are
    deterministic in (config, seed) and are hashed from their arrays."""
    src = cfg_get(cfg, "data.source", "synthetic")
    seed = cfg_get(cfg, "data.seed", 0)
    start, end = cfg_get(cfg, "data.start"), cfg_get(cfg, "data.end")
    raw = None
    if src == "snapshot":
        return load_snapshot(cfg_get(cfg, "data.snapshot"), start=start, end=end)
    if src == "csv_pit":
        raw = from_csv_dir(ROOT / cfg_get(cfg, "data.csv_dir"), pit=cfg_get(cfg, "data.pit", True))
    elif src == "sharadar":
        d = ROOT / cfg_get(cfg, "data.sharadar_dir")
        raw = from_sharadar(d / "SEP.csv", d / "ACTIONS.csv", d / cfg_get(cfg, "data.membership_file", "SP500.csv"))
    elif src == "synthetic_pit":
        raw, _ = synthetic_pit(cfg_get(cfg, "data.n_symbols", 30), cfg_get(cfg, "data.n_assets", 20),
                               cfg_get(cfg, "data.n_days", 3000), seed, cfg_get(cfg, "data.signal_ic", 0.05))
    elif src == "yfinance":
        try:
            raw = from_yfinance(cfg_get(cfg, "data.tickers") or DEFAULT_UNIVERSE, start or "2007-01-01", end)
        except Exception as e:  # noqa: BLE001 - network blocked / rate limited
            if not cfg_get(cfg, "data.fallback_to_synthetic", True):
                raise
            log.warning("yfinance failed (%s); falling back to synthetic_pit", e)
            raw, _ = synthetic_pit(seed=seed)
    if raw is not None:
        h = write_snapshot(raw)
        return load_snapshot(h, start=start, end=end)
    n_assets, n_days = cfg_get(cfg, "data.n_assets", 20), cfg_get(cfg, "data.n_days", 4000)
    if src in ("synthetic", "regime_garch"):
        return regime_garch_market(n_assets=n_assets, n_days=n_days, seed=seed,
                                   signal_ic=cfg_get(cfg, "data.signal_ic", 0.05))
    if src == "planted":
        return planted_signal_market(n_assets=n_assets, n_days=n_days, seed=seed,
                                     ic=cfg_get(cfg, "data.signal_ic", 0.15))
    if src == "noise":
        return noise_market(n_assets=n_assets, n_days=n_days, seed=seed)
    raise ValueError(f"unknown data.source {src!r}")
